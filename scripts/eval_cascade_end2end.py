#!/usr/bin/env python3
"""实验D：端到端级联评测 —— ①地面几何 + ②offset实例分割 + ③B2物体判类 串成完整级联，
量化 thing 类(11个B2覆盖类)语义精度相对端到端 dense 基线的提升，闭合级联闭环。

背景：
  级联三环 = ①地面几何分割 ②实例分割(offset回归) ③物体级判类(B2分类器)。
  三环已各自验证(①95%精度 ②PQ0.50 ③87%)。本脚本把它们串起来，回答最终问题：
  「级联的 thing 类语义精度 vs 端到端 0.2342 mIoU 的 dense 模型，谁更高、高在哪」。

方法（每个 val 场景）：
  1. 一次前向拿到 dense 语义 logits 与逐点 offset。
  2. thing 点 = sem in 11个B2类 且 inst>0（用真值做 thing/stuff 门控，隔离 ②③ 的贡献）。
  3. thing 点按「坐标+预测offset」平移后 DBSCAN 聚类 → 预测实例。
  4. 每个预测实例 crop(居中+单位球归一化+采样1024) → B2分类器 → 判类，回填到实例所有点。
  5. 对比 dense 逐点预测 vs 级联实例判类，在 thing 点上的准确率（整体 + 逐类）。

注意：B2 只覆盖 11 个高频 thing 类(车/行人/护栏/卡车/锥桶/拖车/推车/公交/摩托/施工/自行车)，
     因此级联的 thing 类精度在这 11 类上度量，其余 thing 类(动物/救护车/铰接公交等)不计入。

用法:
    python scripts/eval_cascade_end2end.py \
        --val data/nuscenes_seg_full_16k/val \
        --offset_ckpt experiments/cascade_models/ckpt-offset.pth \
        --cls_ckpt experiments/cascade_models/ckpt-cls.pth \
        --n 500 --eps 0.5 --min_points 3 --seed 0
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.point_mamba_seg import PointMambaSeg
from models.point_mamba_scan import PointMambaScan

# B2 分类器的 11 类（重映射 0..10）→ nuScenes 语义类 id
CLASS2SEM = [17, 2, 9, 23, 12, 22, 11, 16, 21, 18, 14]
SEM2CLASS = {s: i for i, s in enumerate(CLASS2SEM)}
CLASS_NAMES = ['car', 'ped.adult', 'barrier', 'truck', 'trafficcone', 'trailer',
               'pushable', 'bus.rigid', 'motorcycle', 'construction', 'bicycle']


def build_seg(offset_ckpt):
    cfg = EasyDict({
        'NAME': 'PointMambaSeg', 'trans_dim': 384, 'depth': 12,
        'group_size': 32, 'num_group': 64, 'encoder_dims': 384,
        'in_channels': 6, 'num_classes': 32, 'upsample_k': 3,
        'rms_norm': False, 'drop_path': 0.1, 'drop_out': 0.0,
        'num_materials': 0, 'num_instances': 3,
    })
    model = PointMambaSeg(cfg)
    sd = torch.load(offset_ckpt, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.cuda().eval()
    return model


def build_cls(cls_ckpt):
    cfg = EasyDict({
        'NAME': 'PointMambaScan', 'trans_dim': 384, 'depth': 12,
        'cls_dim': len(CLASS2SEM), 'group_size': 32, 'num_group': 64,
        'encoder_dims': 384, 'in_channels': 6, 'rms_norm': False,
        'drop_path': 0.1, 'drop_out': 0.0,
        'use_cls_token': False, 'max_head': False, 'avg_head': True,
    })
    model = PointMambaScan(cfg)
    sd = torch.load(cls_ckpt, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.cuda().eval()
    return model


def prep_crop(xyz, rgb, npoints=1024, seed_rng=None):
    """与 B2 训练 _prep(train=False) 一致：居中 + 单位球 + 采样 npoints。"""
    xyz = xyz.astype(np.float32).copy()
    rgb = rgb.astype(np.float32).copy()
    xyz = xyz - xyz.mean(0)
    norm = np.linalg.norm(xyz, axis=1).max()
    if norm > 1e-6:
        xyz = xyz / norm
    Ni = xyz.shape[0]
    if seed_rng is not None:
        idx = seed_rng.choice(Ni, npoints, replace=(Ni < npoints))
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(Ni, npoints, replace=(Ni < npoints))
    return np.concatenate([xyz[idx], rgb[idx]], axis=1).astype(np.float32)


def cluster_dbscan(xyz, eps, min_points):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points,
                                         print_progress=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', required=True)
    ap.add_argument('--offset_ckpt', required=True)
    ap.add_argument('--cls_ckpt', required=True)
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--eps', type=float, default=0.5)
    ap.add_argument('--min_points', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    files = sorted(glob.glob(os.path.join(args.val, '*.npz')))
    if args.n < len(files):
        files = rng.choice(files, args.n, replace=False).tolist()

    print(f'[expD] 载入 offset 头 + B2 分类器, {len(files)} val 场景', flush=True)
    seg = build_seg(args.offset_ckpt)
    cls = build_cls(args.cls_ckpt)

    # 逐类统计：dense 逐点 vs 级联实例判类
    dense_correct = np.zeros(len(CLASS2SEM), dtype=np.int64)
    dense_total = np.zeros(len(CLASS2SEM), dtype=np.int64)
    casc_correct = np.zeros(len(CLASS2SEM), dtype=np.int64)
    casc_total = np.zeros(len(CLASS2SEM), dtype=np.int64)

    t0 = time.time()
    for fi, f in enumerate(files):
        d = np.load(f)
        xyz = d['xyz'].astype(np.float32)
        rgb = d['rgb'].astype(np.float32)
        sem = d['sem'].astype(np.int64)
        inst = d['inst'].astype(np.int64)

        pts = torch.from_numpy(np.concatenate([xyz, rgb], 1)).float().unsqueeze(0).cuda()
        with torch.no_grad():
            logits, _, off = seg(pts)
        dense_pred = logits[0].argmax(-1).cpu().numpy().astype(np.int64)
        off = off[0].cpu().numpy()  # (N,3)

        # thing 点（11 个 B2 类，inst>0）
        thing = np.isin(sem, list(SEM2CLASS.keys())) & (inst > 0)
        if thing.sum() < args.min_points:
            continue
        thing_idx = np.where(thing)[0]
        sem_t = sem[thing_idx]          # (n_thing,)
        dense_p = dense_pred[thing_idx]  # dense 逐点预测

        # ---- 级联：offset 平移 + DBSCAN 聚类 + B2 判类 ----
        xyz_t = xyz[thing_idx]
        shifted = xyz_t + off[thing_idx]
        pred_inst = cluster_dbscan(shifted, args.eps, args.min_points)
        casc_p = np.full(thing_idx.shape[0], -1, dtype=np.int64)  # -1 = 噪声未分配
        for pid in np.unique(pred_inst[pred_inst >= 0]):
            m = pred_inst == pid
            crop = prep_crop(xyz_t[m], rgb[thing_idx][m], 1024, seed_rng=rng)
            crop_t = torch.from_numpy(crop).float().unsqueeze(0).cuda()
            with torch.no_grad():
                cls_logit = cls(crop_t)
            casc_p[m] = CLASS2SEM[int(cls_logit[0].argmax(-1).item())]

        # 逐类统计：dense vs 级联（未分配点按判错计）
        for c, s in enumerate(CLASS2SEM):
            m = sem_t == s
            dense_total[c] += m.sum()
            dense_correct[c] += (dense_p[m] == s).sum()
            casc_total[c] += m.sum()
            casc_correct[c] += (casc_p[m] == s).sum()

        if (fi + 1) % 200 == 0:
            print(f"  [{fi+1}/{len(files)}] 用时 {time.time()-t0:.0f}s", flush=True)

    print("\n================ 实验D 结果（端到端级联 vs dense 基线）================", flush=True)
    print(f"  场景数 {len(files)}", flush=True)
    print(f"  {'类':<14s} {'dense_acc':>10s} {'cascade_acc':>12s} {'提升':>8s} {'thing点数':>8s}", flush=True)
    for c, name in enumerate(CLASS_NAMES):
        da = dense_correct[c] / max(dense_total[c], 1)
        ca = casc_correct[c] / max(casc_total[c], 1)
        print(f"  {name:<14s} {da:>10.4f} {ca:>12.4f} {ca-da:>+8.4f} {dense_total[c]:>8d}", flush=True)

    da_all = dense_correct.sum() / max(dense_total.sum(), 1)
    ca_all = casc_correct.sum() / max(casc_total.sum(), 1)
    print(f"  {'整体(11类)':<14s} {da_all:>10.4f} {ca_all:>12.4f} {ca_all-da_all:>+8.4f} "
          f"{dense_total.sum():>8d}", flush=True)
    print("  (端到端 dense 模型全 32 类 mIoU = 0.2342，作为参照)", flush=True)
    print("======================================================================", flush=True)


if __name__ == '__main__':
    main()
