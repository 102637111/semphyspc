#!/usr/bin/env python3
"""用材质单任务模型(UR-MAT 14 类)对街道级点云 street_pc 做逐点材质标注。

背景：
  street_pc 无真值材质标签。用已收敛的材质单任务模型(mIoU 0.9406)对合成街道点云
  推理，得到逐点材质标签，供下游物理材质映射(碰撞安全/可破坏性)使用。

坐标系（关键）：
  street_pc 的 xyz 与 UR-MAT 训练数据同属相机坐标系 (x右/y下/z前)，材质模型即
  在该坐标系下训练，故**直接喂相机系 xyz，不做 lidar 转换**（与语义伪标注不同）。
  语义模型(nuScenes)是 lidar z-up，材质模型(UR-MAT)是 camera 系。

流程：
  1. 加载材质单任务模型(PointMambaSeg num_classes=14)。
  2. 遍历 street_pc/{scene}/*.npz，xyz 为相机系，直接使用。
  3. 采样到 npoints=16384，推理得 (mat_pred, conf)。
  4. 过滤 conf>thresh 的点，写成 {xyz(相机系), rgb, mat, conf}。

用法：
    python scripts/pseudolabel_street_pc_mat.py \
        --street /mnt/data/datasets/UR-MAT/street_pc \
        --out /mnt/data/datasets/UR-MAT/street_pc_mat \
        --mat_cfg experiments/seg_urmat_16k/cfgs/singletask_mat/20260822-063146/config.yaml \
        --mat_ckpt experiments/seg_urmat_16k/cfgs/singletask_mat/20260822-063146/ckpt-best.pth \
        --conf_thresh 0.5 --seed 0
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from easydict import EasyDict

NPOINTS = 16384

# UR-MAT 14 类材质名（与 scripts/build_urmat_dataset.py MATERIAL_NAMES 一致）
MATERIAL_NAMES = None  # 由 build 脚本导入时自动填充，或从 legend 读


def build_model(cfg_path, ckpt_path):
    from models.point_mamba_seg import PointMambaSeg
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = PointMambaSeg(EasyDict(cfg['model']))
    sd = torch.load(ckpt_path, map_location='cpu')
    base = sd.get('base_model') or sd.get('model')
    base = {k.replace('module.', ''): v for k, v in base.items()}
    model.load_state_dict(base, strict=False)
    model.cuda().eval()
    return model


@torch.no_grad()
def infer(model, pts):
    logits, _ = model(pts)
    prob = torch.softmax(logits, dim=-1)
    conf, pred = prob.max(-1)
    return pred[0].cpu().numpy().astype(np.int64), conf[0].cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--street', required=True, help='street_pc 根目录(含 {scene}/*.npz)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mat_cfg', required=True)
    ap.add_argument('--mat_ckpt', required=True)
    ap.add_argument('--conf_thresh', type=float, default=0.5)
    ap.add_argument('--min_keep', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_train = os.path.join(args.out, 'train')
    os.makedirs(out_train, exist_ok=True)

    print(f"[load model] {args.mat_ckpt}", flush=True)
    model = build_model(args.mat_cfg, args.mat_ckpt)

    files = sorted(glob_safe(args.street))
    print(f"[pseudolabel] {len(files)} 个 street 点云 (材质14类, 相机系直接推理)", flush=True)

    n_used = 0
    n_skip = 0
    n_pts_total = 0
    n_pts_keep = 0
    hist_keep = None
    t0 = time.time()

    for fi, fp in enumerate(files):
        scene = os.path.basename(os.path.dirname(fp))
        stem = os.path.splitext(os.path.basename(fp))[0]
        d = np.load(fp)
        xyz = d['xyz'].astype(np.float32)   # 相机系，与 UR-MAT 同帧，直接使用
        rgb = d['rgb'].astype(np.float32)

        N = xyz.shape[0]
        if N > NPOINTS:
            idx = rng.choice(N, NPOINTS, replace=False)
            xyz, rgb = xyz[idx], rgb[idx]

        pts = torch.from_numpy(np.concatenate([xyz, rgb], axis=1)).float().unsqueeze(0).cuda()
        pred, conf = infer(model, pts)

        keep = conf > args.conf_thresh
        if keep.sum() < args.min_keep:
            n_skip += 1
            continue

        out_name = f"street_mat_{scene}_{stem}.npz"
        np.savez_compressed(
            os.path.join(out_train, out_name),
            xyz=xyz[keep].astype(np.float32),
            rgb=rgb[keep].astype(np.float32),
            mat=pred[keep].astype(np.int16),
            conf=conf[keep].astype(np.float32),
        )
        n_used += 1
        n_pts_total += N
        n_pts_keep += int(keep.sum())
        if hist_keep is None:
            hist_keep = np.zeros(14, dtype=np.int64)
        hist_keep += np.bincount(pred[keep], minlength=14)

        if (fi + 1) % 100 == 0:
            print(f"  [{fi+1}/{len(files)}] 已处理, 用时 {time.time()-t0:.0f}s", flush=True)

    print(f"[done] 保留 {n_used}/{len(files)} 样本, 跳过 {n_skip} (不足 {args.min_keep} 点)",
          flush=True)
    print(f"       总点 {n_pts_total}, 保留点 {n_pts_keep} ({n_pts_keep/max(n_pts_total,1)*100:.1f}%)",
          flush=True)
    if hist_keep is not None:
        print(f"       材质类别直方图(14类): {hist_keep.tolist()}", flush=True)

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump({'n_used': n_used, 'n_skip': n_skip,
                   'n_pts_total': n_pts_total, 'n_pts_keep': n_pts_keep,
                   'hist_keep': hist_keep.tolist() if hist_keep is not None else [],
                   'frame': 'camera (x-right/y-down/z-forward, 同 UR-MAT)'}, f, indent=2)
    print(f"[done] 伪标注 -> {out_train}  total {time.time()-t0:.0f}s", flush=True)


def glob_safe(root):
    import glob
    return sorted(glob.glob(os.path.join(root, '*', '*.npz')))


if __name__ == '__main__':
    main()
