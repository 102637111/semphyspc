#!/usr/bin/env python3
"""用语义单任务模型对街道级点云做伪标注，产出可混入 nuScenes 训练的样本。

背景：
  语义模型(32类)在合成城市场景上判别力弱(noise 17.5%)，但 street_pc 无真值语义标签。
  用当前语义模型自训练(self-training)做域适配：对 street_pc 逐点推理得到伪标签，
  保留「高置信 + 非 noise」的点，写成 nuScenes 训练格式 {xyz(lidar系), rgb, sem}，
  供后续 NuscenesStreetAugDataset 混入训练。

流程：
  1. 加载语义单任务模型(PointMambaSeg 32类)。
  2. 遍历 street_pc/{scene}/*.npz，xyz 为相机系(x右/y下/z前)，转 lidar 系
     xyz_lidar = [x, z, -y]（与 label_rendered_scenes.py 一致）。
  3. 采样到 npoints=16384，推理得 (pred, conf)。
  4. 过滤 conf>thresh 且 pred!=0(noise) 的点，写成 {xyz, rgb, sem, conf}。

用法：
    python scripts/pseudolabel_street_pc.py \
        --street /mnt/data/datasets/UR-MAT/street_pc \
        --out /mnt/data/datasets/UR-MAT/street_pc_pseudolabel \
        --sem_cfg experiments/seg_nuscenes_full_16k/cfgs/singletask_sem/20260821-221520/config.yaml \
        --sem_ckpt experiments/seg_nuscenes_full_16k/cfgs/singletask_sem/20260821-221520/ckpt-best.pth \
        --conf_thresh 0.7 --seed 0
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

# 合成城市场景能可靠提供的静态语义类（nuScenes-lidarseg 32 类下标）。
# 车辆/行人/可移动物等对象类在合成场景中不存在，模型会幻觉，故伪标注时一律丢弃。
STATIC_CLASSES = {24, 25, 26, 27, 28, 29, 30}  # drivable/other/sidewalk/terrain/manmade/static_other/vegetation


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
    ap.add_argument('--sem_cfg', required=True)
    ap.add_argument('--sem_ckpt', required=True)
    ap.add_argument('--conf_thresh', type=float, default=0.7)
    ap.add_argument('--min_keep', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_train = os.path.join(args.out, 'train')
    os.makedirs(out_train, exist_ok=True)

    print(f"[load model] {args.sem_ckpt}", flush=True)
    model = build_model(args.sem_cfg, args.sem_ckpt)

    files = sorted(glob_safe(args.street))
    print(f"[pseudolabel] {len(files)} 个 street 点云", flush=True)

    summary = {'conf_thresh': args.conf_thresh, 'min_keep': args.min_keep, 'samples': []}
    n_used = 0
    n_skip = 0
    n_pts_total = 0
    n_pts_keep = 0
    hist_keep = np.zeros(32, dtype=np.int64)
    t0 = time.time()

    for fi, fp in enumerate(files):
        scene = os.path.basename(os.path.dirname(fp))
        stem = os.path.splitext(os.path.basename(fp))[0]
        d = np.load(fp)
        xyz_cam = d['xyz'].astype(np.float32)
        rgb = d['rgb'].astype(np.float32)

        # 相机系 -> lidar 系 (x右 / y前 / z上)
        xyz_lidar = np.stack([xyz_cam[:, 0], xyz_cam[:, 2], -xyz_cam[:, 1]], axis=-1)

        N = xyz_lidar.shape[0]
        if N > NPOINTS:
            idx = rng.choice(N, NPOINTS, replace=False)
            xyz_lidar, rgb = xyz_lidar[idx], rgb[idx]

        pts = torch.from_numpy(np.concatenate([xyz_lidar, rgb], axis=1)).float().unsqueeze(0).cuda()
        pred, conf = infer(model, pts)

        # 丢弃 noise + 低置信 + 非静态类（幻觉的车辆/行人等）
        keep = (conf > args.conf_thresh) & np.isin(pred, list(STATIC_CLASSES))
        if keep.sum() < args.min_keep:
            n_skip += 1
            continue

        out_name = f"street_{scene}_{stem}.npz"
        np.savez_compressed(
            os.path.join(out_train, out_name),
            xyz=xyz_lidar[keep].astype(np.float32),
            rgb=rgb[keep].astype(np.float32),
            sem=pred[keep].astype(np.int16),
            conf=conf[keep].astype(np.float32),
        )
        n_used += 1
        n_pts_total += N
        n_pts_keep += int(keep.sum())
        hist_keep += np.bincount(pred[keep], minlength=32)

        if (fi + 1) % 100 == 0:
            print(f"  [{fi+1}/{len(files)}] 已处理, 用时 {time.time()-t0:.0f}s", flush=True)

    print(f"[done] 保留 {n_used}/{len(files)} 样本, 跳过 {n_skip} (不足 {args.min_keep} 点)",
          flush=True)
    print(f"       总点 {n_pts_total}, 保留点 {n_pts_keep} ({n_pts_keep/max(n_pts_total,1)*100:.1f}%)",
          flush=True)
    print(f"       保留类别直方图: {hist_keep.tolist()}", flush=True)

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump({'n_used': n_used, 'n_skip': n_skip,
                   'n_pts_total': n_pts_total, 'n_pts_keep': n_pts_keep,
                   'hist_keep': hist_keep.tolist()}, f, indent=2)
    print(f"[done] 伪标注 -> {out_train}  total {time.time()-t0:.0f}s", flush=True)


def glob_safe(root):
    import glob
    return sorted(glob.glob(os.path.join(root, '*', '*.npz')))


if __name__ == '__main__':
    main()
