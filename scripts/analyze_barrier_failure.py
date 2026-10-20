#!/usr/bin/env python3
"""实验E：barrier(护栏)级联失败根因分析。

背景：
  端到端级联里 barrier 是唯一倒退类：dense 0.73 -> 级联 0.40。
  假设：barrier 是长条连续结构、没有良定义质心，offset 回归「投票到质心」对其病态，
  导致 (a) 平移点被 DBSCAN 判为噪声，或 (b) 聚出的碎片被 B2 误判，或 (c) 与相邻物体过合并。

本脚本对 barrier 点(语义类 9)统计三类失败来源，定位根因，指导专门处理。

用法:
    python scripts/analyze_barrier_failure.py \
        --val data/nuscenes_seg_full_16k/val \
        --offset_ckpt experiments/cascade_models/ckpt-offset-finetuned.pth \
        --cls_ckpt experiments/cascade_models/ckpt-cls.pth \
        --n 300 --eps 0.5 --min_points 3 --seed 0
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.point_mamba_seg import PointMambaSeg
from models.point_mamba_scan import PointMambaScan
from scripts.eval_cascade_end2end import CLASS2SEM, SEM2CLASS, CLASS_NAMES, prep_crop, cluster_dbscan

BARRIER_SEM = 9


def build_seg(ckpt):
    cfg = EasyDict({'NAME': 'PointMambaSeg', 'trans_dim': 384, 'depth': 12,
                    'group_size': 32, 'num_group': 64, 'encoder_dims': 384,
                    'in_channels': 6, 'num_classes': 32, 'upsample_k': 3,
                    'rms_norm': False, 'drop_path': 0.1, 'drop_out': 0.0,
                    'num_materials': 0, 'num_instances': 3})
    m = PointMambaSeg(cfg)
    sd = torch.load(ckpt, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m.cuda().eval()


def build_cls(ckpt):
    cfg = EasyDict({'NAME': 'PointMambaScan', 'trans_dim': 384, 'depth': 12,
                    'cls_dim': len(CLASS2SEM), 'group_size': 32, 'num_group': 64,
                    'encoder_dims': 384, 'in_channels': 6, 'rms_norm': False,
                    'drop_path': 0.1, 'drop_out': 0.0,
                    'use_cls_token': False, 'max_head': False, 'avg_head': True})
    m = PointMambaScan(cfg)
    sd = torch.load(ckpt, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m.cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', required=True)
    ap.add_argument('--offset_ckpt', required=True)
    ap.add_argument('--cls_ckpt', required=True)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--eps', type=float, default=0.5)
    ap.add_argument('--min_points', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    files = sorted(glob.glob(os.path.join(args.val, '*.npz')))
    if args.n < len(files):
        files = rng.choice(files, args.n, replace=False).tolist()

    seg = build_seg(args.offset_ckpt)
    cls = build_cls(args.cls_ckpt)

    n_barrier = 0
    n_noise = 0                 # barrier 点被判 DBSCAN 噪声（未聚类）
    n_clustered = 0             # barrier 点被聚进某实例
    n_correct = 0               # 聚类后 B2 判对 barrier
    confusion = np.zeros(len(CLASS2SEM), dtype=np.int64)  # B2 判成什么类
    # 过合并：barrier 点所在 cluster 里非 barrier GT 点占比
    merged_fracs = []

    for fi, f in enumerate(files):
        d = np.load(f)
        xyz = d['xyz'].astype(np.float32)
        rgb = d['rgb'].astype(np.float32)
        sem = d['sem'].astype(np.int64)
        inst = d['inst'].astype(np.int64)

        thing = np.isin(sem, list(SEM2CLASS.keys())) & (inst > 0)
        if thing.sum() < args.min_points:
            continue
        thing_idx = np.where(thing)[0]
        sem_t = sem[thing_idx]
        barrier = sem_t == BARRIER_SEM
        if barrier.sum() == 0:
            continue
        barrier_pos = np.where(barrier)[0]  # 在 thing_idx 里的下标

        pts = torch.from_numpy(np.concatenate([xyz, rgb], 1)).float().unsqueeze(0).cuda()
        with torch.no_grad():
            _, _, off = seg(pts)
        off = off[0].cpu().numpy()

        xyz_t = xyz[thing_idx]
        shifted = xyz_t + off[thing_idx]
        pred_inst = cluster_dbscan(shifted, args.eps, args.min_points)

        # 每个 barrier 点的命运
        for bp in barrier_pos:
            n_barrier += 1
            pid = pred_inst[bp]
            if pid < 0:
                n_noise += 1
                continue
            n_clustered += 1
            m = pred_inst == pid
            # 过合并：该 cluster 里非 barrier GT 点的占比
            other = (sem_t[m] != BARRIER_SEM).sum()
            merged_fracs.append(other / max(m.sum(), 1))
            # B2 判类
            crop = prep_crop(xyz_t[m], rgb[thing_idx][m], 1024, seed_rng=rng)
            crop_t = torch.from_numpy(crop).float().unsqueeze(0).cuda()
            with torch.no_grad():
                cls_logit = cls(crop_t)
            pred_sem = CLASS2SEM[int(cls_logit[0].argmax(-1).item())]
            confusion[SEM2CLASS[pred_sem]] += 1
            if pred_sem == BARRIER_SEM:
                n_correct += 1

    print('\n================ barrier(护栏) 级联失败根因分析 ================', flush=True)
    print(f'  barrier GT 点数: {n_barrier}', flush=True)
    print(f'  DBSCAN 噪声(未聚类, 自动判错): {n_noise}  ({n_noise/max(n_barrier,1)*100:.1f}%)', flush=True)
    print(f'  被聚类: {n_clustered}  ({n_clustered/max(n_barrier,1)*100:.1f}%)', flush=True)
    print(f'    其中 B2 判对 barrier: {n_correct}  ({n_correct/max(n_clustered,1)*100:.1f}%)', flush=True)
    print(f'  聚类内 B2 判类分布(混淆):', flush=True)
    for c, name in enumerate(CLASS_NAMES):
        if confusion[c] > 0:
            print(f'    -> {name:<14s} {confusion[c]}', flush=True)
    if merged_fracs:
        arr = np.array(merged_fracs)
        print(f'  过合并: barrier 点所在 cluster 中非 barrier 点平均占比 '
              f'{arr.mean()*100:.1f}% (中位 {np.median(arr)*100:.1f}%)', flush=True)
    print('=====================================================================', flush=True)


if __name__ == '__main__':
    main()
