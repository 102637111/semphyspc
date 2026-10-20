#!/usr/bin/env python3
"""实验C4：eps 扫描 —— 对 offset 实例分割的 DBSCAN 聚类阈值做扫描，
定位「欠分割(rec<prec)」是否由 eps 过大导致，找出最优聚类后处理。

背景：
  C2/C3 的 offset 回归在 eps=0.5 下 recall~0.52 << precision~0.67，
  说明相邻实例被过合并（欠分割）。eps 越小越不易合并，本脚本扫描 eps，
  找 recall/PQ 的最优点，作为与「冻结 vs 解冻」正交的后处理杠杆。

用法:
    python scripts/sweep_eps.py \
        --val data/nuscenes_seg_full_16k/val \
        --ckpt experiments/cascade_models/ckpt-offset-finetuned.pth \
        --n_val 300 --min_points 3 --seed 0
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
from scripts.train_instance_offset import evaluate, SIZE_BINS


def build_model(ckpt):
    cfg = EasyDict({
        'NAME': 'PointMambaSeg', 'trans_dim': 384, 'depth': 12,
        'group_size': 32, 'num_group': 64, 'encoder_dims': 384,
        'in_channels': 6, 'num_classes': 32, 'upsample_k': 3,
        'rms_norm': False, 'drop_path': 0.1, 'drop_out': 0.0,
        'num_materials': 0, 'num_instances': 3,
    })
    model = PointMambaSeg(cfg)
    sd = torch.load(ckpt, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model.cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n_val', type=int, default=300)
    ap.add_argument('--min_points', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    val_files = sorted(glob.glob(os.path.join(args.val, '*.npz')))
    if args.n_val < len(val_files):
        val_files = rng.choice(val_files, args.n_val, replace=False).tolist()

    model = build_model(args.ckpt)
    print(f'[C4] ckpt={args.ckpt}, {len(val_files)} val 场景, min_points={args.min_points}',
          flush=True)

    eps_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"\n{'eps':>5s} {'rec':>8s} {'prec':>8s} {'PQ':>8s} "
          f"{'小物体<50点召回':>14s}", flush=True)
    best_pq, best_eps = 0.0, 0.5
    for eps in eps_list:
        t0 = time.time()
        rec, prec, pq, bin_tp, bin_gt = evaluate(model, val_files, eps, args.min_points)
        small = bin_tp[SIZE_BINS[0]] / max(bin_gt[SIZE_BINS[0]], 1)
        tag = ''
        if pq > best_pq:
            best_pq, best_eps = pq, eps
            tag = ' *best'
        print(f"{eps:>5.2f} {rec:>8.4f} {prec:>8.4f} {pq:>8.4f} "
              f"{small:>14.4f}  用时{time.time()-t0:.0f}s{tag}", flush=True)

    print(f"\n[C4 done] 最优 eps={best_eps:.2f} -> PQ={best_pq:.4f}", flush=True)


if __name__ == '__main__':
    main()
