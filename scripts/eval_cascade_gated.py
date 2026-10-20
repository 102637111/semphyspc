#!/usr/bin/env python3
"""实验F：类别门控级联 —— 针对 barrier 被 B2 误判成 trailer 的修复。

背景：
  端到端级联中 barrier 唯一倒退(0.73->0.40)，根因(E 实验)是 B2 把 barrier 误判成 trailer
  (49% barrier 点 -> trailer)，两者都是长条水平结构、单位球归一化后难分。
  但 dense 模型对 barrier 本身有 0.73，问题出在「级联用 B2 覆盖了 dense」。

门控规则（部署现实，只用模型预测不用真值）：
  对每个聚出的实例，取 dense 逐点预测的多数投票；若多数为 gate 类(默认 barrier=9)，
  则保留 dense 预测(不 B2 判类)；否则按原级联用 B2 判类。

用法:
    python scripts/eval_cascade_gated.py \
        --val data/nuscenes_seg_full_16k/val \
        --offset_ckpt experiments/cascade_models/ckpt-offset-finetuned.pth \
        --cls_ckpt experiments/cascade_models/ckpt-cls.pth \
        --n 500 --eps 0.5 --min_points 3 --gate 9 --seed 0
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_cascade_end2end import (
    CLASS2SEM, SEM2CLASS, CLASS_NAMES, prep_crop, cluster_dbscan, build_seg, build_cls,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', required=True)
    ap.add_argument('--offset_ckpt', required=True)
    ap.add_argument('--cls_ckpt', required=True)
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--eps', type=float, default=0.5)
    ap.add_argument('--min_points', type=int, default=3)
    ap.add_argument('--gate', type=int, nargs='+', default=[9],
                    help='门控保留 dense 的语义类 id（默认 9=barrier）')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    files = sorted(glob.glob(os.path.join(args.val, '*.npz')))
    if args.n < len(files):
        files = rng.choice(files, args.n, replace=False).tolist()

    print(f'[expF] 类别门控级联 gate={args.gate}, {len(files)} val 场景', flush=True)
    seg = build_seg(args.offset_ckpt)
    cls = build_cls(args.cls_ckpt)

    dense_correct = np.zeros(len(CLASS2SEM), dtype=np.int64)
    dense_total = np.zeros(len(CLASS2SEM), dtype=np.int64)
    casc_correct = np.zeros(len(CLASS2SEM), dtype=np.int64)
    casc_total = np.zeros(len(CLASS2SEM), dtype=np.int64)
    n_gated = 0  # 被门控保留 dense 的实例数

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
        off = off[0].cpu().numpy()

        thing = np.isin(sem, list(SEM2CLASS.keys())) & (inst > 0)
        if thing.sum() < args.min_points:
            continue
        thing_idx = np.where(thing)[0]
        sem_t = sem[thing_idx]
        dense_p = dense_pred[thing_idx]

        xyz_t = xyz[thing_idx]
        shifted = xyz_t + off[thing_idx]
        pred_inst = cluster_dbscan(shifted, args.eps, args.min_points)
        casc_p = np.full(thing_idx.shape[0], -1, dtype=np.int64)

        for pid in np.unique(pred_inst[pred_inst >= 0]):
            m = pred_inst == pid
            # dense 多数投票作为门控
            dense_votes = dense_p[m]
            if len(dense_votes) > 0:
                maj = int(np.bincount(dense_votes).argmax())
            else:
                maj = -1
            if maj in args.gate:
                # 门控：保留 dense 预测
                casc_p[m] = dense_p[m]
                n_gated += 1
            else:
                crop = prep_crop(xyz_t[m], rgb[thing_idx][m], 1024, seed_rng=rng)
                crop_t = torch.from_numpy(crop).float().unsqueeze(0).cuda()
                with torch.no_grad():
                    cls_logit = cls(crop_t)
                casc_p[m] = CLASS2SEM[int(cls_logit[0].argmax(-1).item())]

        for c, s in enumerate(CLASS2SEM):
            mm = sem_t == s
            dense_total[c] += mm.sum()
            dense_correct[c] += (dense_p[mm] == s).sum()
            casc_total[c] += mm.sum()
            casc_correct[c] += (casc_p[mm] == s).sum()

        if (fi + 1) % 200 == 0:
            print(f"  [{fi+1}/{len(files)}] 用时 {time.time()-t0:.0f}s", flush=True)

    print("\n================ 实验F 结果（类别门控级联 vs dense / 原级联）================", flush=True)
    print(f"  场景数 {len(files)}, 门控实例数 {n_gated}, gate 类 = {args.gate}", flush=True)
    print(f"  {'类':<14s} {'dense_acc':>10s} {'gate_cascade':>13s} {'提升':>8s} {'thing点数':>8s}", flush=True)
    for c, name in enumerate(CLASS_NAMES):
        da = dense_correct[c] / max(dense_total[c], 1)
        ca = casc_correct[c] / max(casc_total[c], 1)
        print(f"  {name:<14s} {da:>10.4f} {ca:>13.4f} {ca-da:>+8.4f} {dense_total[c]:>8d}", flush=True)

    da_all = dense_correct.sum() / max(dense_total.sum(), 1)
    ca_all = casc_correct.sum() / max(casc_total.sum(), 1)
    print(f"  {'整体(11类)':<14s} {da_all:>10.4f} {ca_all:>13.4f} {ca_all-da_all:>+8.4f} "
          f"{dense_total.sum():>8d}", flush=True)
    print("  (对照: 无门控级联整体 0.6901 @ eps=0.5)", flush=True)
    print("=============================================================================", flush=True)


if __name__ == '__main__':
    main()
