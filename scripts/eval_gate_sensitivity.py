#!/usr/bin/env python3
"""实验G：类别门控敏感性分析 —— 系统验证最优门控集合。

背景：
  门控级联对 barrier 有效(0.69->0.72)，但「是否该推广到 construction 等其他类」需严谨回答。
  本脚本一次前向缓存每个实例的 (dense 多数投票, B2 预测)，随后纯后处理评估
  「全 B2」「全 dense」「每类单独门控」「barrier+construction」等所有门控集合，
  找出使整体 thing 精度最大的门控集合。

用法:
    python scripts/eval_gate_sensitivity.py \
        --val data/nuscenes_seg_full_16k/val \
        --offset_ckpt experiments/cascade_models/ckpt-offset-finetuned.pth \
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
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    files = sorted(glob.glob(os.path.join(args.val, '*.npz')))
    if args.n < len(files):
        files = rng.choice(files, args.n, replace=False).tolist()

    seg = build_seg(args.offset_ckpt)
    cls = build_cls(args.cls_ckpt)

    # 逐点缓存（跨场景累积）：gt / dense 预测 / B2 预测 / dense 多数投票
    gt_all, dense_all, b2_all, dmaj_all = [], [], [], []

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

        # 每个有效簇：B2 预测 + dense 多数投票
        b2_p = np.full(thing_idx.shape[0], -1, dtype=np.int64)
        dmaj_p = np.full(thing_idx.shape[0], -1, dtype=np.int64)
        for pid in np.unique(pred_inst[pred_inst >= 0]):
            m = pred_inst == pid
            dmaj = int(np.bincount(dense_p[m]).argmax())
            crop = prep_crop(xyz_t[m], rgb[thing_idx][m], 1024, seed_rng=rng)
            crop_t = torch.from_numpy(crop).float().unsqueeze(0).cuda()
            with torch.no_grad():
                cls_logit = cls(crop_t)
            b2 = CLASS2SEM[int(cls_logit[0].argmax(-1).item())]
            b2_p[m] = b2
            dmaj_p[m] = dmaj

        gt_all.append(sem_t)
        dense_all.append(dense_p)
        b2_all.append(b2_p)
        dmaj_all.append(dmaj_p)

        if (fi + 1) % 200 == 0:
            print(f"  [{fi+1}/{len(files)}] 用时 {time.time()-t0:.0f}s", flush=True)

    gt = np.concatenate(gt_all)
    dense = np.concatenate(dense_all)
    b2 = np.concatenate(b2_all)
    dmaj = np.concatenate(dmaj_all)
    print(f'\n[expG] 缓存 {gt.shape[0]} 个 thing 点, 用时 {time.time()-t0:.0f}s', flush=True)

    def acc(casc_p):
        tot = correct = 0
        for c, s in enumerate(CLASS2SEM):
            m = gt == s
            tot += m.sum()
            correct += (casc_p[m] == s).sum()
        return correct / max(tot, 1)

    # dense 基线 / 全 B2（无门控）
    dense_acc = acc(dense)
    b2_acc = acc(b2)
    print(f'\n{"配置":<28s} {"整体 thing 精度":>14s}', flush=True)
    print(f'{"dense 逐点(无级联)":<28s} {dense_acc:>14.4f}', flush=True)
    print(f'{"全 B2(无门控)":<28s} {b2_acc:>14.4f}', flush=True)

    # 每类单独门控
    print('\n单类门控（dense 多数投票为该类时保留 dense）:', flush=True)
    single = {}
    for c, s in enumerate(CLASS2SEM):
        casc = np.where(dmaj == s, dense, b2)
        a = acc(casc)
        single[s] = a
        flag = ' *' if a > b2_acc else ''
        print(f'  gate {s:2d} {CLASS_NAMES[c]:<14s} -> {a:.4f}  (vs 全B2 {b2_acc:.4f}){flag}',
              flush=True)

    # 最优门控 = 所有「单类门控优于全 B2」的类
    better = sorted([s for s, a in single.items() if a > b2_acc])
    casc = np.where(np.isin(dmaj, better), dense, b2)
    print(f'\n最优门控集合(所有优于全B2的类) = {better} -> {acc(casc):.4f}', flush=True)

    # 对照：barrier+construction 联合门控
    casc2 = np.where(np.isin(dmaj, [CLASS2SEM[SEM2CLASS[9]], CLASS2SEM[SEM2CLASS[18]]]), dense, b2)
    print(f'对照 barrier+construction 联合门控 -> {acc(casc2):.4f}', flush=True)
    print('\n(dense 多数投票为该类=barrier 类 id 9 / construction 类 id 18)', flush=True)


if __name__ == '__main__':
    main()
