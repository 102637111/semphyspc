#!/usr/bin/env python3
"""实验I：通用置信度门控（拒识曲线 / selective prediction）。

背景：
  把 barrier 的「类别门控」推广为通用「置信度门控」：对每个 B2 判类的实例，
  计算其 softmax 不确定度，按不确定度从高到低「拒识」（交人工），量化剩余点的精度。
  这回答「知道什么时候不知道」这一可信决策问题，也是开集识别/主动学习的入口。

置信度信号（每实例一个标量）：
  maxp    : B2 softmax 最大概率（越高越确信）
  margin  : top1 - top2 softmax（越高越确信）
  entropy : -Σ p log p（越低越确信）
  agree   : dense 多数投票 == B2 判类（1=一致=确信, 0=分歧=不确定），二值
  对照    : 随机拒识（精度应保持平坦，作为无信息基线）

输出：拒识曲线（keep_ratio=保留最确信点占比 -> 剩余点 thing 精度）。
  拒掉 X% 最不确定的点 = 保留最确信的 (1-X)% 点。
  有效信号：keep_ratio 越小精度越高（保留最确信子集时精度上升）。
  缓存落盘 experiments/confidence_gate_cache.npz，后续纯后处理无需重跑前向。

用法:
    python scripts/eval_confidence_gate.py \
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
    ap.add_argument('--cache', default='experiments/confidence_gate_cache.npz')
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    files = sorted(glob.glob(os.path.join(args.val, '*.npz')))
    if args.n < len(files):
        files = rng.choice(files, args.n, replace=False).tolist()

    # ---- 前向缓存（若缓存存在则直接复用）----
    if os.path.exists(args.cache):
        c = np.load(args.cache)
        gt, b2, maxp, margin, entropy, agree = (
            c['gt'], c['b2'], c['maxp'], c['margin'], c['entropy'], c['agree'])
        print(f'[expI] 复用缓存 {args.cache}: {gt.shape[0]} 点', flush=True)
    else:
        seg = build_seg(args.offset_ckpt)
        cls = build_cls(args.cls_ckpt)
        gt_all, b2_all, maxp_all, margin_all, entropy_all, agree_all = (
            [], [], [], [], [], [])

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

            b2_p = np.full(thing_idx.shape[0], -1, dtype=np.int64)
            maxp_p = np.full(thing_idx.shape[0], np.nan, dtype=np.float32)
            margin_p = np.full(thing_idx.shape[0], np.nan, dtype=np.float32)
            entropy_p = np.full(thing_idx.shape[0], np.nan, dtype=np.float32)
            agree_p = np.full(thing_idx.shape[0], np.nan, dtype=np.float32)

            for pid in np.unique(pred_inst[pred_inst >= 0]):
                m = pred_inst == pid
                crop = prep_crop(xyz_t[m], rgb[thing_idx][m], 1024, seed_rng=rng)
                crop_t = torch.from_numpy(crop).float().unsqueeze(0).cuda()
                with torch.no_grad():
                    cls_logit = cls(crop_t)[0]  # (11,)
                probs = torch.softmax(cls_logit, -1)
                p = probs.cpu().numpy()
                b2 = CLASS2SEM[int(cls_logit.argmax(-1).item())]
                srt = np.sort(p)[::-1]
                # dense 多数投票（sem id）
                dmaj = int(np.bincount(dense_p[m].astype(np.int64)).argmax())
                b2_p[m] = b2
                maxp_p[m] = srt[0]
                margin_p[m] = srt[0] - srt[1]
                entropy_p[m] = float(-(p * np.log(p + 1e-12)).sum())
                agree_p[m] = 1.0 if dmaj == b2 else 0.0

            valid = b2_p >= 0
            gt_all.append(sem_t[valid])
            b2_all.append(b2_p[valid])
            maxp_all.append(maxp_p[valid])
            margin_all.append(margin_p[valid])
            entropy_all.append(entropy_p[valid])
            agree_all.append(agree_p[valid])

            if (fi + 1) % 200 == 0:
                print(f"  [{fi+1}/{len(files)}] 用时 {time.time()-t0:.0f}s", flush=True)

        gt = np.concatenate(gt_all).astype(np.int64)
        b2 = np.concatenate(b2_all).astype(np.int64)
        maxp = np.concatenate(maxp_all).astype(np.float32)
        margin = np.concatenate(margin_all).astype(np.float32)
        entropy = np.concatenate(entropy_all).astype(np.float32)
        agree = np.concatenate(agree_all).astype(np.float32)
        np.savez(args.cache, gt=gt, b2=b2, maxp=maxp, margin=margin,
                 entropy=entropy, agree=agree)
        print(f'\n[expI] 前向缓存 {gt.shape[0]} 个被聚类 thing 点, 用时 {time.time()-t0:.0f}s, '
              f'已存 {args.cache}', flush=True)

    correct = (b2 == gt).astype(np.float32)
    N = len(gt)
    print(f'  基线（无拒识，全 B2 级联，被聚类子集）thing 精度 = {correct.mean():.4f}', flush=True)

    def curve(conf, name, higher_is_confident=True):
        """保留最确信的点，输出 keep_ratio -> 精度。"""
        order = np.argsort(-conf if higher_is_confident else conf)  # 最确信在前
        cum = np.cumsum(correct[order])
        print(f'\n[{name}] 拒识曲线 (保留最确信的 keep_ratio -> 精度):', flush=True)
        for r in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k = int(N * r)
            print(f'    keep {r:.1f} ({k:7d} 点) -> 精度 {cum[k-1]/k if k else 0:.4f}', flush=True)
        return cum

    # maxp/margin：高=确信；entropy：低=确信
    curve(maxp, 'maxp(最大概率)')
    curve(margin, 'margin(top1-top2)')
    curve(entropy, 'entropy(熵)', higher_is_confident=False)

    # dense↔B2 分歧（二值）：保留一致(agree=1)子集 vs 分歧(agree=0)子集
    agree1 = correct[agree == 1.0]
    agree0 = correct[agree == 0.0]
    print(f'\n[agree(dense↔B2分歧)] 一致 {agree1.shape[0]} 点 精度 {agree1.mean():.4f} | '
          f'分歧 {agree0.shape[0]} 点 精度 {agree0.mean():.4f}', flush=True)

    # 无信息基线：随机拒识
    rng_r = np.random.default_rng(0)
    order = rng_r.permutation(N)
    cum = np.cumsum(correct[order])
    print(f'\n[random] 随机拒识对照 (keep_ratio -> 精度):', flush=True)
    for r in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k = int(N * r)
        print(f'    keep {r:.1f} -> 精度 {cum[k-1]/k if k else 0:.4f}', flush=True)


if __name__ == '__main__':
    main()
