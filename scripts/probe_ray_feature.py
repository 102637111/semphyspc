#!/usr/bin/env python3
"""数据级探针：盲区邻接特征 blind_adj 是否对罕见类有判别力。

对 val 集每个 npz + 其 _ray.npz sidecar，按 12 类合并语义统计 blind_adj 的类均值，
并报「相对 ground 的提升（lift）」。若 frangible/two_wheeler/manmade/person 等罕见
几何类显著高于 ground/vegetation 背景，说明射线盲区特征有判别价值，值得训练 smoke；
否则长线接 PointMamba 大概率无效，先止损。

用法：
  python3 scripts/probe_ray_feature.py --split val --limit 0
"""
import os, argparse
import numpy as np
from pathlib import Path

# 12 类合并（与 NuscenesSegFull16kMerged.yaml 的 CLASS_MAP 一致）的语义名
CLASS_NAMES = {0: 'noise', 1: 'animal', 2: 'person', 3: 'car', 4: 'bus', 5: 'truck',
               6: 'two_wheeler', 7: 'ground', 8: 'manmade', 9: 'vegetation',
               10: 'frangible', 11: 'ego'}
# 原始 32 类 lidarseg id → 12 类合并 id（npz 存的是原始 32 类，须先映射）
CLASS_MAP = np.array([0, 1, 2, 2, 2, 2, 2, 2, 2, 10, 10, 10, 10, 8, 6, 4, 4,
                      3, 5, 3, 3, 6, 5, 5, 7, 7, 7, 7, 8, 8, 9, 11], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/nuscenes_seg_full_16k')
    ap.add_argument('--split', default='val')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    files = sorted((Path(args.data) / args.split).glob('*.npz'))
    if args.limit > 0:
        files = files[:args.limit]

    sums = np.zeros(12, np.float64)
    cnts = np.zeros(12, np.int64)
    n_miss = 0
    ray_dir = Path(args.data) / 'ray_feat' / args.split
    for i, fp in enumerate(files):
        ray_path = ray_dir / fp.name
        if not ray_path.exists():
            n_miss += 1
            continue
        z = np.load(str(fp))
        sem = z['sem'].astype(np.int64)
        sem = CLASS_MAP[np.clip(sem, 0, 31)]  # 原始 32 类 → 12 类合并
        adj = np.load(ray_path)['blind_adj'].astype(np.float64)
        for c in range(12):
            m = sem == c
            sums[c] += adj[m].sum()
            cnts[c] += int(m.sum())

    ground_mean = sums[7] / max(cnts[7], 1)
    print(f'\n[盲区邻接特征 类均值] split={args.split} frames={len(files) - n_miss} '
          f'(缺 sidecar={n_miss})')
    print(f'{"类":<14}{"点数":>10}{"mean_adj":>10}{"lift/ground":>12}')
    for c in range(12):
        if cnts[c] == 0:
            continue
        m = sums[c] / cnts[c]
        lift = m / max(ground_mean, 1e-9)
        print(f'{CLASS_NAMES[c]:<14}{cnts[c]:>10}{m:>10.4f}{lift:>12.3f}')

    # 简明结论：罕见几何类 lift > 1.2 即视为有判别力
    rare = [10, 6, 8, 2, 5, 4]
    lifts = {c: (sums[c] / max(cnts[c], 1)) / max(ground_mean, 1e-9)
             for c in rare if cnts[c] > 0}
    disc = [c for c, l in lifts.items() if l > 1.2]
    print(f'\n[结论] 罕见几何类 lift(相对 ground): ' +
          ', '.join(f'{CLASS_NAMES[c]}={l:.2f}' for c, l in lifts.items()))
    print(f'        lift>1.2 的类: {[CLASS_NAMES[c] for c in disc] or "无"} '
          f'→ {"有判别力，值得 smoke" if disc else "无判别力，建议止损"}')


if __name__ == '__main__':
    main()
