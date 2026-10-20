#!/usr/bin/env python3
"""
背景过滤工具 — 从采集的原始点云中提取前景物体。

策略:
  1. depth:   深度阈值（物体距离 < max_z 米，默认 0.8m）
  2. cluster: DBSCAN 聚类，保留最大簇
  3. hybrid:  深度阈值 + 聚类（推荐）

用法:
    # 预览效果（显示过滤前后点数）
    python scripts/filter_background.py --raw_dir ./my_raw_dataset --preview

    # 原地过滤（覆盖原文件，谨慎！）
    python scripts/filter_background.py --raw_dir ./my_raw_dataset --apply --strategy hybrid

    # 输出到新目录
    python scripts/filter_background.py --raw_dir ./my_raw_dataset --output_dir ./my_raw_dataset_clean --apply
"""

import os, sys, argparse, numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import DBSCAN


def filter_depth(pc, max_z=0.8, min_z=0.1):
    """深度过滤：保留 min_z < Z < max_z 的点"""
    z = pc[:, 2]
    mask = (z > min_z) & (z < max_z)
    return pc[mask]


def filter_cluster(pc, eps=0.02, min_samples=10):
    """聚类过滤：保留最大簇（假设物体是最大的连通区域）"""
    if len(pc) < 50:
        return pc
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pc[:, :3])
    # 找最大簇（排除噪声 label=-1）
    unique, counts = np.unique(labels, return_counts=True)
    # 排除噪声点
    valid = [(u, c) for u, c in zip(unique, counts) if u >= 0]
    if not valid:
        return pc
    best_label = max(valid, key=lambda x: x[1])[0]
    return pc[labels == best_label]


def filter_hybrid(pc, max_z=1.0, min_z=0.1, eps=0.02, min_samples=10):
    """深度 + 聚类组合过滤"""
    pc = filter_depth(pc, max_z=max_z, min_z=min_z)
    if len(pc) > 50:
        pc = filter_cluster(pc, eps=eps, min_samples=min_samples)
    return pc


FILTERS = {'depth': filter_depth, 'cluster': filter_cluster, 'hybrid': filter_hybrid}


def main():
    parser = argparse.ArgumentParser(description='Background removal for point cloud dataset')
    parser.add_argument('--raw_dir', required=True, help='原始数据目录')
    parser.add_argument('--output_dir', default=None, help='输出目录（默认覆盖原文件）')
    parser.add_argument('--strategy', default='hybrid', choices=['depth', 'cluster', 'hybrid'])
    parser.add_argument('--max_z', type=float, default=0.8, help='最大深度 (m)')
    parser.add_argument('--min_z', type=float, default=0.1, help='最小深度 (m)')
    parser.add_argument('--eps', type=float, default=0.02, help='DBSCAN 半径')
    parser.add_argument('--min_samples', type=int, default=10)
    parser.add_argument('--preview', action='store_true', help='仅预览，不修改')
    parser.add_argument('--apply', action='store_true', help='执行过滤')
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir) if args.output_dir else raw_dir
    filter_fn = FILTERS[args.strategy]

    # 收集文件
    files = sorted(raw_dir.glob('*/*.npy'))
    if not files:
        print(f"No .npy files in {raw_dir}")
        sys.exit(1)

    print(f"Strategy: {args.strategy}  max_z={args.max_z}m  eps={args.eps}")
    print(f"{len(files)} files in {len(set(f.parent.name for f in files))} classes\n")

    total_before = 0
    total_after = 0

    for f in tqdm(files, desc="Processing"):
        pc = np.load(str(f)).astype(np.float32)
        total_before += len(pc)

        filtered = filter_fn(pc, max_z=args.max_z, min_z=args.min_z,
                            eps=args.eps, min_samples=args.min_samples)
        total_after += len(filtered)

        if args.apply and not args.preview:
            out_path = out_dir / f.relative_to(raw_dir) if args.output_dir else f
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), filtered)
        else:
            removed = len(pc) - len(filtered)
            print(f"  {f.parent.name}/{f.name}: {len(pc):,} → {len(filtered):,} "
                  f"(-{removed:,} = -{removed/len(pc)*100:.0f}%)")

    print(f"\nTotal: {total_before:,} → {total_after:,} "
          f"(-{total_before-total_after:,} = -{(total_before-total_after)/total_before*100:.0f}%)")

    if not args.apply:
        print("\n预览模式。加 --apply 执行过滤。")


if __name__ == '__main__':
    main()
