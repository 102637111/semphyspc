#!/usr/bin/env python3
"""从 nuScenes 提取非地面「物体(thing)」crop，供物体级分类器训练（实验B2）。

对每个样本按 (sem, inst) 分组（用 sem*1000+inst 还原 panoptic label 作唯一 key），
取 thing 类(1..23)且 inst>0 的实例。每个实例保存其原始点(xyz+rgb)，类别重映射为 0..K-1。

输出单文件聚合格式（紧凑、快速加载）：
  {out}/objects.npz
    points  : (total_pts, 6) float32  [xyz, rgb]，按对象连续排列
    offsets : (num_obj+1,) int64       对象 i 的点区间 = points[offsets[i]:offsets[i+1]]
    labels  : (num_obj,) int8          重映射后的类别 0..K-1
    class_names: (K,) 字符串数组

用法:
    python scripts/extract_nuscenes_objects.py \
        --data data/nuscenes_seg_full_16k/train \
        --out /mnt/data/datasets/UR-MAT/nuscenes_objects \
        --n 10000 --max_per_class 4000 --min_pts 16 --seed 0
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

# 参与判类的 thing 类（按 nuScenes 频率降序），重映射为 0..K-1
CLASSES = [17, 2, 9, 23, 12, 22, 11, 16, 21, 18, 14]
CLASS_NAMES = ['car', 'ped.adult', 'barrier', 'truck', 'trafficcone',
               'trailer', 'pushable', 'bus.rigid', 'motorcycle',
               'construction', 'bicycle']
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='nuScenes train 目录')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=10000, help='采样样本数')
    ap.add_argument('--max_per_class', type=int, default=4000)
    ap.add_argument('--min_pts', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    rng = np.random.default_rng(args.seed)
    if args.n < len(files):
        files = rng.choice(files, args.n, replace=False).tolist()

    os.makedirs(args.out, exist_ok=True)
    print(f"[extract] 遍历 {len(files)} 个样本, 提取 {len(CLASSES)} 类物体", flush=True)

    pts_buf = []       # 每对象 (Ni,6)
    labels_buf = []
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    n_inst = 0
    t0 = time.time()

    for fi, fp in enumerate(files):
        d = np.load(fp)
        sem = d['sem'].astype(np.int64)
        inst = d['inst'].astype(np.int64)
        xyz = d['xyz'].astype(np.float32)
        rgb = d['rgb'].astype(np.float32)

        # 唯一对象 key = panoptic label (sem*1000 + inst)
        key = sem * 1000 + inst
        # 有效：thing 类且 inst>0（inst=0 是 stuff/背景）
        valid = (inst > 0) & np.isin(sem, CLASSES)

        for k in np.unique(key[valid]):
            m = key == k
            c = int(sem[m][0])  # 同一实例 sem 一致
            if c not in CLASS2IDX:
                continue
            idx = CLASS2IDX[c]
            if counts[idx] >= args.max_per_class:
                continue
            if m.sum() < args.min_pts:
                continue
            pts_buf.append(np.concatenate([xyz[m], rgb[m]], axis=1).astype(np.float32))
            labels_buf.append(idx)
            counts[idx] += 1
            n_inst += 1

        if (fi + 1) % 1000 == 0:
            print(f"  [{fi+1}/{len(files)}] 实例 {n_inst} 用时 {time.time()-t0:.0f}s",
                  flush=True)

    # 聚合保存
    total_pts = int(sum(p.shape[0] for p in pts_buf))
    points = np.empty((total_pts, 6), dtype=np.float32)
    offsets = np.zeros(n_inst + 1, dtype=np.int64)
    off = 0
    for i, p in enumerate(pts_buf):
        points[off:off + p.shape[0]] = p
        off += p.shape[0]
        offsets[i + 1] = off
    labels = np.array(labels_buf, dtype=np.int8)

    np.savez_compressed(
        os.path.join(args.out, 'objects.npz'),
        points=points, offsets=offsets, labels=labels,
        class_names=np.array(CLASS_NAMES),
        class_ids=np.array(CLASSES, dtype=np.int64),
    )

    print(f"\n[extract done] 对象总数 {n_inst}, 总点数 {total_pts}", flush=True)
    print("  各类数量:", flush=True)
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {i:2d} {name:<16s} : {counts[i]}", flush=True)

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        import json
        json.dump({'counts': counts.tolist(), 'class_names': CLASS_NAMES,
                   'class_ids': CLASSES, 'n_inst': n_inst}, f, indent=2)
    print(f"[extract] 保存 -> {os.path.join(args.out, 'objects.npz')}  "
          f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
