#!/usr/bin/env python3
"""离线生成实例边界标签 is_boundary，追加写回每个 16k npz。

边界定义：点 p 为边界，当且仅当其半径 r 邻域内存在点 q，使
    (sem_q, inst_q) != (sem_p, inst_p)
即语义类别不同，或语义相同但属于不同实例（inst 不同）。这同时覆盖语义边界与实例边界。

为什么用**原始 32 类 sem**而不是合并后的 12 类 sem：
  nuScenes-lidarseg 的 inst 在同一帧内会跨类复用（见
  NuscenesSegDataset._build_rare_weights 的坑），所以"实例唯一身份"必须用
  (原始 sem, inst) 联合键。若用合并后 sem 做键，adult(inst=3) 与 child(inst=3)
  合并成 person 后会撞键、漏掉两个人之间的实例边界。实例结构不随语义合并而改变，
  因此本标签对 12 类 / 32 类训练同样有效。

用法:
    python scripts/build_boundary_label.py --data data/nuscenes_seg_full_16k --radius 0.3 \
        [--subsets train val] [--workers 8]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def process_file(args):
    npz_path, radius = args
    with np.load(npz_path) as d:
        xyz_raw = d['xyz']
        rgb_raw = d['rgb']
        sem_raw = d['sem']
        inst_raw = d['inst']
        xyz = d['xyz'].astype(np.float64)
        sem = d['sem'].astype(np.int64)
        inst = d['inst'].astype(np.int64)

    # 实例唯一键：(原始 sem, inst)。inst 在同一帧内跨类复用，必须带 sem。
    # inst<0 表示非实例点，用 max(inst,0) 防御。
    key = sem * 1000 + np.maximum(inst, 0)

    tree = cKDTree(xyz)
    idx_lists = tree.query_ball_point(xyz, radius)

    is_boundary = np.zeros(xyz.shape[0], dtype=np.uint8)
    for i, nbr in enumerate(idx_lists):
        nbr = np.asarray(nbr)
        nbr = nbr[nbr != i]          # 排除自身
        if nbr.size and np.any(key[nbr] != key[i]):
            is_boundary[i] = 1

    # 写回：保留原始 xyz/rgb/sem/inst（含 dtype），追加 is_boundary。
    # np.savez_compressed 会自动补 .npz 后缀，故临时名也以 .npz 结尾。
    tmp = str(npz_path) + '.tmp.npz'
    np.savez_compressed(tmp,
                        xyz=xyz_raw, rgb=rgb_raw, sem=sem_raw, inst=inst_raw,
                        is_boundary=is_boundary)
    os.replace(tmp, npz_path)
    return npz_path, float(is_boundary.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='数据集根目录（含 train/ val/）')
    ap.add_argument('--radius', type=float, default=0.3, help='边界判定半径（米）')
    ap.add_argument('--subsets', nargs='+', default=['train', 'val'])
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    files = []
    for s in args.subsets:
        d = Path(args.data) / s
        if not d.exists():
            print(f'[boundary] 跳过不存在的子集目录 {d}', file=sys.stderr)
            continue
        files += sorted(d.glob('*.npz'))
    files = [str(f) for f in files]
    print(f'[boundary] {len(files)} 个 npz，radius={args.radius}, workers={args.workers}',
          flush=True)

    tasks = [(f, args.radius) for f in files]
    t0 = time.time()
    fracs = []
    if args.workers > 1 and len(files) > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for i, (fp, frac) in enumerate(pool.imap_unordered(process_file, tasks, chunksize=16), 1):
                fracs.append(frac)
                if i % 2000 == 0:
                    print(f'  ... {i}/{len(files)} ({time.time()-t0:.0f}s)', flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            fp, frac = process_file(task)
            fracs.append(frac)
            if i % 2000 == 0:
                print(f'  ... {i}/{len(files)} ({time.time()-t0:.0f}s)', flush=True)

    if fracs:
        print(f'[boundary] 完成 {len(files)} 个 npz，用时 {time.time()-t0:.0f}s，'
              f'平均边界点占比 {np.mean(fracs)*100:.2f}%', flush=True)


if __name__ == '__main__':
    main()
