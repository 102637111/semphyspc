#!/usr/bin/env python3
"""UR-MAT -> PointMamba npz 转换：彩色点云 + 14 类材质标签。

UR-MAT (Zenodo 18911512, v3) 是 Unreal Engine 5 合成的城市级材质数据集，
每个帧提供像素对齐的 rgb / depth_npy / segmentation_material_npy / metadata。

本脚本把 2D 图像反投影成彩色点云，并带上逐点材质标签，输出与
NuscenesSegDataset 一致的 npz 格式：
    {xyz (N,3) float32 [m], rgb (N,3) float32 [0,1], mat (N,) int16}

用法:
    python scripts/build_urmat_dataset.py \
        --src /mnt/data/datasets/UR-MAT/extracted/Dataset_URMAT_v3 \
        --dst /mnt/data/datasets/UR-MAT/seg_16k \
        --npoints 16384 --num_workers 8

划分:
    UR-MAT/train/*  -> {dst}/train/    (5600 帧)
    UR-MAT/test/*   -> 前 600 帧 val, 后 600 帧 test
"""

import os, argparse, glob
from multiprocessing import Pool
import numpy as np
from PIL import Image

# ---- UnrealCV 默认内参 (FOV=90°, 640x480) ----
FX, FY = 320.0, 320.0
CX, CY = 320.0, 240.0
H, W = 480, 640
DEPTH_SENTINEL = 10000.0   # 100m：裁剪远距离/天空点，避免 Hilbert 序列化 depth>16 (范围需 <655m)

# 14 类材质 label -> name（UR-MAT segmentation_material_npy 的整数标签）
MATERIAL_NAMES = [
    'Limestone', 'Slate', 'Glass', 'Steel', 'Brick', 'Cobblestone',
    'Plaster', 'Concrete', 'Gravel', 'Tiles', 'Wood', 'Asphalt',
    'Plastic', 'Unknown',
]


def backproject(depth):
    """depth (H,W) [cm] -> xyz (H,W,3) [m], 相机坐标系 (x右 y下 z前)."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    z = depth.astype(np.float32) / 100.0  # cm -> m
    x = (u - CX) / FX * z
    y = (v - CY) / FY * z
    return np.stack([x, y, z], axis=-1)


def class_balanced_sample(mat_valid, npoints, rng):
    """在有效像素里做类别均衡采样，保证稀有材质（Glass/Steel 等）不被随机采样淹没。

    - 每个出现的类别至少取 floor(npoints/n_classes) 个点（不足则全取）；
    - 剩余配额从全体有效像素随机补齐。
    """
    N = mat_valid.shape[0]
    classes = np.unique(mat_valid)
    floor = max(1, npoints // len(classes))

    chosen = []
    for c in classes:
        c_idx = np.where(mat_valid == c)[0]
        take = min(floor, len(c_idx))
        chosen.append(rng.choice(c_idx, size=take, replace=False))
    chosen = np.concatenate(chosen)

    if len(chosen) < npoints:
        remain = np.setdiff1d(np.arange(N), chosen)
        extra = rng.choice(remain, size=npoints - len(chosen), replace=False)
        chosen = np.concatenate([chosen, extra])
    elif len(chosen) > npoints:
        chosen = rng.choice(chosen, size=npoints, replace=False)
    return chosen


def worker(t):
    """单帧转换，返回转出的点数，或 None（有效点太少跳过）。"""
    src = t['src']; split = t['src_split']; sid = t['sid']
    depth = np.load(os.path.join(src, split, 'depth_npy', f'{sid}.npy'))
    rgb = np.array(Image.open(os.path.join(src, split, 'rgb', f'{sid}.png')).convert('RGB'),
                   dtype=np.float32) / 255.0
    mat = np.load(os.path.join(src, split, 'segmentation_material_npy', f'{sid}.npy')).astype(np.int16)

    valid = (depth > 0) & (depth < DEPTH_SENTINEL)
    if valid.sum() < t['min_valid']:
        return None

    xyz = backproject(depth)[valid]   # (M,3)
    rgb_v = rgb[valid]                # (M,3)
    mat_v = mat[valid]                # (M,)

    if xyz.shape[0] > t['npoints']:
        rng = np.random.default_rng()
        idx = class_balanced_sample(mat_v, t['npoints'], rng)
        xyz, rgb_v, mat_v = xyz[idx], rgb_v[idx], mat_v[idx]

    os.makedirs(os.path.dirname(t['dst']), exist_ok=True)
    np.savez_compressed(t['dst'], xyz=xyz.astype(np.float32),
                        rgb=rgb_v.astype(np.float32), mat=mat_v.astype(np.int16))
    return xyz.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='UR-MAT 解压根目录 (含 train/ test/)')
    ap.add_argument('--dst', required=True, help='输出 npz 根目录 (含 train/val/test/)')
    ap.add_argument('--npoints', type=int, default=16384)
    ap.add_argument('--min_valid', type=int, default=1000, help='有效点数低于此值跳过该帧')
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='调试：只转前 N 帧')
    args = ap.parse_args()

    jobs = []
    for frame in sorted(glob.glob(os.path.join(args.src, 'train', 'metadata', '*.json'))):
        jobs.append(('train', os.path.basename(frame)[:-5]))
    test_frames = sorted(glob.glob(os.path.join(args.src, 'test', 'metadata', '*.json')))
    for i, frame in enumerate(test_frames):
        sub = 'val' if i < 600 else 'test'
        jobs.append((sub, os.path.basename(frame)[:-5]))

    if args.limit:
        jobs = jobs[:args.limit]

    print(f'[UR-MAT] 待转换 {len(jobs)} 帧 -> {args.dst}')
    task_args = []
    for sub, sid in jobs:
        src_split = 'train' if sub == 'train' else 'test'
        task_args.append(dict(
            src=args.src, src_split=src_split, sid=sid,
            dst=os.path.join(args.dst, sub, f'{sid}.npz'),
            npoints=args.npoints, min_valid=args.min_valid))

    n_done = 0; n_skip = 0
    with Pool(args.num_workers) as pool:
        for r in pool.imap_unordered(worker, task_args, chunksize=8):
            if r is None:
                n_skip += 1
            else:
                n_done += 1
            if (n_done + n_skip) % 200 == 0:
                print(f'  ... done={n_done} skip={n_skip}')

    print(f'[UR-MAT] 完成: {n_done} 帧转换, {n_skip} 帧跳过(有效点不足)')
    print(f'[UR-MAT] 材质类别({len(MATERIAL_NAMES)}):')
    for i, name in enumerate(MATERIAL_NAMES):
        print(f'    {i:2d} {name}')


if __name__ == '__main__':
    main()
