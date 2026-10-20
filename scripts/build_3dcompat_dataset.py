#!/usr/bin/env python3
"""3DCoMPaT++ -> PointMamba npz 转换：物体级彩色点云 + 293 类细粒度材质标签。

3DCoMPaT++ (public S3) 提供逐点材质/部件标签的合成物体点云，hdf5 结构：
    points            (N, 2048, 6)  float32  [xyz(0:3) ~[-0.5,0.5], rgb(3:6) 0-255]
    points_mat_labels (N, 2048)     uint8     293 类细粒度材质
    points_part_labels(N, 2048)     uint16    部件标签
    shape_label       (N,)          uint8     42 类形状类别
    shape_id / style_id             S6 字符串

本脚本转出与 NuscenesSegDataset 一致的 npz（每物体一个）：
    {xyz (2048,3) float32, rgb (2048,3) float32 [0,1], mat (2048,) int16}

用途：作为材质编码器的预训练集（293 类物体级材质 → 后续微调到 UR-MAT 14 类）。

用法:
    python scripts/build_3dcompat_dataset.py \
        --src /mnt/data/datasets/3DCoMPaT/3D_PC \
        --dst /mnt/data/datasets/3DCoMPaT/seg_npz \
        --num_workers 8
"""

import os, argparse, glob
from multiprocessing import Pool
import h5py
import numpy as np

SPLITS = ['train', 'valid', 'test']


def worker(t):
    """读取 hdf5 中索引区间 [i0, i1) 的物体，逐个存 npz。返回转出数量。"""
    path, dst_dir, i0, i1 = t
    cnt = 0
    with h5py.File(path, 'r') as f:
        pts = f['points'][i0:i1]              # (k,2048,6)
        mat = f['points_mat_labels'][i0:i1]   # (k,2048)
        for j in range(pts.shape[0]):
            idx = i0 + j
            xyz = pts[j, :, :3].astype(np.float32)
            rgb = (pts[j, :, 3:] / 255.0).astype(np.float32)
            np.savez_compressed(os.path.join(dst_dir, f'{idx:06d}.npz'),
                                xyz=xyz, rgb=rgb, mat=mat[j].astype(np.int16))
            cnt += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='含 train_fine.hdf5 等文件的目录')
    ap.add_argument('--dst', required=True, help='输出 npz 根目录 (含 train/valid/test/)')
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='调试：每 split 只转前 N 个物体')
    args = ap.parse_args()

    for split in SPLITS:
        path = os.path.join(args.src, f'{split}_fine.hdf5')
        if not os.path.exists(path):
            print(f'[3DCoMPaT] 跳过 {split}: 无 {path}')
            continue
        dst_dir = os.path.join(args.dst, split)
        os.makedirs(dst_dir, exist_ok=True)
        with h5py.File(path, 'r') as f:
            total = f['points'].shape[0]
        total = args.limit if args.limit else total
        print(f'[3DCoMPaT] {split}: {total} 物体 -> {dst_dir}')

        # 按 worker 分片（每片 200 物体，避免单 worker 长时间持有 hdf5）
        chunk = 200
        tasks = [(path, dst_dir, i, min(i + chunk, total))
                 for i in range(0, total, chunk)]
        n = 0
        with Pool(args.num_workers) as pool:
            for c in pool.imap_unordered(worker, tasks, chunksize=1):
                n += c
        print(f'[3DCoMPaT] {split} 完成: {n} 物体')

    print('[3DCoMPaT] 全部完成。材质类别数: 293')


if __name__ == '__main__':
    main()
