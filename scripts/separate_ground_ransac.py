#!/usr/bin/env python3
"""几何地面分离：把语义推理结果里缺失的「地面」用 RANSAC 平面补回，其余保留语义障碍。

背景：nuScenes 零样本模型在真实室内 RGB-D 点云上失效（98.9% 点被判成车/墙，0% 地面），
导致语义代价层全障碍、规划器无路可走。本脚本用几何先验补救：
  RANSAC 拟合最低主平面（法线朝上 = 地面）→ 距该平面 < 阈值的点标为 driveable_surface(24, passable)
  → 其余点保留语义（manmade/bus/car=hard 障碍、vegetation=deformable 软通道）。

输出仍为 Semantic3D 格式语义 PLY（xyz + rgb + semantic_id），可直接喂 semantic_costmap_publisher.py。

用法：
  python3 scripts/separate_ground_ransac.py \
      -i /tmp/real_map_seg/real_map_map_semantic.ply \
      -o /tmp/real_map_seg/real_map_semantic_ground.ply \
      --ground-dist 0.05 --dry-run
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.materials import LIDARSEG_32, class_physics  # noqa: E402

_DT = {'float': '<f4', 'uchar': 'u1', 'int': '<i4'}


def read_ply(path):
    with open(path, 'rb') as f:
        n = 0
        props = []
        while True:
            line = f.readline().decode('ascii', 'ignore').strip()
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            elif line.startswith('property'):
                p = line.split()
                props.append((p[-1], p[1]))
            elif line == 'end_header':
                break
        dt = np.dtype([(name, _DT[t]) for name, t in props])
        return np.fromfile(f, dtype=dt, count=n)


def write_ply(path, xyz, rgb, sem):
    n = xyz.shape[0]
    with open(path, 'wb') as f:
        f.write('ply\nformat binary_little_endian 1.0\n'
                f'element vertex {n}\n'
                'property float x\nproperty float y\nproperty float z\n'
                'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                'property int semantic_id\nend_header\n'.encode('ascii'))
        for i in range(n):
            f.write(struct.pack('<fffBBB i', xyz[i, 0], xyz[i, 1], xyz[i, 2],
                                int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]), int(sem[i])))


def fit_ground_plane(xyz, dist_thresh=0.04, ransac_n=3, iters=2000):
    """RANSAC 拟合最低主水平面（法线朝上），返回 (a,b,c,d) 使 ax+by+cz+d=0。"""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    best = None
    best_inl = 0
    for _ in range(5):  # 多次 RANSAC，取法线朝上(c>0.8)且内点 z 均值最低的平面
        plane, inl = pcd.segment_plane(distance_threshold=dist_thresh,
                                       ransac_n=ransac_n, num_iterations=iters)
        a, b, c, d = plane
        inl = np.asarray(inl)
        z_mean = xyz[inl, 2].mean()
        if c > 0.8 and (best is None or z_mean < best_z):
            best = plane
            best_inl = len(inl)
            best_z = z_mean
    return best, best_inl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True)
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--ground-dist', type=float, default=0.05,
                    help='点到地面平面距离 < 该值判为地面（米）')
    ap.add_argument('--dry-run', action='store_true', help='只统计不写文件')
    args = ap.parse_args()

    data = read_ply(args.input)
    xyz = np.column_stack([data['x'], data['y'], data['z']]).astype(np.float32)
    rgb = np.column_stack([data['red'], data['green'], data['blue']]).astype(np.uint8)
    sem = data['semantic_id'].astype(np.int64)

    plane, _ = fit_ground_plane(xyz)
    a, b, c, d = plane
    dist = np.abs(a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d) / np.sqrt(a * a + b * b + c * c)
    ground_mask = dist < args.ground_dist
    print(f'地面平面 法线=({a:.3f},{b:.3f},{c:.3f}) d={d:.3f}')
    print(f'地面点 {int(ground_mask.sum()):,} ({ground_mask.mean() * 100:.1f}%)，其余保留语义')

    new_sem = sem.copy()
    new_sem[ground_mask] = 24  # driveable_surface → passable

    # 分层统计
    from collections import Counter
    cnt = Counter()
    for s, g in zip(new_sem, ground_mask):
        key = 'ground(24)' if g else LIDARSEG_32[int(s)]
        cnt[key] += 1
    print('分层点数：')
    for k, v in cnt.most_common():
        print(f'  {k:28s} {v:7d} ({v / len(sem) * 100:.1f}%)')

    # 代价分层（class_to_cost）
    def class_to_cost(cid):
        ph = class_physics(int(cid))
        safety = ph['safety']
        if safety == 'passable':
            return 0
        if safety == 'lethal':
            return 254
        L = float(ph['lethality'])
        if safety == 'hard':
            return int(150 + (L - 0.50) / 0.40 * 70)
        return int(30 + (L - 0.05) / 0.45 * 50)

    costs = np.array([class_to_cost(s) for s in new_sem])
    free = (costs == 0).sum(); hard = ((costs >= 150) & (costs < 254)).sum()
    soft = ((costs > 0) & (costs < 150)).sum()
    print(f'代价分层：free(0)={free}  hard(150-253)={hard}  soft(1-149)={soft}')

    if not args.dry_run:
        write_ply(args.output, xyz, rgb, new_sem)
        print(f'已写 {args.output}')


if __name__ == '__main__':
    main()
