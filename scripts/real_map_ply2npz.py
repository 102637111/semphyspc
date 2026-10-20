#!/usr/bin/env python3
"""真实建图点云 → 语义推理输入：把 PCL 格式的 captured_map.ply 转成 infer_map.py 吃的 npz（xyz+rgb）。

真实点云来自 RTAB-Map / ORB-SLAM3 RGB-D 建图（orbslam3_nav2_bringup/scripts/captured_map.ply），
坐标即 map 帧（已验证与 my_map.yaml origin(-5.63,-5.4) 对齐：71.8% 点落在 free 区、20.6% 落墙）。
PCL PLY 每个顶点带 x y z rgb nx ny nz curvature（可选 view_*/focal/k* 相机元数据），
这里只取前 7 个属性（x y z r g b）转 npz，供 infer_map.py --npz 做语义推理。

用法：
  python3 scripts/real_map_ply2npz.py -i <captured_map.ply> -o /tmp/real_map.npz [--max-points N]
"""
import argparse

import numpy as np

# PCL binary PLY 顶点属性顺序（前 10 个）：x y z r g b nx ny nz curvature
_DT10 = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                  ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
                  ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'), ('curv', '<f4')])


def parse_pcl_ply(path):
    """解析 PCL binary_little_endian PLY，返回 (xyz (N,3) float32, rgb (N,3) float32 [0,1])。"""
    n_vertex = 0
    with open(path, 'rb') as f:
        while True:
            line = f.readline()
            if line.startswith(b'element vertex'):
                n_vertex = int(line.split()[-1])
            elif line.startswith(b'end_header'):
                break
        arr = np.fromfile(f, dtype=_DT10, count=n_vertex)
    xyz = np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float32)
    rgb = np.column_stack([arr['r'], arr['g'], arr['b']]).astype(np.float32) / 255.0
    return xyz, rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True, help='PCL binary PLY（captured_map.ply）')
    ap.add_argument('-o', '--out', default='/tmp/real_map.npz')
    ap.add_argument('--max-points', type=int, default=0, help='>0 时只保留前 N 点（快速验证）')
    args = ap.parse_args()

    xyz, rgb = parse_pcl_ply(args.input)
    if args.max_points:
        xyz, rgb = xyz[:args.max_points], rgb[:args.max_points]

    # 剔除 NaN / 无效点（真实建图偶尔有无效深度点）
    valid = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[valid], rgb[valid]

    np.savez_compressed(args.out, xyz=xyz, rgb=rgb)
    print(f'转换 {args.input} -> {args.out}')
    print(f'  有效点 {len(xyz):,}（剔除 NaN {int((~valid).sum()):,}）')
    print(f'  x[{xyz[:, 0].min():.2f},{xyz[:, 0].max():.2f}] '
          f'y[{xyz[:, 1].min():.2f},{xyz[:, 1].max():.2f}] '
          f'z[{xyz[:, 2].min():.2f},{xyz[:, 2].max():.2f}]')


if __name__ == '__main__':
    main()
