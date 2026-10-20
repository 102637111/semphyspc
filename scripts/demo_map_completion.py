#!/usr/bin/env python3
"""实验J（可行性验证）：语义引导 vs 纯几何 的地图规则结构补全。

核心命题：规则结构（地面等平面类）的采集损毁，可用「几何(平面拟合)+语义(知道该补成平面)」
补回；语义引导的价值随「洞周围结构混合程度」增强而显现——纯地面时纯几何已够，地面+墙面
交界时语义引导才能分离出正确平面。

方法：在 Semantic3D 一帧上，选三个「离墙面远近」不同的地面点作洞中心（近墙/中间/纯地面），
各挖半径 R 的洞，用洞外环带做 RANSAC 平面拟合，对比两种点来源：
  - 纯几何  : 环带内所有点（会被墙面/植被干扰）
  - 语义引导: 环带内仅地面点
度量（越干净越有说服力）：
  1) 法向夹角：拟合平面法向 vs 整帧真值地面法向（0°=方向完全正确）
  2) 补全点 vs 洞内真值地面的 mean 距离 + 覆盖率(<0.5m)

语义来源说明：用真值语义证明「语义先验」上界价值；实际部署换 PointMambaSeg 预测语义
（地面/墙面常见类预测较准，见论文 §4.2）。

用法:
    python scripts/demo_map_completion.py \
        --scene /mnt/data/datasets/Semantic3D/seg_npz_labeled/train/domfountain_station3.npz \
        --R 3.0 --band 2.0 --step 0.5
"""

import argparse
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d

GROUND, BUILD = 0, 4


def ransac_plane(pts, thresh=0.3, iters=1500):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    plane, inliers = pcd.segment_plane(distance_threshold=thresh, ransac_n=3,
                                       num_iterations=iters)
    return np.asarray(plane), np.asarray(inliers)


def normal(plane):
    n = np.array(plane[:3]); return n / np.linalg.norm(n)


def angle_deg(n1, n2):
    c = abs(np.dot(n1, n2)); c = min(1.0, c)
    return float(np.degrees(np.arccos(c)))


def sample_disk(plane, center, R, s):
    a, b, c, d = plane
    n = np.array([a, b, c]); n = n / np.linalg.norm(n)
    p0 = -d * n
    c = center - np.dot(center - p0, n) * n
    u = np.array([1.0, 0, 0])
    if abs(np.dot(u, n)) > 0.9:
        u = np.array([0.0, 1.0, 0])
    u -= np.dot(u, n) * n; u /= np.linalg.norm(u)
    v = np.cross(n, u)
    xs = np.arange(-R, R + 1e-9, s)
    pts = [c + x * u + y * v for x in xs for y in xs if x * x + y * y <= R * R]
    return np.array(pts, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='/mnt/data/datasets/Semantic3D/seg_npz_labeled/train/domfountain_station3.npz')
    ap.add_argument('--R', type=float, default=3.0)
    ap.add_argument('--band', type=float, default=2.0)
    ap.add_argument('--step', type=float, default=0.5)
    args = ap.parse_args()

    d = np.load(args.scene)
    xyz = d['xyz'].astype(np.float64)
    sem = d['sem'].astype(np.int64)
    g = sem == GROUND
    b = sem == BUILD

    # 真值地面法向（整帧地面点拟合）
    gt_plane, _ = ransac_plane(xyz[g])
    n_gt = normal(gt_plane)
    print(f'真值地面法向 = {n_gt.round(3)}', flush=True)

    # 三个洞中心：按「离最近墙面距离」取 近墙/中间/纯地面
    tree_b = cKDTree(xyz[b])
    d_g2b, _ = tree_b.query(xyz[g])
    order = np.argsort(d_g2b)
    picks = {'近墙(0.1m)': order[0], '中间': order[len(order) // 2], '纯地面': order[-1]}

    print(f'\n{"洞位置":<14s} {"洞内地面/总":>10s} | {"方法":<12s} {"法向夹角":>8s} '
          f'{"mean":>7s} {"RMSE":>7s} {"覆盖<0.5m":>9s}', flush=True)
    print('-' * 78, flush=True)

    for qname, qi in picks.items():
        center = xyz[g][qi]
        dist = np.linalg.norm(xyz - center, axis=1)
        inside = dist < args.R
        ring = (dist >= args.R) & (dist < args.R + args.band)
        gt_inside = xyz[inside & g]
        ng = int(inside.sum()); nt = int(gt_inside.shape[0])

        for mname, pts in [('纯几何', xyz[ring]), ('语义引导', xyz[ring & g])]:
            if len(pts) < 10:
                continue
            plane, inl = ransac_plane(pts)
            ang = angle_deg(normal(plane), n_gt)
            fill = sample_disk(plane, center, args.R, args.step)
            dd, _ = cKDTree(gt_inside).query(fill) if len(gt_inside) else (np.full(len(fill), 9e9), None)
            mean = float(dd.mean()); rmse = float(np.sqrt((dd ** 2).mean()))
            cov = float((dd < 0.5).mean())
            if mname == '纯几何':
                print(f'{qname:<14s} {nt:>4d}/{ng:<4d} | {mname:<12s} {ang:>7.1f}° '
                      f'{mean:>6.2f}m {rmse:>6.2f}m {cov:>8.1%}', flush=True)
            else:
                print(f'{"":<14s} {"":>10s} | {mname:<12s} {ang:>7.1f}° '
                      f'{mean:>6.2f}m {rmse:>6.2f}m {cov:>8.1%}', flush=True)


if __name__ == '__main__':
    main()
