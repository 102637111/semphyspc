#!/usr/bin/env python3
"""实验A：用纯几何线索(法向量+高度)做「地面 / 非地面」二分类，量化几何法能白送多少。

背景：
  语义 32 类端到端 mIoU 只有 0.2342，但下游真正需要的第一步是把地图分成
  「地面(可通行)」和「物体(需判类/碰撞)」。地面有极强几何线索，理论上不训练也能分。
  本脚本验证这一点：地面真值 = nuScenes-lidarseg 平坦类 {24 driveable, 25 flat.other,
  26 sidewalk, 27 terrain}，其余为非地面。用几何法预测地面，算二分类 IoU 与混淆。

方法(纯几何，不训练)：
  1. open3d 估计逐点法向量 n，水平面判定 |n_z| > 0.75（法向量近竖直，含斜坡）。
  2. 高度：地面在场景最低处附近，floor = z 的 10 分位；低点 = z < floor + 0.5m。
  3. 预测地面 = 水平面 且 低点。

输出：
  - 地面二分类的 IoU / precision / recall（几何法白送多少）。
  - 地面占全部点的比例（这比例意味着场景有多大一部分被几何法"解决"）。
  - 非地面类被误判为地面的比例（哪些类最容易被当空地，如屋顶/植被）。

用法:
    python scripts/analyze_ground_geom.py --data data/nuscenes_seg_full_16k/val --n 200 --seed 0
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GROUND_CLASSES = {24, 25, 26, 27}  # driveable / flat.other / sidewalk / terrain
NORMAL_THRESH = 0.75
HEIGHT_MARGIN = 0.5

# 32 类名（与 utils/materials.py LIDARSEG_32 一致，仅打印用）
CLASS_NAMES = {
    0: 'noise', 1: 'animal',
    2: 'ped.adult', 3: 'ped.child', 4: 'ped.worker', 5: 'ped.mobility',
    6: 'ped.police', 7: 'ped.stroller', 8: 'ped.wheelchair',
    9: 'barrier', 10: 'debris', 11: 'pushable', 12: 'trafficcone',
    13: 'bicycle_rack', 14: 'bicycle',
    15: 'bus.bendy', 16: 'bus.rigid', 17: 'car', 18: 'construction',
    19: 'ambulance', 20: 'police_car', 21: 'motorcycle', 22: 'trailer',
    23: 'truck', 24: 'driveable', 25: 'flat_other', 26: 'sidewalk', 27: 'terrain',
    28: 'manmade', 29: 'other', 30: 'vegetation', 31: 'ego',
}


def estimate_ground_mask(xyz):
    """xyz (N,3) lidar z-up -> 预测地面布尔掩码 (N,)."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    normals = np.asarray(pcd.normals)  # (N,3)，方向有符号歧义，取 |n_z|
    nz_abs = np.abs(normals[:, 2])

    z = xyz[:, 2]
    floor = np.percentile(z, 10)

    horizontal = nz_abs > NORMAL_THRESH
    low = z < floor + HEIGHT_MARGIN
    return horizontal & low


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='nuScenes 子集目录(如 .../val)')
    ap.add_argument('--n', type=int, default=200, help='采样样本数')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import glob
    files = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    rng = np.random.default_rng(args.seed)
    if args.n < len(files):
        files = rng.choice(files, args.n, replace=False).tolist()

    print(f"[experimentA] 采样 {len(files)} 个样本, 几何法阈值 "
          f"|n_z|>{NORMAL_THRESH}, 高度 z<floor+{HEIGHT_MARGIN}m", flush=True)

    # 聚合：二分类混淆 + 每类被误判为地面的比例
    tp_c = fp_c = tn_c = fn_c = 0
    n_ground_pts = 0
    n_total_pts = 0
    # 误判统计：真值为某类 -> 被预测为地面的比例 (fp_rate[class])
    fp_rate_sum = np.zeros(32, dtype=np.float64)
    fp_rate_cnt = np.zeros(32, dtype=np.int64)
    t0 = time.time()

    for fi, fp in enumerate(files):
        d = np.load(fp)
        xyz = d['xyz'].astype(np.float32)
        sem = d['sem'].astype(np.int64)

        true_ground = np.isin(sem, list(GROUND_CLASSES))
        pred_ground = estimate_ground_mask(xyz)

        tp_c += int((pred_ground & true_ground).sum())
        fp_c += int((pred_ground & ~true_ground).sum())
        tn_c += int((~pred_ground & ~true_ground).sum())
        fn_c += int((~pred_ground & true_ground).sum())

        n_ground_pts += int(true_ground.sum())
        n_total_pts += sem.shape[0]

        # 每类被误判为地面的比例：fp 点按真值类累加
        fp_mask = pred_ground & ~true_ground
        if fp_mask.any():
            cls, cnt = np.unique(sem[fp_mask], return_counts=True)
            for c, k in zip(cls, cnt):
                fp_rate_sum[c] += k
                fp_rate_cnt[c] += 1

        if (fi + 1) % 100 == 0:
            print(f"  [{fi+1}/{len(files)}] 用时 {time.time()-t0:.0f}s", flush=True)

    # 汇总
    iou = tp_c / max(tp_c + fp_c + fn_c, 1)
    prec = tp_c / max(tp_c + fp_c, 1)
    rec = tp_c / max(tp_c + fn_c, 1)
    ground_frac = n_ground_pts / max(n_total_pts, 1)

    print("\n================ 实验A 结果（地面 vs 非地面，纯几何法）================", flush=True)
    print(f"  样本数           : {len(files)}", flush=True)
    print(f"  地面点占比       : {ground_frac*100:.1f}%  (这部分被几何法直接覆盖)", flush=True)
    print(f"  二分类 IoU       : {iou:.4f}", flush=True)
    print(f"  二分类 precision : {prec:.4f}", flush=True)
    print(f"  二分类 recall    : {rec:.4f}", flush=True)
    print(f"  混淆            : TP={tp_c} FP={fp_c} TN={tn_c} FN={fn_c}", flush=True)

    print("\n  最容易被误判为「地面」的非地面类（FP 占比高的类，按 FP 点数排序）:", flush=True)
    order = np.argsort(-fp_rate_sum)
    for c in order[:8]:
        if fp_rate_sum[c] <= 0:
            continue
        name = CLASS_NAMES.get(int(c), str(int(c)))
        print(f"    class {int(c):2d} {name:22s} : FP={fp_rate_sum[c]:.0f}", flush=True)

    print("======================================================================", flush=True)


if __name__ == '__main__':
    main()
