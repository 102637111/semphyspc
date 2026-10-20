#!/usr/bin/env python3
"""实验C：实例分割几何基线（DBSCAN 欧式聚类），量化级联第②环的下界。

背景：
  级联三环 = ①地面几何分割 ②实例分割 ③物体判类。①③已证，②是缺的硬骨头。
  本脚本用纯几何欧式聚类(DBSCAN)对非地面 thing 点聚类，与真值 inst 标签对齐，
  测实例分割的召回/精度/PQ，作为「学习式实例分割」要超越的基线。

方法：
  1. 取 thing 类(1..23)且 inst>0 的点（用真值非地面，隔离①的误差，只看②本身）。
  2. open3d cluster_dbscan(eps=0.5m, min_points=20) 欧式聚类。
  3. 贪心匹配：预测簇 vs 真值实例，IoU>0.5 记为命中。
  4. 报召回/精度/PQ，并按实例大小分桶看小物体表现。

用法:
    python scripts/analyze_instance_geom.py \
        --data data/nuscenes_seg_full_16k/val --n 500 --eps 0.5 --min_points 20 --seed 0
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THING_CLASSES = set(range(1, 24))
IOU_THRESH = 0.5


def cluster(xyz, eps, min_points):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points,
                                           print_progress=False))
    return labels  # -1 = 噪声


def match_instances(gt_labels, pred_labels):
    """贪心匹配预测簇与真值实例，返回 (matched_ious, tp, fp, fn, matched_gt_set)."""
    gt_ids = np.unique(gt_labels)
    pred_ids = np.unique(pred_labels[pred_labels >= 0])
    if len(pred_ids) == 0:
        return [], 0, 0, len(gt_ids), set()

    # 每个簇/实例的点集
    pred_sets = {p: set(np.where(pred_labels == p)[0]) for p in pred_ids}
    gt_sets = {g: set(np.where(gt_labels == g)[0]) for g in gt_ids}

    # 按簇大小降序贪心
    matched_ious = []
    matched_gt = set()
    matched_pred = set()
    for p in sorted(pred_sets, key=lambda x: -len(pred_sets[x])):
        best_g, best_iou = None, 0.0
        for g in gt_sets:
            if g in matched_gt:
                continue
            inter = len(pred_sets[p] & gt_sets[g])
            union = len(pred_sets[p] | gt_sets[g])
            iou = inter / max(union, 1)
            if iou > best_iou:
                best_iou, best_g = iou, g
        if best_iou >= IOU_THRESH:
            matched_ious.append(best_iou)
            matched_gt.add(best_g)
            matched_pred.add(p)

    tp = len(matched_ious)
    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - len(matched_gt)
    return matched_ious, tp, fp, fn, matched_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--eps', type=float, default=0.5)
    ap.add_argument('--min_points', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    rng = np.random.default_rng(args.seed)
    if args.n < len(files):
        files = rng.choice(files, args.n, replace=False).tolist()

    print(f"[experimentC] {len(files)} 样本, DBSCAN eps={args.eps}m min_pts={args.min_points}",
          flush=True)

    sum_iou = 0.0
    tp = fp = fn = 0
    n_gt_inst = 0
    n_pred_inst = 0
    # 按实例大小分桶统计召回
    size_bins = [(0, 50), (50, 200), (200, 1000), (1000, 10**9)]
    bin_tp = {b: 0 for b in size_bins}
    bin_gt = {b: 0 for b in size_bins}
    t0 = time.time()

    for fi, fp_ in enumerate(files):
        d = np.load(fp_)
        xyz = d['xyz'].astype(np.float32)
        sem = d['sem'].astype(np.int64)
        inst = d['inst'].astype(np.int64)

        mask = np.isin(sem, list(THING_CLASSES)) & (inst > 0)
        if mask.sum() < args.min_points:
            continue
        xyz_t = xyz[mask]
        gt = inst[mask]  # 真值实例 id（帧内唯一）

        pred = cluster(xyz_t, args.eps, args.min_points)

        matched_ious, t, f_p, f_n, matched_gt = match_instances(gt, pred)
        sum_iou += float(np.sum(matched_ious))
        tp += t
        fp += f_p
        fn += f_n
        n_gt_inst += len(np.unique(gt))
        n_pred_inst += len(np.unique(pred[pred >= 0]))

        # 按大小分桶
        for g in np.unique(gt):
            sz = int((gt == g).sum())
            b = next(b for b in size_bins if b[0] <= sz < b[1])
            bin_gt[b] += 1
            if g in matched_gt:
                bin_tp[b] += 1

        if (fi + 1) % 100 == 0:
            print(f"  [{fi+1}/{len(files)}] 用时 {time.time()-t0:.0f}s", flush=True)

    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    pq = sum_iou / max(tp + 0.5 * fp + 0.5 * fn, 1)
    mean_iou = sum_iou / max(tp, 1)

    print("\n================ 实验C 结果（实例分割几何基线）================", flush=True)
    print(f"  样本数 {len(files)}, 真值实例 {n_gt_inst}, 预测簇 {n_pred_inst}", flush=True)
    print(f"  召回(recall)  : {rec:.4f}", flush=True)
    print(f"  精度(precision): {prec:.4f}", flush=True)
    print(f"  PQ            : {pq:.4f}", flush=True)
    print(f"  命中实例平均IoU: {mean_iou:.4f}", flush=True)

    print("\n  按实例大小分桶的召回:", flush=True)
    for b in size_bins:
        r = bin_tp[b] / max(bin_gt[b], 1)
        print(f"    {b[0]:>5}-{b[1]:<6} 点 : {r:.4f}  (GT实例 {bin_gt[b]})", flush=True)
    print("======================================================================", flush=True)


if __name__ == '__main__':
    main()
