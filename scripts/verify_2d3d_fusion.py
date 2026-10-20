#!/usr/bin/env python3
"""验证「6 相机 2D bbox 能否正确覆盖 3D 罕见类点」——门控投票有效性的前提。

对 val 集每个样本：把 3D 点投影到 6 个相机，统计罕见类 2D bbox 覆盖的 3D 点
其 GT 语义是否匹配 bbox 类别。对比两种关联方式（均为 6 相机并集）：
  - naive       ：点在任意相机 2D bbox 内即归属（忽略深度，会被遮挡点污染）
  - depth-gated ：点在 2D bbox 内 且 点深度 ∈ [depth_min-δ, depth_max+δ]

关键对比：单 CAM_FRONT 覆盖上限 ~12.6%，扩到 6 相机后覆盖率应大幅提升。

用法:
    python scripts/verify_2d3d_fusion.py \
        --seg_dir data/nuscenes_seg_full_16k \
        --proj data/nuscenes_projection_index.npz \
        --bbox data/nuscenes_2d_bbox_cache.npz \
        --depth_margin 0.5 \
        --max_samples 0
"""

import os
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

IMG_W, IMG_H = 1600, 900

RARE_NAMES = {
    2: 'ped.adult', 3: 'ped.child', 4: 'ped.worker', 5: 'ped.mobility',
    6: 'ped.police', 7: 'ped.stroller', 8: 'ped.wheelchair',
    9: 'barrier', 12: 'trafficcone', 14: 'bicycle', 21: 'motorcycle',
}


def project(xyz, K, T):
    """xyz (N,3) lidar -> (u, v, depth, valid)。depth 为 cam 系 z。"""
    N = len(xyz)
    pts_h = np.concatenate([xyz, np.ones((N, 1))], axis=1).astype(np.float64)
    pts_cam = (T @ pts_h.T).T  # N,4
    depth = pts_cam[:, 2]
    uv = (K @ pts_cam[:, :3].T)  # 2,N
    u = uv[0] / uv[2]
    v = uv[1] / uv[2]
    valid = (depth > 0) & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    return u, v, depth, valid


def assign(xyz, K_all, T_all, boxes, depth_margin, use_depth):
    """6 相机并集关联。返回 point_sem (N,)（-1=不在任何 bbox 内）。"""
    N = len(xyz)
    out = np.full(N, -1, dtype=np.int64)
    n_cam = K_all.shape[0]
    for cam_idx in range(n_cam):
        u, v, depth, valid = project(xyz, K_all[cam_idx], T_all[cam_idx])
        for row in boxes:
            x, y, w, h, sem_id = row[0], row[1], row[2], row[3], int(row[4])
            if int(row[7]) != cam_idx:
                continue
            in_box = valid & (u >= x) & (u < x + w) & (v >= y) & (v < y + h)
            if use_depth and row.shape[0] >= 7:
                dmin, dmax = row[5], row[6]
                in_box &= (depth >= dmin - depth_margin) & (depth <= dmax + depth_margin)
            out[in_box] = sem_id
    return out


def report(title, RARE_NAMES, gt_total, gt_covered, prop_total, prop_hit):
    """两个容易混淆的「覆盖」指标，都列出，不要混用：

    - **覆盖recall**（类别无关）：GT 类 c 的点落在**任意** 2D 框内即算覆盖
      （`gt_c & (pm >= 0)`）。衡量「2D 框圈住了多少罕见类点」，不分框的类别。
    - **匹配recall**（类别匹配）：GT 类 c 的点必须落在**类别为 c** 的框内
      （`gt_c & (pm == c)` == prop_hit）。论文表 16 的「罕见类 recall」用的是**这一个**
      （0.192/0.205/0.205），可由本表推出的 precision × 提案点 / GT点 复算。
      两者在行人类上几乎相等，但 barrier/trafficcone 在零样本下匹配recall 恒为 0
      （COCO 无此类名），而覆盖recall 仍有 ~0.005~0.013 的残值，差别就出在这里。
    """
    print('\n' + '=' * 82)
    print(title)
    print('=' * 82)
    print(f'{"类":<14} {"GT点":>9} {"覆盖recall":>11} {"匹配recall":>11} {"提案点":>9} {"precision":>10}')
    print('-' * 82)
    for c in sorted(RARE_NAMES):
        gt = gt_total[c]
        cov = gt_covered[c] / gt if gt else 0.0
        prop = prop_total[c]
        prec = prop_hit[c] / prop if prop else 0.0
        match = prop_hit[c] / gt if gt else 0.0
        print(f'{RARE_NAMES[c]:<14} {gt:>9,} {cov:>10.3f} {match:>11.3f} {prop:>9,} {prec:>10.3f}')
    g = sum(gt_total.values()); gc = sum(gt_covered.values())
    p = sum(prop_total.values()); ph = sum(prop_hit.values())
    print('-' * 82)
    print(f'罕见类汇总: GT={g:,}  覆盖recall={gc/g:.3f}  匹配recall={ph/g:.3f}  '
          f'提案={p:,}  precision={ph/p:.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg_dir', required=True)
    ap.add_argument('--proj', required=True)
    ap.add_argument('--bbox', required=True)
    ap.add_argument('--depth_margin', type=float, default=0.5)
    ap.add_argument('--max_samples', type=int, default=0, help='0=全部 val')
    args = ap.parse_args()

    proj = np.load(args.proj)
    bbox = np.load(args.bbox)
    has_depth = bbox['boxes'].shape[1] >= 7

    proj_idx = {t: i for i, t in enumerate(proj['tokens'])}
    bbox_idx = {t: i for i, t in enumerate(bbox['tokens'])}
    bbox_offsets = bbox['offsets']
    bbox_boxes = bbox['boxes']

    val_dir = os.path.join(args.seg_dir, 'val')
    val_files = sorted(f for f in os.listdir(val_dir) if f.endswith('.npz'))
    if args.max_samples:
        val_files = val_files[:args.max_samples]

    stat = {
        'naive': {'gt_total': defaultdict(int), 'gt_covered': defaultdict(int),
                  'prop_total': defaultdict(int), 'prop_hit': defaultdict(int)},
        'depth': {'gt_total': defaultdict(int), 'gt_covered': defaultdict(int),
                  'prop_total': defaultdict(int), 'prop_hit': defaultdict(int)},
    }

    n_noproj = n_nobox = 0
    for f in tqdm(val_files, desc='verifying 6-cam 2D->3D coverage'):
        tok = f[:-4]
        if tok not in proj_idx:
            n_noproj += 1
            continue
        d = np.load(os.path.join(val_dir, f))
        xyz = d['xyz'].astype(np.float64)
        sem = d['sem'].astype(np.int64)
        N = len(xyz)

        K_all = proj['K'][proj_idx[tok]].astype(np.float64)  # (6,3,3)
        T_all = proj['T'][proj_idx[tok]].astype(np.float64)  # (6,4,4)
        if tok in bbox_idx:
            bi = bbox_idx[tok]
            boxes = bbox_boxes[bbox_offsets[bi]:bbox_offsets[bi + 1]]  # (K,8)
        else:
            # 该帧不在 bbox 缓存里 = 检测器整帧无输出，必须**按零框计入分母**。
            # 早先这里直接 `continue` 整帧跳过，会把「检测器什么都没检出」的帧从
            # GT 分母里剔除，从而系统性虚高覆盖率——而且虚高幅度取决于检测器有多差：
            # 零样本 S2 被丢掉 1,191/6,019 帧，S3 只丢 157 帧，于是这个 bug 恰好
            # 单方向抬高 S3 的相对优势（实测 GT 分母 1,027,081 vs 1,156,755）。
            n_nobox += 1
            boxes = np.zeros((0, 8), np.float64)

        point_naive = assign(xyz, K_all, T_all, boxes, args.depth_margin, use_depth=False)
        point_depth = assign(xyz, K_all, T_all, boxes, args.depth_margin, use_depth=True)

        for c in RARE_NAMES:
            gt_c = sem == c
            for mode, pm in (('naive', point_naive), ('depth', point_depth)):
                s = stat[mode]
                s['gt_total'][c] += int(gt_c.sum())
                s['gt_covered'][c] += int((gt_c & (pm >= 0)).sum())
                s['prop_total'][c] += int((pm == c).sum())
                s['prop_hit'][c] += int(((pm == c) & gt_c).sum())

    print(f'\n[分母] val 帧 {len(val_files)}  无投影索引(跳过) {n_noproj}  '
          f'检测器无输出(按零框计入分母) {n_nobox}')

    report('naive 2D bbox（6 相机，无深度门控）', RARE_NAMES,
           stat['naive']['gt_total'], stat['naive']['gt_covered'],
           stat['naive']['prop_total'], stat['naive']['prop_hit'])
    report(f'depth-gated（6 相机，δ={args.depth_margin}m）', RARE_NAMES,
           stat['depth']['gt_total'], stat['depth']['gt_covered'],
           stat['depth']['prop_total'], stat['depth']['prop_hit'])


if __name__ == '__main__':
    main()
