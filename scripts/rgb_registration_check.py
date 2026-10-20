#!/usr/bin/env python3
"""检验「给点云上色」的投影链是否正确，用独立的手工 2D 框当裁判（纯 CPU）。

背景：`build_nuscenes_seg_dataset.py` 与 `build_projection_index.py` 都用的是**简化链**

    T = inv(T_ego<-cam) @ T_ego<-lidar

即假设「LiDAR 关键帧与相机共享同一个 ego 位姿」，注释里的理由是「关键帧同步」。
但 nuScenes 的 sample_data 各有自己的 timestamp 与 ego_pose：同一 sample 下 6 个相机的
时间戳与 LiDAR 相差可达数十毫秒，10 m/s 下车体已移动零点几米，在 20 m 处折合图像上
几十像素。devkit 的 `map_pointcloud_to_image` 用的是**严格链**：

    p_cam = inv(ego_pose_cam) · inv(cs_cam) · ego_pose_lidar · cs_lidar · p_lidar

本脚本对同一批点分别用两条链投影，然后拿**手工标注的 2D 框**（`data/s3_det_labels_val.npz`，
由 3D 真值框投影而来、但按相机逐个投影并用可见性判据筛过）作为独立裁判，比较：

  - 像素偏移：两条链的 (Δu, Δv) 分布，以及它随距离的增长；
  - 命中率：点的语义类别与所在 2D 框类别一致的比率（配准正确的必要条件）；
  - 污染率：点落进**类别不符**的框里的比率（配准错位的典型症状）。

注意一个与检验目标无关但会影响绝对命中率的因素：nuScenes 图像**未去畸变**，
`camera_intrinsic` 直接用在原始畸变图上（devkit 自己也这么做），所以图像边缘
天然有几到几十像素误差。两条链共享这一误差，因此**比较是公平的**，绝对值则会偏低。

用法：
    python3 scripts/rgb_registration_check.py --n_frames 60 \
        --out results/rgb_registration_val.json
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CAM_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# 语义索引 → COCO 索引（与 scripts/safety_metrics.py 保持一致）
SEM_TO_COCO = {2: 0, 14: 1, 21: 3, 9: 80, 12: 81, 17: 2, 16: 5, 15: 5, 23: 7, 22: 7}

DEPTH_BINS = [(0, 10), (10, 20), (20, 40), (40, 1e9)]


def pose_matrix(rec):
    from pyquaternion import Quaternion
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Quaternion(rec['rotation']).rotation_matrix
    T[:3, 3] = rec['translation']
    return T


def chains(nusc, lidar_token):
    """返回 (T_simple (6,4,4), T_strict (6,4,4), K (6,3,3), 每相机的 dt 秒)。"""
    sd_l = nusc.get('sample_data', lidar_token)
    sample = nusc.get('sample', sd_l['sample_token'])
    cs_l = nusc.get('calibrated_sensor', sd_l['calibrated_sensor_token'])
    T_ego_l = pose_matrix(cs_l)                      # lidar -> ego(标定)
    P_l = pose_matrix(nusc.get('ego_pose', sd_l['ego_pose_token']))
    T_lidar_global = P_l @ T_ego_l                   # lidar -> global

    Ts, Tt, Ks, dts = [], [], [], []
    for cam in CAM_CHANNELS:
        sd_c = nusc.get('sample_data', sample['data'][cam])
        cs_c = nusc.get('calibrated_sensor', sd_c['calibrated_sensor_token'])
        T_ego_c = pose_matrix(cs_c)
        P_c = pose_matrix(nusc.get('ego_pose', sd_c['ego_pose_token']))
        Ts.append(np.linalg.inv(T_ego_c) @ T_ego_l)              # 简化：忽略双方 ego_pose
        Tt.append(np.linalg.inv(T_ego_c) @ np.linalg.inv(P_c) @ T_lidar_global)   # 严格（devkit）
        Ks.append(np.array(cs_c['camera_intrinsic'], dtype=np.float64))
        dts.append((sd_c['timestamp'] - sd_l['timestamp']) / 1e6)
    return np.stack(Ts), np.stack(Tt), np.stack(Ks), np.array(dts)


def project(xyz, K, T):
    """返回 (u, v, depth, inside)。inside 只表示落在图像内且在前方。"""
    h = np.c_[xyz.astype(np.float64), np.ones(len(xyz))]
    cam = h @ T[:, :4].T                     # (N,3)
    z = cam[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = K[0, 0] * cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return u, v, z, np.isfinite(u) & np.isfinite(v) & (z > 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/nuscenes_seg_full_16k')
    ap.add_argument('--split', default='val')
    ap.add_argument('--gt', default='data/s3_det_labels_val.npz')
    ap.add_argument('--dataroot', default='/mnt/data/nuscenes')
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--n_frames', type=int, default=60)
    ap.add_argument('--out', default='results/rgb_registration_val.json')
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    gt = np.load(args.gt, allow_pickle=True)
    gt_tok = {str(t): i for i, t in enumerate(gt['tokens'])}
    g_boxes, g_off = gt['boxes'], gt['offsets']

    files = sorted(os.listdir(os.path.join(args.data, args.split)))[:args.n_frames]
    offs, hit, bad = [], {'simple': 0, 'strict': 0}, {'simple': 0, 'strict': 0}
    tot = {'simple': 0, 'strict': 0}
    dt_all = []

    for n, fn in enumerate(files):
        tok = fn[:-4]
        d = np.load(os.path.join(args.data, args.split, fn))
        xyz, sem = d['xyz'], d['sem']
        Ts, Tt, Ks, dts = chains(nusc, tok)
        dt_all.extend(dts.tolist())
        gi = gt_tok.get(tok)
        boxes = g_boxes[g_off[gi]:g_off[gi + 1]] if gi is not None else np.zeros((0, 9), np.float32)

        for ci, cam in enumerate(CAM_CHANNELS):
            img_path = os.path.join(nusc.dataroot,
                                    nusc.get('sample_data',
                                             nusc.get('sample', nusc.get('sample_data', tok)['sample_token'])['data'][cam])['filename'])
            if not os.path.exists(img_path):
                continue
            im = np.array(Image.open(img_path))
            H, W = im.shape[:2]
            cb = boxes[boxes[:, 7].astype(int) == ci] if len(boxes) else boxes
            for name, T in (('simple', Ts[ci]), ('strict', Tt[ci])):
                u, v, z, fwd = project(xyz, Ks[ci], T)
                ins = fwd & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                tot[name] += int(ins.sum())
                if name == 'simple':
                    us, vs, zs, inss = u, v, z, ins
                else:
                    # 像素偏移只统计两条链都在图内的点
                    m = ins & inss
                    if m.sum():
                        offs.append(np.stack([np.abs(u[m] - us[m]), np.abs(v[m] - vs[m]),
                                              zs[m]], axis=1).astype(np.float32))
                # 命中/污染：只看有 COCO 对应类的点
                ok_cls = np.array([SEM_TO_COCO.get(int(s), -1) >= 0 for s in sem])
                sel = ins & ok_cls
                if not sel.any() or not len(cb):
                    continue
                uu, vv, ss = u[sel], v[sel], sem[sel]
                want = np.array([SEM_TO_COCO[int(s)] for s in ss])
                mh = np.zeros(len(uu), bool)          # 命中自己的类
                mb = np.zeros(len(uu), bool)          # 落进别的类的框
                for b in cb:
                    x, y, w, h = b[0], b[1], b[2], b[3]
                    inside = (uu >= x) & (uu <= x + w) & (vv >= y) & (vv <= y + h)
                    if not inside.any():
                        continue
                    same = int(b[4]) == want
                    mh |= inside & same
                    mb |= inside & ~same
                hit[name] += int(mh.sum())
                bad[name] += int(mb.sum())
        if (n + 1) % 20 == 0:
            print(f'  已处理 {n + 1}/{len(files)} 帧', flush=True)

    offs = np.concatenate(offs) if offs else np.zeros((0, 3), np.float32)
    dt_all = np.array(dt_all)

    res = {
        'n_frames': len(files),
        'cam_dt_ms': {'mean_abs': float(np.abs(dt_all).mean() * 1000),
                      'p95_abs': float(np.percentile(np.abs(dt_all), 95) * 1000),
                      'max_abs': float(np.abs(dt_all).max() * 1000)},
        'pixel_offset': {},
        'hit_rate': {}, 'mismatch_rate': {}, 'n_points': tot,
    }
    print('\n' + '=' * 72)
    print('相机 sample_data 与 LiDAR 关键帧的时间差（ms）')
    print(f"  平均 |dt| {res['cam_dt_ms']['mean_abs']:.1f}   p95 {res['cam_dt_ms']['p95_abs']:.1f}   "
          f"最大 {res['cam_dt_ms']['max_abs']:.1f}")
    print('=' * 72)
    print(f"\n{'深度区间':<14}{'点数':>10}{'|Δu| 均值':>12}{'|Δu| p95':>11}{'|Δv| 均值':>12}{'|Δv| p95':>11}")
    print('-' * 72)
    for lo, hi in DEPTH_BINS:
        m = (offs[:, 2] >= lo) & (offs[:, 2] < hi)
        if m.sum() == 0:
            continue
        nm = f'{lo:.0f}–{hi:.0f} m' if hi < 1e8 else f'>{lo:.0f} m'
        res['pixel_offset'][nm] = {
            'n': int(m.sum()),
            'du_mean': round(float(offs[m, 0].mean()), 2),
            'du_p95': round(float(np.percentile(offs[m, 0], 95)), 2),
            'dv_mean': round(float(offs[m, 1].mean()), 2),
            'dv_p95': round(float(np.percentile(offs[m, 1], 95)), 2)}
        r = res['pixel_offset'][nm]
        print(f"{nm:<14}{r['n']:>10}{r['du_mean']:>12.2f}{r['du_p95']:>11.2f}"
              f"{r['dv_mean']:>12.2f}{r['dv_p95']:>11.2f}")

    print('\n' + '=' * 72)
    print('用独立手工 2D 框当裁判（只统计有 COCO 对应类的点）')
    print('=' * 72)
    print(f"{'投影链':<12}{'图内点数':>12}{'命中同类框':>13}{'落进他类框':>13}{'背景':>10}")
    print('-' * 72)
    for name in ('simple', 'strict'):
        t = max(tot[name], 1)
        res['hit_rate'][name] = round(hit[name] / t, 4)
        res['mismatch_rate'][name] = round(bad[name] / t, 4)
        print(f"{name:<12}{tot[name]:>12}{hit[name] / t * 100:>12.1f}%"
              f"{bad[name] / t * 100:>12.1f}%{(t - hit[name] - bad[name]) / t * 100:>9.1f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f'\n已写出：{args.out}')


if __name__ == '__main__':
    main()
