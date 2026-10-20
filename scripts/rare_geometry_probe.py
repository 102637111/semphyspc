#!/usr/bin/env python3
"""用地面当参考，量罕见类实例的「离地高度 + 几何尺寸」，看它们和 ped.adult 分不分得开。

动机：§4.14 把罕见类全灭归因到「分组粒度（主因）+ 标注量（不可约下限）」。但分组粒度
只解释了 bicycle/motorcycle/trafficcone/cw/barrier 这一档（A→B 2.57×）。剩下 5 个类
（child / police / stroller / pers_mob / wheelchair）在 g1024 下**仍然是 0**。一个自然
的追问是：这些类在物理几何上到底有没有可区分的特征？如果有，就说明还值得救；
如果 police 和 adult 的几何完全重叠，那它就真的只能靠 RGB 外观。

做法（纯 CPU，不碰 GPU）：
  1. 每帧用 GT 地面类（driveable / flat.other / sidewalk / terrain）迭代最小二乘拟合
     平面 z = a x + b y + c（两轮残差裁剪，抗离群）。
  2. 每个实例（键 (sem, inst)，inst >= 0）算：
       height   = p95(HAG)          离地高度（抗头顶零星点）
       height_mx= max(HAG)          最高点
       vert_ext = p95 - p05         竖直跨度
       foot_L/W = XY 主成分两轴跨度  水平尺寸
       n_pts, dist = 点数、到自车水平距离
  3. 按类别汇总分位数。

用法：
    python3 scripts/rare_geometry_probe.py --split data/nuscenes_seg_full_16k/val \
        --out results/rare_geometry_val.json
"""
import argparse
import glob
import json
import os

import numpy as np

NAMES = ['noise', 'animal', 'ped.adult', 'ped.child', 'ped.cw', 'ped.pers_mob', 'ped.police',
         'ped.stroller', 'ped.wheelchair', 'barrier', 'debris', 'pushable', 'trafficcone',
         'bicycle_rack', 'bicycle', 'bus.bendy', 'bus.rigid', 'car', 'construction',
         'ambulance', 'emergency.police', 'motorcycle', 'trailer', 'truck', 'driveable',
         'flat.other', 'sidewalk', 'terrain', 'manmade', 'static.other', 'vegetation', 'ego']

GROUND = (24, 25, 26, 27)          # driveable / flat.other / sidewalk / terrain
# 关注的类：5 个「全灭」的 + 作为参照的 ped.adult，外加两个已知可救的做对照
TARGET = [2, 3, 5, 6, 7, 8, 9, 12, 14, 21]
MIN_PTS = 5                        # 少于这么多点的实例不统计（大多是被截断的远目标）


def fit_ground(xyz, sem):
    """迭代最小二乘拟合地面平面，返回 (a, b, c)，HAG = z - (a x + b y + c)。"""
    m = np.isin(sem, GROUND)
    g = xyz[m]
    if len(g) < 50:
        return None
    A = np.c_[g[:, 0], g[:, 1], np.ones(len(g))]
    coef = np.linalg.lstsq(A, g[:, 2], rcond=None)[0]
    for _ in range(2):
        r = g[:, 2] - A @ coef
        keep = np.abs(r - np.median(r)) < 0.3
        if keep.sum() < 50:
            break
        g, A = g[keep], A[keep]
        coef = np.linalg.lstsq(A, g[:, 2], rcond=None)[0]
    return coef


def inst_features(p, a, b, c):
    """p: (n,3) 一个实例的点。返回特征字典，点太少返回 None。"""
    hag = p[:, 2] - (a * p[:, 0] + b * p[:, 1] + c)
    xy = p[:, :2]
    xy = xy - xy.mean(0)
    # XY 主成分：长轴/短轴跨度
    if len(xy) >= 3:
        cov = np.cov(xy.T)
        evals, evecs = np.linalg.eigh(cov)
        major = evecs[:, -1]
        minor = evecs[:, 0]
        proj_maj = xy @ major
        proj_min = xy @ minor
        foot_L = float(proj_maj.max() - proj_maj.min())
        foot_W = float(proj_min.max() - proj_min.min())
    else:
        foot_L = foot_W = float(np.ptp(xy, axis=0).max())
    return dict(
        height=float(np.percentile(hag, 95)),
        height_mx=float(hag.max()),
        vert_ext=float(np.percentile(hag, 95) - np.percentile(hag, 5)),
        foot_L=foot_L, foot_W=foot_W,
        n_pts=int(len(p)),
        dist=float(np.hypot(p[:, 0].mean(), p[:, 1].mean())),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='data/nuscenes_seg_full_16k/val')
    ap.add_argument('--out', default='results/rare_geometry_val.json')
    ap.add_argument('--min_pts', type=int, default=MIN_PTS)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.split, '*.npz')))
    print(f'{len(files)} 帧，来自 {args.split}')
    per_cls = {c: [] for c in TARGET}
    n_inst_frames = 0
    bad_ground = 0

    for i, f in enumerate(files):
        d = np.load(f)
        xyz, sem, inst = d['xyz'], d['sem'], d['inst']
        coef = fit_ground(xyz, sem)
        if coef is None:
            bad_ground += 1
            continue
        a, b, c = coef
        # 只挑关注类里、属于真实例的点
        sel = np.isin(sem, TARGET) & (inst >= 0)
        if not sel.any():
            continue
        s_s, i_s, p_s = sem[sel], inst[sel], xyz[sel]
        for key in set(zip(s_s.tolist(), i_s.tolist())):
            cls, iid = key
            m = (s_s == cls) & (i_s == iid)
            if m.sum() < args.min_pts:
                continue
            feat = inst_features(p_s[m], a, b, c)
            feat['cls'] = int(cls)
            per_cls[cls].append(feat)
            n_inst_frames += 1
        if (i + 1) % 1000 == 0:
            print(f'  {i + 1}/{len(files)} 帧，累计实例 {n_inst_frames}')

    print(f'\n地面拟合失败 {bad_ground} 帧；共 {n_inst_frames} 个实例（n_pts >= {args.min_pts}）\n')

    def q(v, p):
        return float(np.percentile(v, p)) if len(v) else float('nan')

    hdr = (f"{'类别':<16}{'实例':>6}{'点数中位':>9}{'高度p50':>9}{'高度p95':>9}"
           f"{'竖直跨度':>9}{'长轴p50':>9}{'短轴p50':>9}{'距离p50':>9}")
    print(hdr)
    print('-' * len(hdr))
    out = {}
    for cls in TARGET:
        rows = per_cls[cls]
        if not rows:
            print(f'{NAMES[cls]:<16}{0:>6}   —— 无实例')
            out[NAMES[cls]] = dict(n=0)
            continue
        h = np.array([r['height'] for r in rows])
        hm = np.array([r['height_mx'] for r in rows])
        ve = np.array([r['vert_ext'] for r in rows])
        fl = np.array([r['foot_L'] for r in rows])
        fw = np.array([r['foot_W'] for r in rows])
        np_ = np.array([r['n_pts'] for r in rows])
        ds = np.array([r['dist'] for r in rows])
        print(f'{NAMES[cls]:<16}{len(rows):>6}{int(np.median(np_)):>9}'
              f'{q(h, 50):>9.2f}{q(h, 95):>9.2f}{q(ve, 50):>9.2f}'
              f'{q(fl, 50):>9.2f}{q(fw, 50):>9.2f}{q(ds, 50):>9.1f}')
        out[NAMES[cls]] = dict(
            n=len(rows),
            height_p10=q(h, 10), height_p50=q(h, 50), height_p90=q(h, 90),
            height_mx_p90=q(hm, 90),
            vert_ext_p50=q(ve, 50),
            foot_L_p50=q(fl, 50), foot_W_p50=q(fw, 50),
            n_pts_p50=int(np.median(np_)), dist_p50=q(ds, 50),
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n已写出：{args.out}')


if __name__ == '__main__':
    main()
