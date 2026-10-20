#!/usr/bin/env python3
"""统计每个类「每实例点数」的完整分布，并给出**可达性上界**：

一个实例如果有 k 个点，任何逐点分割器最多只能把这 k 个点判对。因此
「点数 < K 的实例占比」直接给出了该类在给定采样密度下**不可能被分割**的部分，
与模型、训练策略无关。用来回答「罕见类失效是不是采样问题」。

输出每类：
  - 实例数，及 ≥1/≥3/≥5/≥10/≥20/≥50 点的实例数（累积）
  - 该类 GT 点中落在「≥K 点实例」上的比例（点加权，这才是逐点 IoU 的口径）
  - 最大实例点数

用法:
    python3 scripts/instance_size_hist.py --dir data/nuscenes_seg_full_64k --split val \
        --stride 2 --out_json results/inst_size_64k_val.json
"""
import os
import sys
import glob
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NC = 32
THRESH = [1, 3, 5, 10, 20, 50]
NAMES = ['noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child',
         'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility',
         'human.pedestrian.police_officer', 'human.pedestrian.stroller',
         'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
         'movable_object.pushable_pullable', 'movable_object.trafficcone',
         'static_object.bicycle_rack', 'vehicle.bicycle', 'vehicle.bus.bendy',
         'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction',
         'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.motorcycle',
         'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface', 'flat.other',
         'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other',
         'static.vegetation', 'vehicle.ego']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.split, '*.npz')))[::args.stride]
    if not files:
        raise SystemExit(f'没有 npz：{args.dir}/{args.split}')

    # 每类的实例点数列表（内存可控：罕见类实例数万级，stuff 类实例数少）
    inst_sizes = [[] for _ in range(NC)]
    class_pts = np.zeros(NC, dtype=np.int64)

    for fi, fp in enumerate(files):
        d = np.load(fp)
        sem = np.clip(d['sem'].astype(np.int64), 0, NC - 1)
        inst = d['inst'].astype(np.int64)
        # inst 在同一帧内会跨类复用（诊断见 dataset_class_stats.py），故用 (sem, inst) 做键
        key = sem * 1000 + np.clip(inst, 0, 999)
        uk, cnt = np.unique(key, return_counts=True)
        for k, c in zip(uk, cnt):
            cls = int(k // 1000)
            inst_sizes[cls].append(int(c))
            class_pts[cls] += int(c)
        if (fi + 1) % 1000 == 0:
            print(f'  ... {fi + 1}/{len(files)}', flush=True)

    rows = []
    for c in range(NC):
        s = np.array(inst_sizes[c], dtype=np.int64)
        if len(s) == 0:
            rows.append(dict(cls=c, name=NAMES[c], instances=0, points=int(class_pts[c]),
                             sizes={}, max_size=0))
            continue
        sizes = {f'>={t}': int((s >= t).sum()) for t in THRESH}
        # 点加权可达率：该类 GT 点里，落在 ≥t 点实例上的占比
        pw = {f'>={t}': float(s[s >= t].sum() / s.sum()) if s.sum() else 0.0 for t in THRESH}
        rows.append(dict(cls=c, name=NAMES[c], instances=int(len(s)),
                         points=int(class_pts[c]), sizes=sizes, point_weighted=pw,
                         max_size=int(s.max())))

    print()
    print('=' * 118)
    print(f'{args.dir} / {args.split}  ({len(files)} 帧)')
    print('-' * 118)
    print(f"{'id':>3} {'类别':<20}{'实例数':>8}" +
          ''.join(f"{'≥' + str(t):>8}" for t in THRESH) +
          f"{'最大':>8}{'点加权≥10':>11}{'点加权≥20':>11}")
    print('-' * 118)
    for r in rows:
        if r['instances'] == 0:
            continue
        print(f"{r['cls']:>3} {r['name']:<20}{r['instances']:>8}" +
              ''.join(f"{r['sizes'][f'>={t}']:>8}" for t in THRESH) +
              f"{r['max_size']:>8}{r['point_weighted']['>=10']:>11.3f}"
              f"{r['point_weighted']['>=20']:>11.3f}")
    print('=' * 118)
    print('（≥t 列 = 点数不少于 t 的实例个数；点加权≥t = 该类 GT 点中落在 ≥t 点实例上的比例，'
          '即逐点 IoU 的「可达」部分）')

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(dict(dir=args.dir, split=args.split, stride=args.stride,
                           n_frames=len(files), thresholds=THRESH, classes=rows),
                      f, ensure_ascii=False, indent=2)
        print(f'已写出：{args.out_json}')


if __name__ == '__main__':
    main()
