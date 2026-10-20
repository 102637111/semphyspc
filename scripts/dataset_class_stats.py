#!/usr/bin/env python3
"""统计每个语义类在数据集中的分布，用来把「罕见类失效」的三种可能成因分开：

  (a) 采样密度 —— 点数是否随分辨率提升而改善（16k → 34.7k）
  (b) 标注稀缺 —— 实例数 / 出现帧数是否本来就极少
  (c) 场景罕见 —— 出现帧数占比是否本来就很低

对每个分辨率目录遍历指定 split，累计每类：
  总点数 / 出现帧数(占比) / 实例数 / 每实例点数(中位、均值、p10、p90) / 出现帧内每帧点数(中位)

实例键取 (sem, inst) 对：先诊断 inst 是否在帧内跨类复用（若是，则必须带 sem 一起做键）。

用法:
    python3 scripts/dataset_class_stats.py --dir data/nuscenes_seg_full --split val --stride 2 \
        --out_json results/class_stats_4096_val.json
"""
import os
import sys
import glob
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LIDARSEG_32 = [
    'noise', 'animal',
    'human.pedestrian.adult', 'human.pedestrian.child',
    'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer', 'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.barrier', 'movable_object.debris',
    'movable_object.pushable_pullable', 'movable_object.trafficcone',
    'static_object.bicycle_rack', 'vehicle.bicycle',
    'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car',
    'vehicle.construction', 'vehicle.emergency.ambulance',
    'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer',
    'vehicle.truck', 'flat.driveable_surface', 'flat.other',
    'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other',
    'static.vegetation', 'vehicle.ego',
]
NC = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='如 data/nuscenes_seg_full')
    ap.add_argument('--split', default='val')
    ap.add_argument('--stride', type=int, default=2, help='每 N 帧取 1 帧')
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.split, '*.npz')))[::args.stride]
    if not files:
        raise SystemExit(f'没有 npz：{args.dir}/{args.split}')

    pts = np.zeros(NC, dtype=np.int64)          # 每类总点数
    frames = np.zeros(NC, dtype=np.int64)       # 出现该类的帧数
    inst_cnt = np.zeros(NC, dtype=np.int64)     # 实例数 (sem,inst) 唯一对
    inst_pts = [[] for _ in range(NC)]          # 每实例点数
    frame_pts = [[] for _ in range(NC)]         # 出现帧内每帧点数
    n_pts_total = 0
    n_dup_inst_cross_sem = 0                    # inst 在帧内跨类复用的次数（诊断用）

    for fi, fp in enumerate(files):
        d = np.load(fp)
        sem = d['sem'].astype(np.int64)
        inst = d['inst'].astype(np.int64)
        n_pts_total += len(sem)
        sem = np.where(sem < 0, 0, np.clip(sem, 0, NC - 1))

        # 诊断：同一 inst 值是否出现在多个 sem 下
        for v in np.unique(inst):
            if v < 0:
                continue
            if len(np.unique(sem[inst == v])) > 1:
                n_dup_inst_cross_sem += 1

        key = sem * 1000 + np.clip(inst, -1, 999)
        for c in np.unique(sem):
            m = sem == c
            k = key[m]
            uk = np.unique(k)
            # inst < 0（无实例标注）的整块算作一个伪实例，不计入实例统计
            n_inst = int((uk % 1000 >= 0).sum())
            n_real = int((uk % 1000 >= 0).sum())
            pts[c] += int(m.sum())
            frames[c] += 1
            frame_pts[c].append(int(m.sum()))
            for kk in uk:
                cnt = int((k == kk).sum())
                inst_pts[c].append(cnt)
            inst_cnt[c] += n_real
        if (fi + 1) % 500 == 0:
            print(f'  ... {fi + 1}/{len(files)}', flush=True)

    rows = []
    for c in range(NC):
        ip = np.array(inst_pts[c], dtype=np.float64)
        fp_ = np.array(frame_pts[c], dtype=np.float64)
        rows.append(dict(
            cls=c, name=LIDARSEG_32[c], points=int(pts[c]),
            frames=int(frames[c]), frames_frac=float(frames[c] / len(files)),
            instances=int(inst_cnt[c]),
            inst_points_median=float(np.median(ip)) if len(ip) else 0.0,
            inst_points_mean=float(ip.mean()) if len(ip) else 0.0,
            inst_points_p10=float(np.percentile(ip, 10)) if len(ip) else 0.0,
            inst_points_p90=float(np.percentile(ip, 90)) if len(ip) else 0.0,
            frame_points_median=float(np.median(fp_)) if len(fp_) else 0.0,
        ))

    print()
    print('=' * 128)
    print(f'{args.dir} / {args.split}  ({len(files)} 帧抽样, stride={args.stride}, '
          f'总点数 {n_pts_total}, 平均 {n_pts_total / len(files):.0f} 点/帧)')
    print(f'inst 在帧内跨类复用次数：{n_dup_inst_cross_sem}')
    print('-' * 128)
    print(f"{'id':>3} {'类别':<42}{'点数':>12}{'点数占比':>10}{'出现帧':>8}{'帧占比':>9}"
          f"{'实例数':>8}{'点/实例':>9}{'实例p10':>9}{'实例p90':>9}{'点/帧':>8}")
    print('-' * 128)
    for r in rows:
        print(f"{r['cls']:>3} {r['name']:<42}{r['points']:>12}{r['points'] / max(n_pts_total, 1):>10.5f}"
              f"{r['frames']:>8}{r['frames_frac']:>9.3f}{r['instances']:>8}"
              f"{r['inst_points_median']:>9.0f}{r['inst_points_p10']:>9.0f}"
              f"{r['inst_points_p90']:>9.0f}{r['frame_points_median']:>8.0f}")
    print('=' * 128)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(dict(dir=args.dir, split=args.split, stride=args.stride,
                           n_frames=len(files), n_points=n_pts_total,
                           inst_cross_sem=n_dup_inst_cross_sem, classes=rows),
                      f, ensure_ascii=False, indent=2)
        print(f'已写出：{args.out_json}')


if __name__ == '__main__':
    main()
