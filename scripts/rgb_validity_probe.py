#!/usr/bin/env python3
"""诊断数据集里的 RGB 通道到底有没有用、以及 RGB 与坐标是否配准（纯 CPU）。

动机：论文把「RGB+XYZ 早期融合」当作分割前置的核心设计（§3.2、摘要、Semantic3D 迁移），
但**从来没有做过有 RGB / 无 RGB 的对照**。审稿人一定会问：颜色通道真的被用上了吗？
更基本的：6 相机投影上色这一步有没有出错？

三个模型无关的检验，全部只读 npz，不需要 GPU、不需要定标参数：

1. 覆盖率——rgb 恰好是 (0,0,0) 的点（= 6 个相机都看不到、被填零）占多少，按语义类分解。
   填零点会把「无颜色」编码成纯黑，是一个可被模型学到的伪线索，而它恰恰集中在远处/遮挡目标。

2. 配准（决定性检验）——**类内颜色一致性**。车、公交、卡车、交通锥都是单色物体：
   若投影正确，同一实例的点应共享同一个颜色（实例内 RGB 标准差小），而不同实例
   （不同车漆）之间的均值颜色应明显散开。配准一旦错位，颜色就从别处采样，
   实例内散度会抬升、实例间区分度会塌掉。用「实例内 std」与「实例间 mean 的 std」
   之比来量化，比值越小说明点与像素对应得越准。

3. 颜色信息量——最近类均值分类器按 RGB 三通道预测语义，看精度是否显著高于多数类基线。
   远高于基线 ⇒ 颜色携带真实判别信息；接近基线 ⇒ 通道无效（全零/常数/纯噪声）。

用法：
    python3 scripts/rgb_validity_probe.py --split val --n_frames 300 \
        --out results/rgb_validity_val.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

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

# 「单色物体」——同一实例在现实中几乎是同一个颜色（车漆/塑料），因此适合做配准检验。
# 建筑与植被不入选：它们的颜色本来就杂（窗户/砖缝/枝叶），实例内散度天然大。
MONO = [9, 12, 14, 15, 16, 17, 21, 23]      # barrier, trafficcone, bicycle, bus.bendy/rigid, car, motorcycle, truck
MIN_PTS = 50                                 # 实例至少这么多点才统计（少了 std 不可靠）
MIN_INST = 8                                 # 同类至少这么多实例才给出该类比值


def probe(frames, n_frames):
    """返回覆盖率、类内/类间颜色一致性、以及逐点的 (rgb, sem) 供分类器用。"""
    cover_tot = np.zeros(len(LIDARSEG_32), dtype=np.int64)     # 该类总点数
    cover_blk = np.zeros(len(LIDARSEG_32), dtype=np.int64)     # 其中被填零的点数
    # 实例内 std / 实例间均值 std：按类累加
    within = {c: [] for c in MONO}
    between = {c: [] for c in MONO}
    feats, labels = [], []
    used = 0

    for p in frames[:n_frames]:
        d = np.load(p)
        xyz, rgb, sem, inst = d['xyz'], d['rgb'], d['sem'], d['inst']
        zeros = ~rgb.any(axis=1)                                # 三通道全 0 = 无相机覆盖
        for c in range(len(LIDARSEG_32)):
            m = sem == c
            cover_tot[c] += int(m.sum())
            cover_blk[c] += int((m & zeros).sum())

        # 类内一致性：只看有真实颜色的点，否则填零会把 std 抬起来（那是覆盖问题，不是配准问题）
        for c in MONO:
            means = []
            for i in np.unique(inst[sem == c]):
                if i < 0:
                    continue
                m = (sem == c) & (inst == i) & ~zeros
                if m.sum() < MIN_PTS:
                    continue
                sub = rgb[m]
                within[c].append(float(sub.std(axis=0).mean()))
                means.append(sub.mean(axis=0))
            if len(means) >= MIN_INST:
                between[c].append(float(np.asarray(means).std(axis=0).mean()))

        # 分类器样本：每类最多留 2000 点，避免地面类把统计压死
        for c in range(1, len(LIDARSEG_32)):
            m = (sem == c) & ~zeros
            idx = np.flatnonzero(m)
            if idx.size == 0:
                continue
            idx = idx[:2000]
            feats.append(rgb[idx])
            labels.append(np.full(idx.size, c, dtype=np.int16))
        used += 1
        if used % 50 == 0:
            print(f'  已处理 {used} 帧', flush=True)

    res = {'n_frames': used, 'cover_total': cover_tot, 'cover_blank': cover_blk,
           'mono_classes': MONO,
           'within_std': {c: within[c] for c in MONO},
           'between_std': {c: between[c] for c in MONO}}
    if feats:
        res['feat_rgb'] = np.concatenate(feats)
        res['feat_sem'] = np.concatenate(labels)
    return res


def nearest_class(feats, labels, rng):
    """最近类均值分类器：留出 30% 评估（模型无关、无需训练）。"""
    n = len(labels)
    perm = rng.permutation(n)
    n_tr = int(0.7 * n)
    tr, te = perm[:n_tr], perm[n_tr:]
    classes = np.unique(labels)
    cent = np.stack([feats[tr][labels[tr] == c].mean(axis=0) for c in classes])
    d = ((feats[te][:, None, :] - cent[None]) ** 2).sum(-1)
    pred = classes[d.argmin(1)]
    acc = float((pred == labels[te]).mean())
    # 基线：把测试集全判为训练集里最多的类
    vals, cnts = np.unique(labels[tr], return_counts=True)
    base = float((labels[te] == vals[cnts.argmax()]).mean())
    # 逐类召回（只报样本量足够的类，否则数不稳）
    per = {}
    for c in classes:
        m = labels[te] == c
        if m.sum() >= 100:
            per[LIDARSEG_32[int(c)]] = round(float((pred[m] == c).mean()), 4)
    return acc, base, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/nuscenes_seg_full_16k')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n_frames', type=int, default=300)
    ap.add_argument('--out', default='results/rgb_validity_val.json')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.data, args.split, '*.npz')))
    if not frames:
        raise SystemExit(f'没有找到 npz：{args.data}/{args.split}')
    print(f'[rgb] {args.split} 共 {len(frames)} 帧，取前 {args.n_frames} 帧，纯 CPU', flush=True)

    r = probe(frames, args.n_frames)
    tot, blk = r['cover_total'], r['cover_blank']

    print('\n' + '=' * 74)
    print('检验 1：6 相机颜色覆盖率（rgb 全 0 = 无相机覆盖、被填零）')
    print('=' * 74)
    print(f"{'类别':<38}{'点数':>12}{'无颜色占比':>12}")
    print('-' * 74)
    order = np.argsort(-tot)
    for c in order[:16]:
        if tot[c] == 0:
            continue
        print(f'{LIDARSEG_32[c]:<38}{tot[c]:>12d}{blk[c] / tot[c] * 100:>11.1f}%')
    print(f"{'—— 全部 32 类 ——':<38}{tot.sum():>12d}{blk.sum() / max(tot.sum(), 1) * 100:>11.1f}%")

    print('\n' + '=' * 74)
    print('检验 2：配准（单色物体的类内/类间颜色一致性）')
    print('  比值 = 实例内 RGB std 的均值 / 同类各实例均值颜色之间的 std')
    print('  配准正确 ⇒ 同一实例共享一个颜色，比值应远小于 1；错位 ⇒ 比值趋近 1')
    print('=' * 74)
    print(f"{'类别':<26}{'实例数':>8}{'实例内 std':>12}{'实例间 std':>12}{'比值':>9}")
    print('-' * 74)
    mono = {}
    for c in r['mono_classes']:
        w, b = r['within_std'][c], r['between_std'][c]
        if not w or not b:
            print(f'{LIDARSEG_32[c]:<26}{len(w):>8}{"—":>12}{"—":>12}{"—":>9}')
            continue
        wm, bm = float(np.mean(w)), float(np.mean(b))
        mono[LIDARSEG_32[c]] = {'n_inst': len(w), 'within': round(wm, 4),
                                'between': round(bm, 4), 'ratio': round(wm / bm, 4)}
        print(f'{LIDARSEG_32[c]:<26}{len(w):>8}{wm:>12.4f}{bm:>12.4f}{wm / bm:>9.3f}')

    acc = base = None
    per = {}
    if 'feat_rgb' in r:
        acc, base, per = nearest_class(r['feat_rgb'], r['feat_sem'], np.random.default_rng(args.seed))
        print('\n' + '=' * 74)
        print('检验 3：颜色信息量（最近类均值分类器，RGB 3 通道 → 32 类，70/30 留出）')
        print('=' * 74)
        print(f'  留出精度 {acc:.4f}　多数类基线 {base:.4f}　增益 {acc - base:+.4f}')
        print('  逐类召回（测试样本 ≥100 的类）：')
        for k, v in sorted(per.items(), key=lambda kv: -kv[1]):
            print(f'    {k:<38}{v:.3f}')

    out = {
        'n_frames': r['n_frames'],
        'blank_ratio_overall': float(blk.sum() / max(tot.sum(), 1)),
        'per_class': {LIDARSEG_32[c]: {'pts': int(tot[c]), 'blank': int(blk[c]),
                                       'blank_ratio': float(blk[c] / tot[c]) if tot[c] else None}
                      for c in range(len(LIDARSEG_32)) if tot[c] > 0},
        'mono_consistency': mono,
        'rgb_classifier': {'holdout_acc': acc, 'majority_baseline': base,
                           'per_class_recall': per} if acc is not None else None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n已写出：{args.out}')


if __name__ == '__main__':
    main()
