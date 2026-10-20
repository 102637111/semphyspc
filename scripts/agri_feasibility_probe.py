#!/usr/bin/env python3
"""农业场景可行性估算：几厘米级的杂草，在当前传感器配置下能拿到几个点？（纯 CPU）

动机：论文把「农业机器人忽略可破坏目标（杂草）走最优路线」写成核心应用之一，但
**仓库里没有任何农业数据**，nuScenes 也没有杂草类。在为一个拿不到数据的场景设计
网络之前，先回答一个能用现有数据定量回答的问题：

    在 16k 子采样的 32 线 LiDAR 下，一个 15 cm × 20 cm 的杂草大概有几个点？

做法（不依赖农业数据，只用 nuScenes 的真实采样几何）：

  ① 地面点密度 ρ(r)：可行驶面/人行道/地形这几类的点按距离分箱，除以该环的**实测
     覆盖面积**（按 5° 方位扇区是否有点估计覆盖率，自动处理自车遮挡与视场裁剪）。
  ② 从 ρ(r) **反解激光束的竖直角分辨率** Δθ_v。地面是掠射采样，相邻两束在地面上的
     落点间距 Δr = h_s·Δθ_v/sin²θ（θ 为该束的俯角，h_s 为传感器高度），而方位向
     弧长 = r·Δφ，于是
                        Δθ_v = sin²θ / (ρ(r) · r · Δφ · h_s)
     这是一个**从数据里量出来的**束间距，不查数据手册；再与 HDL-32E 手册值
     （32 线 / 1.33°）对照，作为自检。
  ③ 有了竖直/水平角分辨率，任意距离 r 处、正面面积 A 的目标的预期点数就是
                        N = A / (r² · Δθ_v · Δφ)
     （正对传感器时入射角 ≈ 0）。用真实实例 + **标称真实尺寸**验证这个模型。
  ④ 外推到杂草，并额外给出**结构性漏检距离**：竖直束间距 r·Δθ_v 一旦超过目标高度，
     整株草就会落在两条激光束之间 —— 一个点都没有。这不是精度问题，是采样问题。

为什么必须带第 ④ 步：小目标的失败模式不是「平均点数少」而是「整株漏检」，而漏检
对训练集和评测集是**同样致命**的，且它随距离是阶跃式的。

用法：
    python3 scripts/agri_feasibility_probe.py --n_frames 200 \
        --out results/agri_feasibility_val.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

# nuScenes-lidarseg 32 类的索引（与 eval_seg_full.py 一致）
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
GROUND = [24, 25, 26, 27]                  # driveable / flat.other / sidewalk / terrain

# 标称真实尺寸（宽 × 高，米）——用来验证模型。这些是**近似标称值**，不是逐实例真值，
# 所以验证表格里报的是量级一致性，不是精确复现。
NOMINAL = {
    2:  ('human.pedestrian.adult', 0.60, 1.75),
    9:  ('movable_object.barrier', 1.00, 1.00),
    12: ('movable_object.trafficcone', 0.40, 0.70),
    14: ('vehicle.bicycle', 0.60, 1.10),
    17: ('vehicle.car', 1.90, 1.50),
    21: ('vehicle.motorcycle', 0.80, 1.20),
}
H_SENSOR = 1.84                            # nuScenes 车顶 LiDAR 标称高度（米）
N_BEAMS = 32                               # HDL-32E

RANGE_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 60)]
N_SECTORS = 72                             # 5° 一个扇区，用来估覆盖面积


def per_frame(xyz, sem, inst):
    """返回 (地面密度样本, 逐实例记录)。"""
    r = np.linalg.norm(xyz[:, :2], axis=1)             # 地面用 XY 距离
    az = np.arctan2(xyz[:, 1], xyz[:, 0])

    gmask = np.isin(sem, GROUND)
    dens = []
    for lo, hi in RANGE_BINS:
        m = gmask & (r >= lo) & (r < hi)
        n = int(m.sum())
        if n < 50:
            continue
        # 实测覆盖：该环里有多少个 5° 扇区至少落了一个点（自动含自车遮挡/视场裁剪）
        sec = np.unique(((az[m] + np.pi) / (2 * np.pi) * N_SECTORS).astype(np.int32) % N_SECTORS)
        cov = len(sec) / N_SECTORS
        area = np.pi * (hi ** 2 - lo ** 2) * cov
        if area <= 0:
            continue
        dens.append({'lo': lo, 'hi': hi, 'n': n, 'coverage': round(cov, 3),
                     'area_m2': round(area, 1), 'density': n / area,
                     'r_mid': (lo + hi) / 2})

    # 逐实例
    recs = []
    valid = inst >= 0
    if valid.any():
        key = sem.astype(np.int64) * 100000 + np.clip(inst, 0, 99999)
        uk, inv = np.unique(key[valid], return_inverse=True)
        idx_all = np.flatnonzero(valid)
        order = np.argsort(inv, kind='stable')
        starts = np.searchsorted(inv[order], np.arange(len(uk)))
        ends = np.append(starts[1:], len(order))
        for a, b, k in zip(starts, ends, uk):
            sel = idx_all[order[a:b]]
            if len(sel) < 1:
                continue
            c = int(k // 100000)
            p = xyz[sel]
            ctr = p.mean(axis=0)
            d = float(np.linalg.norm(ctr[:2]))
            if d < 1e-3:
                continue
            # 朝向传感器的水平跨度（垂直于视线方向）与竖直跨度 —— 即实测可见正面尺寸
            u = np.array([ctr[0], ctr[1], 0.0]) / d
            right = np.array([-u[1], u[0], 0.0])
            pr = p[:, :2] @ right[:2]
            pu = p[:, 2]
            recs.append({'cls': c, 'n_pts': int(len(sel)), 'range': d,
                         'w_obs': float(pr.max() - pr.min()),
                         'h_obs': float(pu.max() - pu.min())})
    return dens, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/nuscenes_seg_full_16k')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n_frames', type=int, default=200)
    ap.add_argument('--out', default='results/agri_feasibility_val.json')
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.data, args.split, '*.npz')))[:args.n_frames]
    if not frames:
        raise SystemExit(f'没有找到 npz：{args.data}/{args.split}')
    print(f'[agri] nuScenes {args.split} 取 {len(frames)} 帧（16k 子采样 = 训练实际用的密度）',
          flush=True)

    all_dens, all_recs, all_n = [], [], []
    for i, p in enumerate(frames):
        d = np.load(p)
        xyz = d['xyz'].astype(np.float64)
        dens, recs = per_frame(xyz, d['sem'].astype(np.int64), d['inst'].astype(np.int64))
        all_dens += dens
        all_recs += recs
        all_n.append(len(xyz))
        if (i + 1) % 50 == 0:
            print(f'  已处理 {i + 1}/{len(frames)} 帧', flush=True)

    n_pts_frame = float(np.median(all_n))

    # ---------- ① 地面密度 ----------
    print('\n' + '=' * 90)
    print('① 地面点密度 ρ(r)：16k 子采样后，地面上每平方米还能落几个点')
    print('=' * 90)
    print(f"{'距离环(m)':<14}{'地面点数':>12}{'覆盖扇区':>10}{'环面积(m²)':>12}{'ρ (点/m²)':>12}")
    print('-' * 90)
    dens_tab = []
    for lo, hi in RANGE_BINS:
        sub = [d for d in all_dens if d['lo'] == lo]
        if not sub:
            continue
        n = sum(d['n'] for d in sub)
        area = sum(d['area_m2'] for d in sub)
        cov = float(np.mean([d['coverage'] for d in sub]))
        rho = n / area
        dens_tab.append({'lo': lo, 'hi': hi, 'r_mid': (lo + hi) / 2,
                         'density': rho, 'coverage': cov})
        print(f'{f"{lo}–{hi}":<14}{n:>12}{cov * 100:>9.0f}%{area:>12.0f}{rho:>12.2f}')

    # ---------- ② 从地面密度反解束间距 ----------
    # 水平方向：方位向每个环上的采样数 ≈ 总点数 / 环数，故 Δφ = 2π/(N/N_beams)。
    # 这是**子采样后**的等效步长，不是传感器原始步长 —— 训练用的就是这个密度。
    dph = 2 * np.pi / max(n_pts_frame / N_BEAMS, 1.0)
    print('\n' + '=' * 90)
    print('② 从地面密度反解激光束的竖直角分辨率 Δθ_v（不查数据手册）')
    print('=' * 90)
    print('  地面上相邻两束的落点间距 Δr = h_s·Δθ_v/sin²θ，方位弧长 = r·Δφ，'
          '故 Δθ_v = sin²θ/(ρ·r·Δφ·h_s)')
    print(f'  等效方位步长 Δφ = 2π/({n_pts_frame:.0f}/{N_BEAMS}) = '
          f'{np.degrees(dph):.3f}° = {dph:.5f} rad')
    print(f"\n{'距离环':<12}{'ρ(点/m²)':>12}{'俯角θ':>10}{'反解 Δθ_v':>14}{'与手册 1.33°':>16}")
    print('-' * 90)
    dth_est = []
    for d in dens_tab:
        r = d['r_mid']
        if r < H_SENSOR * 1.5:
            continue                                   # 太近，掠射近似不成立
        sin_th = H_SENSOR / np.sqrt(H_SENSOR ** 2 + r ** 2)
        th = np.degrees(np.arctan2(H_SENSOR, r))
        dth = sin_th ** 2 / (d['density'] * r * dph * H_SENSOR)
        deg = np.degrees(dth)
        dth_est.append(dth)
        bin_name = f"{d['lo']}–{d['hi']}"
        print(f'{bin_name:<12}{d["density"]:>12.2f}{th:>9.1f}°'
              f'{deg:>13.3f}°{deg / 1.33:>15.2f}×')
    dth_v = float(np.median(dth_est)) if dth_est else np.radians(1.33)
    print(f'\n  取中位数 Δθ_v = {np.degrees(dth_v):.3f}° = {dth_v:.5f} rad'
          f'（HDL-32E 手册值 1.33°；反解值随距离有系统性漂移，'
          f'来自地面非严格水平与 h_s 近似，量级一致即可）')
    cell = dth_v * dph
    print(f'  ⇒ 角单元面积 Δθ_v·Δφ = {cell:.3e} sr')

    # ---------- ③ 模型验证 ----------
    def geo_pts(w, h, r):
        """正对传感器的目标在距离 r 处的预期点数。"""
        return (w * h) / (r ** 2 * cell)

    print('\n' + '=' * 90)
    print('③ 模型验证：N = A_frontal/(r²·Δθ_v·Δφ)，用**标称真实尺寸**（不是实测跨度）')
    print('=' * 90)
    print(f"{'类别':<32}{'实例数':>7}{'中位距离':>9}{'标称宽×高':>13}"
          f"{'实测点':>8}{'模型点':>9}{'比值':>8}")
    print('-' * 90)
    cal = []
    for c, (name, w, h) in NOMINAL.items():
        rs = [r for r in all_recs if r['cls'] == c]
        if len(rs) < 5:
            continue
        rng = float(np.median([r['range'] for r in rs]))
        meas = float(np.median([r['n_pts'] for r in rs]))
        pred = geo_pts(w, h, rng)
        cal.append({'cls': name, 'n_inst': len(rs), 'median_range': round(rng, 2),
                    'w': w, 'h': h, 'median_pts_measured': meas,
                    'pts_predicted': round(pred, 2),
                    'ratio_pred_over_meas': round(pred / max(meas, 1e-9), 3)})
        print(f'{name:<32}{len(rs):>7}{rng:>8.1f}m{f"{w:.2f}×{h:.2f}":>13}'
              f'{meas:>8.0f}{pred:>9.2f}{pred / max(meas, 1e-9):>8.2f}×')
    print('\n  比值在 1 附近 = 模型成立。用实测跨度会系统性高估（远处目标只被少量点'
          '命中，实测跨度必然小于真实尺寸），这正是先用实测跨度标定时系数虚高的原因。')
    # 模型系统性高估的原因是可解释的：目标并非正对传感器、背面被遮挡、部分点被
    # 邻域/自车剔除。既然方向明确且稳定，就把它显式折进外推，而不是留着当乐观偏差。
    ratios = [c['ratio_pred_over_meas'] for c in cal if c['ratio_pred_over_meas'] > 0]
    bias = float(np.median(ratios)) if ratios else 1.0
    print(f'  模型在真实实例上的中位高估倍数 = {bias:.2f}×；'
          f'下面的外推同时给出**已折入该修正**的一列，避免乐观。')

    # ---------- ④ 外推到杂草 ----------
    # 两个模型的失效方向相反，所以必须并排给、让它们夹住答案：
    #   物理模型 A：形式正确，但反解的 Δθ_v 偏小（0.86° vs 手册 1.33°），故偏乐观。
    #               用实测比值 bias 折回，再用手册值再算一遍，给出乐观/保守两档。
    #   经验幂律 B：在**大目标**上拟合得很好（R² 高），但小目标区间的点数被地板在
    #               1–3 点，拟合看不见「小于 1 点」这件事，外推到杂草必然偏乐观。
    X, Y = [], []
    for rec in all_recs:
        if rec['n_pts'] < 1 or rec['range'] <= 0:
            continue
        w_, h_ = rec['w_obs'], rec['h_obs']
        if w_ * h_ <= 1e-4:
            continue
        X.append([1.0, np.log(w_ * h_), np.log(rec['range'])])
        Y.append(np.log(rec['n_pts']))
    fit = None
    if len(X) > 200:
        X, Y = np.array(X), np.array(Y)
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        pred = X @ coef
        r2 = 1 - float(((Y - pred) ** 2).sum()) / max(float(((Y - Y.mean()) ** 2).sum()), 1e-9)
        fit = {'coef': [round(float(c), 4) for c in coef], 'r2': round(r2, 4), 'n': len(Y)}

    dth_spec = np.radians(1.33)                    # HDL-32E 手册值，作为保守档
    cell_spec = dth_spec * dph

    def geo_pts_spec(w, h, r):
        return (w * h) / (r ** 2 * cell_spec)

    print('\n' + '=' * 90)
    print('④ 外推：杂草每实例预期点数（三档并排，用它们夹住答案）')
    print('=' * 90)
    WEEDS = [('矮草 0.10×0.10 m', 0.10, 0.10), ('典型杂草 0.15×0.20 m', 0.15, 0.20),
             ('大杂草丛 0.30×0.30 m', 0.30, 0.30), ('参照：锥桶 0.40×0.70 m', 0.40, 0.70),
             ('参照：行人 0.60×1.75 m', 0.60, 1.75)]
    RS = [5, 10, 15, 20, 30, 40]
    hdr = ''.join(f'{f"{r} m":>9}' for r in RS)
    weed_tab = {}
    for tag, fn, note in (
            (f'A1 反解Δθ_v 折回{bias:.2f}×', lambda w, h, r: geo_pts(w, h, r) / bias,
             '最贴近实测'),
            ('A2 手册Δθ_v=1.33°（保守）', geo_pts_spec, '偏保守'),
            ('B  经验幂律（偏乐观）', None, '小目标区被地板效应污染')):
        print(f'\n--- {tag} --- {note}')
        print(f"{'目标':<24}" + hdr)
        print('-' * 90)
        for name, w, h in WEEDS:
            if fn is not None:
                row = [fn(w, h, r) for r in RS]
            else:
                c = fit['coef'] if fit else None
                row = ([float(np.exp(c[0] + c[1] * np.log(w * h) + c[2] * np.log(r)))
                        for r in RS] if c else [float('nan')] * len(RS))
            weed_tab.setdefault(name, {'w': w, 'h': h, 'frontal_m2': round(w * h, 4)})[
                tag.split()[0]] = {f'{r}m': round(v, 2) for r, v in zip(RS, row)}
            print(f'{name:<24}' + ''.join(f'{v:>9.1f}' for v in row))
    if fit:
        print(f'\n  模型 B 拟合：log N = {fit["coef"][0]:+.2f} '
              f'{fit["coef"][1]:+.3f}·log(面积) {fit["coef"][2]:+.3f}·log(距离)，'
              f'R² = {fit["r2"]:.3f}（{fit["n"]:,} 实例）')
        print('  注意它的指数远离理论值 (1, −2)——这不是理论错了，而是**地板效应**：'
              '远处小目标\n  的点数被卡在 1–3 点，拟合看到的动态范围被压缩，'
              '于是外推到杂草会严重偏乐观。')
    print(f'\n  Δθ_v 两档相差 {1.33 / np.degrees(dth_v):.2f}×；真实答案应落在 A2（保守）'
          f'与 B（乐观）之间。')

    # ---------- ⑤ 结构性漏检 ----------
    print('\n' + '=' * 90)
    print('⑤ 决定性效应：竖直束间距超过目标高度 ⇒ 整株草落在两束之间 ⇒ 0 点')
    print('=' * 90)
    print('  竖直方向两束的落点间距 = r·Δθ_v。目标高度 h < r·Δθ_v 时，'
          '能否被命中变成碰运气。')
    print(f"\n{'目标高度':<22}{'h/Δθ_v (临界距离)':>24}{'含义':>26}")
    print('-' * 90)
    miss_tab = {}
    for name, w, h in WEEDS:
        r_miss = h / dth_v
        miss_tab[name] = round(float(r_miss), 2)
        print(f'{name.split(" ")[0]:<22}{r_miss:>20.1f} m'
              f'{f"超过此距离基本漏检":>26}')
    print(f'\n  对照：0.2 m 高的杂草在 **{0.2 / dth_v:.1f} m** 以外就开始结构性漏检；'
          f'0.1 m 矮草是 {0.1 / dth_v:.1f} m。')
    print(f'  而 16k 采样下，20 m 处一个普通小目标（0.4×0.7 m）也只剩 '
          f'{geo_pts(0.4, 0.7, 20) / bias:.1f} 点（已修正）。')

    print('\n判据：杂草每实例 <5 点 ⇒ 当前传感器配置下不可用（既训不了也测不了）')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res = {'n_frames': len(frames), 'range_bins': dens_tab,
           'beam': {'d_theta_v_rad': dth_v, 'd_theta_v_deg': float(np.degrees(dth_v)),
                    'd_phi_rad': dph, 'd_phi_deg': float(np.degrees(dph)),
                    'angular_cell_sr': cell, 'median_pts_per_frame': n_pts_frame,
                    'n_beams': N_BEAMS, 'h_sensor_m': H_SENSOR},
           'model_validation': cal, 'model_bias_factor': bias,
           'd_theta_v_spec_deg': 1.33, 'empirical_fit': fit,
           'weeds': weed_tab, 'structural_miss_range_m': miss_tab}
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f'已写出：{args.out}')


if __name__ == '__main__':
    main()
