#!/usr/bin/env python3
"""安全等级的可分性探针：把 32 类语义粗化成 5 级碰撞安全等级后，可分性提升多少？
致命类（L0）能不能被保住？（纯 CPU、模型无关）

动机与立论
----------
§4.14 已经证明：罕见类失效的**主因是分组粒度**，不是采样密度也不是 backbone
（g64→g1024 罕见类 IoU 0.0676→0.1736 = 2.57×）。如果这个归因成立，那么把 32 类
粗化成 5 级安全等级的收益，应该**远大于任何 backbone 层面的改进**。

而且有一个更值钱的推论：现在的评测（罕见类逐类 IoU）**系统性地低估了系统的安全性**。
模型把 ped.child 预测成 ped.adult，对分割是错，对碰撞决策是**完全正确**的。
32 类里「行人→行人」的所有混淆，在安全决策里都是安全的。

本脚本用**模型无关**的方式把这个判断变成数字：只用几何特征（不含颜色），在
nuScenes 真实数据上做逐实例分类，比较三个靶子：

    靶子 A  32 类语义       —— 现状评测的对象
    靶子 B  5 级安全等级     —— 本脚本主张的预测目标
    靶子 C  二分类「致命 vs 可撞」 —— §3.6 select_soft_contact 真正要做的决策

判据（论文的核心指标，不是 mIoU）：
    L0 召回        致命类（动物+全部行人子类）被正确识别为致命的比例
    critical rate  L0 被判成 L2/L3（可破坏/可变形）——**会被主动撞上**，最致命
    severe rate    L0 被判成 L4（可通过）——会被当成路面

为什么**逐实例**而不是逐点：安全决策的对象是「撞不撞这个物体」，是实例级的；
而且逐点分类会让同一物体的点数优势泄漏进统计。切分也按**实例**做，避免泄漏。

为什么只用几何：颜色对语义的判别力实测只有 1.7× 随机（`rgb_validity_probe.py`），
所以几何是这个系统的主力模态；本探针测的就是主力的天花板。

用法：
    python3 scripts/safety_level_probe.py --n_frames 400 \
        --out results/safety_level_val.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# nuScenes-lidarseg 32 类
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

# 安全等级：与 eval_seg_full.py 的安全聚合严格同口径，避免两处漂移
LETHAL = [1, 2, 3, 4, 5, 6, 7, 8]         # 动物 + 全部行人子类
FRANGIBLE = [9, 10, 11, 12]               # 护栏/碎屑/可推动/锥桶
DEFORMABLE = [30]                         # 植被（27 地形 2026-09-14 移入 PASSABLE）
PASSABLE = [24, 25, 26, 27]               # 路面/其他平面/人行道/地形
EXCLUDE = [0, 31]                         # noise / vehicle.ego 不参与

LEVEL_NAMES = ['L0 致命', 'L1 可破坏', 'L2 可变形', 'L3 硬刚性', 'L4 可通过']
LEVEL_IDS = [0, 1, 2, 3, 4]

SEM2LEVEL = {}
for s in LETHAL:
    SEM2LEVEL[s] = 0
for s in FRANGIBLE:
    SEM2LEVEL[s] = 1
for s in DEFORMABLE:
    SEM2LEVEL[s] = 2
for s in PASSABLE:
    SEM2LEVEL[s] = 4
for s in range(32):
    if s not in SEM2LEVEL and s not in EXCLUDE:
        SEM2LEVEL[s] = 3                    # 其余（含全部车辆）= 硬刚性


def ground_z(xyz):
    """估计地面高度：取最低的 25% 点的 z 中位数。nuScenes 是 z 朝上的 ego 坐标系。"""
    return float(np.percentile(xyz[:, 2], 25))


def instance_feats(xyz, sem, inst, k_nn=16):
    """逐实例几何特征（平移不变 + 尺度 + 相对地面的姿态）。"""
    from scipy.spatial import cKDTree
    z0 = ground_z(xyz)
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=min(k_nn, len(xyz)))
    nb = xyz[idx]
    c = nb.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', nb - c, nb - c) / nb.shape[1]
    w, v = np.linalg.eigh(cov)
    l1, l2, l3 = w[:, 2], w[:, 1], w[:, 0]
    eps = 1e-12
    # 逐点局部形状量
    s_lin = l3 / (l1 + eps)                 # 散射度（高 = 非平面）
    s_pln = (l2 - l3) / (l1 + eps)          # 平面度
    s_vert = np.abs(v[:, 2, 2])             # 竖直度（法向 vs 重力）

    valid = inst >= 0
    recs = []
    if not valid.any():
        return recs
    key = sem.astype(np.int64) * 100000 + np.clip(inst, 0, 99999)
    uk, inv = np.unique(key[valid], return_inverse=True)
    idx_all = np.flatnonzero(valid)
    order = np.argsort(inv, kind='stable')
    starts = np.searchsorted(inv[order], np.arange(len(uk)))
    ends = np.append(starts[1:], len(order))
    for a, b, k in zip(starts, ends, uk):
        sel = idx_all[order[a:b]]
        s = int(k // 100000)
        if s in EXCLUDE or s not in SEM2LEVEL:
            continue
        p = xyz[sel]
        if len(sel) < 4:
            continue                                # 实例太小，几何量无意义
        ctr = p.mean(axis=0)
        d = float(np.linalg.norm(ctr[:2]))
        if d < 1e-3:
            continue
        # 水平主方向：对 XY 做 PCA，长轴/短轴跨度
        xy = p[:, :2] - ctr[:2]
        w2, v2 = np.linalg.eigh(xy.T @ xy / len(xy))
        ax = v2[:, -1]
        along = xy @ ax
        across = xy @ np.array([-ax[1], ax[0]])
        length = float(along.max() - along.min())
        width = float(across.max() - across.min())
        height = float(p[:, 2].max() - p[:, 2].min())
        zmin = float(p[:, 2].min()) - z0
        zmean = float(p[:, 2].mean() - z0)
        # 长宽高排序后给出与朝向无关的尺寸三元组（避免朝向泄漏）
        dims = sorted([length, width, height], reverse=True)
        rec = {
            'sem': s, 'level': SEM2LEVEL[s], 'n_pts': int(len(sel)), 'range': d,
            'Dim1': dims[0], 'Dim2': dims[1], 'Dim3': dims[2],
            'height': height, 'z_min': zmin, 'z_mean': zmean,
            'fill': len(sel) / max(dims[0] * dims[1] * dims[2], 1e-6),  # 体填充率
            's_scatter': float(s_lin[sel].mean()),
            's_planar': float(s_pln[sel].mean()),
            's_vertical': float(s_vert[sel].mean()),
            'log_n': float(np.log1p(len(sel))),
        }
        recs.append(rec)
    return recs


FEAT_KEYS = ['Dim1', 'Dim2', 'Dim3', 'height', 'z_min', 'z_mean', 'fill',
             's_scatter', 's_planar', 's_vertical', 'log_n', 'range']


def map_conf_to_levels(conf, classes):
    """把 32 类混淆矩阵按 SEM2LEVEL 聚合成 5 级混淆矩阵。

    这是**路径 1**的关键工具：先预测细粒度语义，再查固定表映射到安全等级。
    32→5 是确定性映射，本来就不需要「学」，所以真正该比较的不是两个分类器的
    平衡精度（粗目标的随机基线本来就高，不可比），而是两条路径最终的
    **致命误判率**。
    """
    L = sorted(set(SEM2LEVEL.values()))
    idx = {l: i for i, l in enumerate(L)}
    c5 = np.zeros((len(L), len(L)), np.int64)
    for i, tc in enumerate(classes):
        if tc not in SEM2LEVEL:
            continue
        for j, pc in enumerate(classes):
            if pc not in SEM2LEVEL:
                continue
            c5[idx[SEM2LEVEL[tc]], idx[SEM2LEVEL[pc]]] += conf[i, j]
    return c5, L


def safety_rates(c5, L):
    """从 5 级混淆矩阵算安全三率（实例级）。与 eval_seg_full.py 同口径。"""
    if 0 not in L:
        return None
    i0 = L.index(0)
    tot = int(c5[i0].sum())
    if tot == 0:
        return None
    coll = int(sum(c5[i0][L.index(c)] for c in (1, 2) if c in L))
    sev = int(c5[i0][L.index(4)]) if 4 in L else 0
    rec0 = c5[i0, i0] / tot
    return {'n_lethal': tot, 'lethal_recall': round(float(rec0), 4),
            'critical_rate': round(coll / tot, 6),
            'severe_rate': round(sev / tot, 6),
            'safe_rate': round((tot - coll - sev) / tot, 6)}


def eval_target(X, y, rng, clf='lda'):
    """按类切分的平衡精度 + 完整混淆矩阵。返回 (bal_acc, conf, classes)。"""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    cls = np.unique(y)
    # 按**类**分层切分，保证每个类在测试集里都有代表，否则平衡精度无法算
    tr, te = [], []
    for c in cls:
        m = np.flatnonzero(y == c)
        rng.shuffle(m)
        n_tr = max(1, int(0.7 * len(m)))
        tr.append(m[:n_tr]); te.append(m[n_tr:])
    tr, te = np.concatenate(tr), np.concatenate(te)
    if clf == 'ncm':
        cent, keep = [], []
        for c in cls:
            m = tr[y[tr] == c]
            if len(m) == 0:
                continue
            cent.append(X[m].mean(0)); keep.append(c)
        d = ((X[te][:, None, :] - np.stack(cent)[None]) ** 2).sum(-1)
        pred = np.asarray(keep)[d.argmin(1)]
    else:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        m = LinearDiscriminantAnalysis()
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        keep = cls
    yte = y[te]
    ci = {c: i for i, c in enumerate(keep)}
    conf = np.zeros((len(keep), len(keep)), np.int64)
    for t, p in zip(yte, pred):
        if t in ci and p in ci:
            conf[ci[t], ci[p]] += 1
    rec = conf.diagonal() / np.maximum(conf.sum(1), 1)
    return float(np.mean(rec)), conf, list(keep), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/nuscenes_seg_full_16k')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n_frames', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/safety_level_val.json')
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.data, args.split, '*.npz')))[:args.n_frames]
    if not frames:
        raise SystemExit(f'没有找到 npz：{args.data}/{args.split}')
    print(f'[safety] nuScenes {args.split} 取 {len(frames)} 帧，逐实例、仅几何特征，纯 CPU',
          flush=True)

    recs = []
    for i, p in enumerate(frames):
        d = np.load(p)
        recs += instance_feats(d['xyz'].astype(np.float64), d['sem'].astype(np.int64),
                               d['inst'].astype(np.int64))
        if (i + 1) % 100 == 0:
            print(f'  已处理 {i + 1}/{len(frames)} 帧，累计 {len(recs):,} 个实例', flush=True)

    X = np.stack([[r[k] for k in FEAT_KEYS] for r in recs])
    sem = np.array([r['sem'] for r in recs])
    lvl = np.array([r['level'] for r in recs])
    print(f'\n合计 {len(recs):,} 个实例')

    cnt32 = {LIDARSEG_32[s]: int((sem == s).sum()) for s in np.unique(sem)}
    print('逐类实例数：', {k: v for k, v in sorted(cnt32.items(), key=lambda x: -x[1])})
    print('逐级实例数：', {LEVEL_NAMES[l]: int((lvl == l).sum()) for l in np.unique(lvl)})

    rng = np.random.default_rng(args.seed)
    res = {'n_frames': len(frames), 'n_inst': int(len(recs)),
           'feat_keys': FEAT_KEYS,
           'level_counts': {LEVEL_NAMES[l]: int((lvl == l).sum()) for l in np.unique(lvl)},
           'sem_counts': cnt32}

    # ---------- 三个靶子 ----------
    print('\n' + '=' * 92)
    print('仅几何特征、逐实例分类：三个靶子的可分性对照（LDA，按类分层 70/30）')
    print('=' * 92)
    print(f"{'靶子':<26}{'类别数':>7}{'随机基线':>10}{'平衡精度':>10}{'相对随机':>10}")
    print('-' * 92)
    targets = [
        ('A  32 类语义（现状）', sem, 32),
        ('B  5 级安全等级', lvl, 5),
    ]
    out = {}
    for name, y, n_cls in targets:
        bal, conf, keep, rec = eval_target(X, y, rng, clf='lda')
        base = 1.0 / len(keep)
        out[name] = {'n_classes': len(keep), 'balanced_acc': round(bal, 4),
                     'chance': round(base, 4), 'ratio': round(bal / base, 3),
                     'confusion': conf.tolist(), 'classes': [int(c) for c in keep],
                     'recall': [round(float(r), 4) for r in rec]}
        print(f'{name:<26}{len(keep):>7}{base:>10.4f}{bal:>10.4f}{bal / base:>9.2f}×')

    # ---------- 决定性对照：两条路径的致命误判率 ----------
    # 路径 1：细粒度预测 32 类 → 查固定表映射到 5 级（现状架构）
    # 路径 2：直接预测 5 级（我原本主张的架构）
    # 平衡精度不可比（粗目标随机基线本来就高），能比的只有最终的致命误判率。
    conf32 = np.array(out['A  32 类语义（现状）']['confusion'])
    cls32 = out['A  32 类语义（现状）']['classes']
    c5_path1, L = map_conf_to_levels(conf32, cls32)
    r1 = safety_rates(c5_path1, L)
    c5_path2 = np.array(out['B  5 级安全等级']['confusion'])
    r2 = safety_rates(c5_path2, out['B  5 级安全等级']['classes'])

    print('\n' + '=' * 92)
    print('★ 决定性对照：细粒度+查表 vs 直接学粗等级（同一批几何特征，同一批实例）')
    print('=' * 92)
    print(f"{'路径':<34}{'致命实例数':>12}{'L0 召回':>10}"
          f"{'critical':>11}{'severe':>10}{'安全率':>10}")
    print('-' * 92)
    for nm, r in (('路径1  预测 32 类 → 查表', r1), ('路径2  直接预测 5 级', r2)):
        if r is None:
            continue
        print(f"{nm:<34}{r['n_lethal']:>12}{r['lethal_recall']:>10.3f}"
              f"{r['critical_rate'] * 100:>10.2f}%{r['severe_rate'] * 100:>9.2f}%"
              f"{r['safe_rate'] * 100:>9.2f}%")
    out['path1_sem32_then_lookup'] = r1
    out['path2_direct_5level'] = r2
    if r1 and r2:
        d = r1['critical_rate'] - r2['critical_rate']
        verdict = ('粗等级头**更差**，不要做' if d < -0.005 else
                   '粗等级头**更好**，值得做' if d > 0.005 else
                   '两条路径**基本等价**，粗等级头没有额外价值')
        print(f'\n  critical 率之差（路径1 − 路径2）= {d * 100:+.2f} 个百分点 ⇒ {verdict}')
        print('  注：balanced accuracy 不能用来比较这两条路径——5 级的随机基线是')
        print('  0.2，32 类是 0.031，粗目标的平衡精度天然更高，那是基线的差异不是')
        print('  能力的差异。只有最终的致命误判率是可比的。')

    # 二分类：致命 vs 可撞（§3.6 真正要做的决策）
    binary = np.full(len(lvl), -1, np.int64)
    binary[lvl == 0] = 0                                   # 致命
    binary[np.isin(lvl, [1, 2])] = 1                       # 可撞（可破坏/可变形）
    m = binary >= 0
    bal_b, conf_b, keep_b, rec_b = eval_target(X[m], binary[m], rng, clf='lda')
    out['C  致命 vs 可撞（二分类）'] = {
        'n_classes': len(keep_b), 'balanced_acc': round(bal_b, 4), 'chance': 0.5,
        'ratio': round(bal_b / 0.5, 3), 'confusion': conf_b.tolist(),
        'recall': [round(float(r), 4) for r in rec_b], 'n_inst': int(m.sum())}
    print(f"{'C  致命 vs 可撞（二分类）':<26}{len(keep_b):>7}{0.5:>10.4f}"
          f"{bal_b:>10.4f}{bal_b / 0.5:>9.2f}×")

    # ---------- 5 级混淆矩阵与安全三率 ----------
    print('\n' + '=' * 92)
    print('B 靶子的完整混淆矩阵（行=真实等级，列=预测等级，按行归一化 %）')
    print('=' * 92)
    conf = np.array(out['B  5 级安全等级']['confusion'])
    classes = out['B  5 级安全等级']['classes']
    names = [LEVEL_NAMES[c] for c in classes]
    corner = '真实\\预测'
    print(f'{corner:<14}' + ''.join(f'{n:>13}' for n in names) + f"{'召回':>9}")
    print('-' * 92)
    for i, n in enumerate(names):
        row = conf[i] / max(conf[i].sum(), 1) * 100
        print(f'{n:<14}' + ''.join(f'{v:>12.1f}%' for v in row)
              + f'{conf[i, i] / max(conf[i].sum(), 1):>9.3f}')

    # 安全三率：与 eval_seg_full.py 同口径，但这里是**实例级**（论文核心指标）
    if 0 in classes:
        i0 = classes.index(0)
        tot = max(conf[i0].sum(), 1)
        coll = sum(conf[i0][classes.index(c)] for c in (1, 2) if c in classes)
        sev = conf[i0][classes.index(4)] if 4 in classes else 0
        safe = tot - coll - sev
        print('\n' + '=' * 92)
        print('安全三率（实例级，仅几何特征、未训练）：致命类实例被判到哪一级')
        print('=' * 92)
        print(f'  致命类实例共 {tot:,} 个')
        print(f'  **判为可撞（L1/L2）—— critical rate = {coll / tot * 100:.2f}%**'
              f'  （会被主动撞上）')
        print(f'  **判为可通过（L4）  —— severe rate   = {sev / tot * 100:.2f}%**'
              f'  （会被当成路面）')
        print(f'  判为安全（L0 自身 或 L3 硬刚性）= {safe / tot * 100:.2f}%')
        out['safety_rates_geometry_only'] = {
            'n_lethal_inst': int(tot),
            'critical_rate': round(coll / tot, 6),
            'severe_rate': round(sev / tot, 6),
            'safe_rate': round(safe / tot, 6)}
        print('\n  这是**未训练**的几何天花板（LDA 直接用几何特征）。训练后的 5 级安全头')
        print('  应当高于此数；若连这个数都远好于 32 类评测暗示的水平，说明现状评测')
        print('  （逐类 IoU）系统性低估了系统安全性。')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res | out, f, indent=2, ensure_ascii=False)
    print(f'\n已写出：{args.out}')


if __name__ == '__main__':
    main()
