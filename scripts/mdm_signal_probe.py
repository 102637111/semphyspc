#!/usr/bin/env python3
"""检验 MDM（材质不连续映射）的**核心前提**是否成立：材质边界处的颜色突变，是否
比材质内部显著更强；以及**几何是否已经标记了同一条边界**（若已标记，MDM 就没有增量）。

这是纯 CPU、模型无关的「信号存在性」闸门，用来决定要不要为 MDM 投 GPU。
比先训练再看好坏便宜几个数量级，而且结论更干净。

为什么用 UR-MAT 当试验台：
  - 有**逐点材质真值**（14 类），可以直接定义「材质边界」这个标签，不需要任何代理指标；
  - RGB 与深度**像素对齐**（同一张渲染图），所以投影误差为零 —— 把「信号不存在」
    与「投影错位把信号冲掉了」这两个失败原因彻底分开。真实数据上的退化另算。

三个量（对每个点，在半径 r 的 3D 邻域内计算）：
  1. 是否材质边界：邻域内是否存在不同材质类别的点（真值标签，非代理）
  2. 颜色不连续性：与邻域点的 RGB 平均距离 —— MDM 想用的那个信号
  3. 几何不连续性：邻域 PCA 形状量（散射度/平面度/竖直度）的变化 —— **对照组**

若 (2) 在边界处显著高于内部，MDM 的信号存在；若 (3) 同样高，则几何已经能标出这条
边界，MDM 的增量有限，必须在论文里如实说明并改用「几何连续但材质突变」的子集
（G2P 所说的 geometric bias 场景）作为卖点，而不是泛泛地说「材质边界」。

用法：
    python3 scripts/mdm_signal_probe.py --n_frames 200 \
        --out results/mdm_signal_val.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_urmat_dataset import MATERIAL_NAMES as MAT_NAMES

RADIUS = 0.15      # 邻域半径（米）；太小则邻域稀疏，太大则跨越遮挡边缘污染统计
MAX_NB = 64        # 每点最多取这么多邻居，控制内存
MIN_NB = 4         # 邻居不足的点不参与统计


def frame_stats(xyz, rgb, mat, tree):
    """返回逐点的 (is_boundary, color_disc, geom_disc)。"""
    nb_idx = tree.query_ball_point(xyz, RADIUS, return_sorted=False)
    n = len(xyz)
    is_b, c_disc, g_disc = np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan)

    # 先算逐点局部 PCA 形状量（散射度、平面度、竖直度）
    _, k_idx = tree.query(xyz, k=min(17, n))
    nbx = xyz[k_idx]
    ctr = nbx.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', nbx - ctr, nbx - ctr) / nbx.shape[1]
    w, v = np.linalg.eigh(cov)
    l1, l2, l3 = w[:, 2], w[:, 1], w[:, 0]
    eps = 1e-12
    shape = np.stack([l3 / (l1 + eps), (l2 - l3) / (l1 + eps),
                      np.abs(v[:, 2, 2]), np.sqrt(l1)], axis=1)

    for i in range(n):
        j = np.asarray(nb_idx[i], dtype=np.int64)
        if j.size > MAX_NB:                               # 随机抽稀，避免大邻域拖慢
            j = j[:MAX_NB]
        j = j[j != i]
        if j.size < MIN_NB:
            continue
        is_b[i] = np.any(mat[j] != mat[i])
        c_disc[i] = np.linalg.norm(rgb[j] - rgb[i], axis=1).mean()
        g_disc[i] = np.abs(shape[j] - shape[i]).sum(axis=1).mean()
    return is_b, c_disc, g_disc


def auc(score, label):
    """score 越大越可能是 label=True 的 AUC（Mann-Whitney U，不依赖阈值）。"""
    m = np.isfinite(score)
    s, y = score[m], label[m]
    if y.sum() == 0 or (~y).sum() == 0:
        return float('nan')
    r = np.empty(len(s), np.float64)
    order = np.argsort(s, kind='mergesort')
    r[order] = np.arange(1, len(s) + 1)
    # 处理并列：同分取平均秩
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        k = i
        while k + 1 < len(s_sorted) and s_sorted[k + 1] == s_sorted[i]:
            k += 1
        if k > i:
            r[order[i:k + 1]] = (i + k + 2) / 2.0
        i = k + 1
    n_pos, n_neg = y.sum(), (~y).sum()
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/mnt/data/datasets/UR-MAT/seg_16k')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n_frames', type=int, default=200)
    ap.add_argument('--out', default='results/mdm_signal_val.json')
    args = ap.parse_args()

    from scipy.spatial import cKDTree
    frames = sorted(glob.glob(os.path.join(args.data, args.split, '*.npz')))
    frames = frames[:args.n_frames]
    print(f'[mdm] {args.split} 取 {len(frames)} 帧，邻域半径 {RADIUS} m，纯 CPU', flush=True)

    B, C, G, M = [], [], [], []
    for k, p in enumerate(frames):
        d = np.load(p)
        xyz = d['xyz'].astype(np.float32)
        b, c, g = frame_stats(xyz, d['rgb'].astype(np.float32),
                              d['mat'].astype(np.int64), cKDTree(xyz))
        B.append(b); C.append(c); G.append(g); M.append(d['mat'].astype(np.int64))
        if (k + 1) % 50 == 0:
            print(f'  已处理 {k + 1}/{len(frames)} 帧', flush=True)

    is_b = np.concatenate(B)
    c_disc = np.concatenate(C)
    g_disc = np.concatenate(G)
    mat = np.concatenate(M)
    ok = np.isfinite(c_disc) & np.isfinite(g_disc)

    print('\n' + '=' * 78)
    print('材质边界的可发现性（UR-MAT 逐点材质真值；投影误差为零）')
    print('=' * 78)
    print(f'参与统计的点 {ok.sum()}，其中材质边界点 {is_b[ok].sum()} '
          f'（{is_b[ok].mean() * 100:.1f}%）')
    print(f"\n{'信号':<28}{'边界均值':>12}{'内部均值':>12}{'比值':>9}{'AUC':>9}")
    print('-' * 78)
    res = {'n_frames': len(frames), 'radius_m': RADIUS,
           'n_pts': int(ok.sum()), 'boundary_frac': float(is_b[ok].mean())}
    for name, s in (('颜色不连续性（MDM 用的）', c_disc), ('几何不连续性（对照）', g_disc)):
        a = auc(s, is_b)
        mb, mi = float(np.nanmean(s[ok & is_b])), float(np.nanmean(s[ok & ~is_b]))
        res[name] = {'boundary_mean': round(mb, 5), 'interior_mean': round(mi, 5),
                     'ratio': round(mb / (mi + 1e-12), 3), 'auc': round(a, 4)}
        print(f'{name:<28}{mb:>12.4f}{mi:>12.4f}{mb / (mi + 1e-12):>9.2f}{a:>9.3f}')

    # 关键子集：几何连续但材质突变（G2P 的 geometric bias 场景）——这才是不与几何重复的部分
    med = np.nanmedian(g_disc[ok])
    sub = ok & (g_disc <= med)
    # 子集外的点必须掩成 NaN，否则它们会以负样本身份混进来，把子集 AUC 算歪
    a_sub = auc(np.where(sub, c_disc, np.nan), is_b)
    res['geom_continuous_subset'] = {
        'threshold': round(float(med), 5), 'n_pts': int(sub.sum()),
        'boundary_frac': float(is_b[sub].mean()), 'color_auc': round(a_sub, 4)}
    print('\n关键子集：**几何连续**（几何不连续性 ≤ 中位数）但材质突变的点')
    print(f'  {sub.sum()} 点，其中材质边界 {is_b[sub].mean() * 100:.1f}%；'
          f'颜色不连续性在该子集上的 AUC = {a_sub:.3f}')
    print('  （几何在这批点上没有信号可用，若颜色 AUC 仍明显 >0.5，'
          '说明 MDM 在几何失效区确实带来新信息）')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f'\n已写出：{args.out}')


if __name__ == '__main__':
    main()
