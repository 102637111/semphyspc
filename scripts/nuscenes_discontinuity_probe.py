#!/usr/bin/env python3
"""在**真实 nuScenes** 数据上测「外观不连续性」这个信号的决策门（纯 CPU，模型无关）。

背景：`mdm_signal_probe.py` 在 UR-MAT 合成渲染上测到，颜色不连续性检测材质边界的
AUC = 0.893，而几何不连续性只有 0.513（≈瞎）。但那组数字有三个上界性质，不能直接
当成真实数据的结论：

  1. UR-MAT 是**合成渲染**，材质按构造决定外观（反射率/光泽是渲染参数）；
  2. UR-MAT 里**材质 ≡ 物体**（一个网格只有一个材质，已核实），所以那条 AUC 严格说是
     「检测物体边界」，而不是「同一物体内部材质突变」；
  3. 该数据集 RGB 与深度**像素对齐**，投影误差为零。

本脚本在真实 nuScenes 16k 子采样上重测同一件事，把三个上界全部去掉：

  - 靶子换成**实例边界**（`(sem, inst)` 发生变化）与**语义边界**两档；
  - 邻域用 3D KNN（不经过图像投影，因此**完全绕开**投影误差与 6 相机覆盖问题）；
  - 单独统计 **rgb 全 0（6 相机都没看到、被填黑）**的点：那里的分数无定义，
    必须被掩掉，而且它在各类上分布极不均（行人 2.8%、摩托 32.5%）。

判据：若真实数据上「几何连续子集」里的颜色 AUC 仍明显高于 0.6，MDM 值得上 GPU；
若掉到 0.6 以下，这条线当场放弃。

用法：
    python3 scripts/nuscenes_discontinuity_probe.py --n_frames 300 \
        --out results/nuscenes_discontinuity_val.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mdm_signal_probe import auc

K_NN = 16
MIN_NB = 4


def frame_stats(xyz, rgb, sem, inst, boundary_key, radius=0.0):
    """逐点 (是否有边界, 颜色不连续性, 几何不连续性, 是否被填黑)。

    boundary_key: 'inst' 用 (sem, inst)，'sem' 只用 sem。
    radius > 0 时用固定半径邻域（与 UR-MAT 的 mdm_signal_probe 同口径）；
    否则用自适应 KNN。

    **公平性关键**：RGB 全 0 的填黑点必须从**邻居**里也剔除。否则一个正常点只要有
    一个填黑邻居，它的颜色不连续性就会被凭空抬高，颜色 AUC 被系统性压低——那样
    得出的「颜色没用」是探针自己造出来的，不是数据的事实。
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    n = len(xyz)

    if radius > 0:
        idx = tree.query_ball_point(xyz, radius, return_sorted=False)
        K = max((len(x) for x in idx), default=0)
        if K == 0:
            return (np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan),
                    ~rgb.any(axis=1))
        pad = np.zeros((n, K), np.int64)                   # 用自身填充，靠 mask 排除
        pad[:] = np.arange(n)[:, None]
        m = np.zeros((n, K), bool)
        for i, j in enumerate(idx):
            j = np.asarray(j, np.int64)
            j = j[j != i]
            if j.size > K:
                j = j[:K]
            pad[i, :j.size] = j
            m[i, :j.size] = True
    else:
        _, idx = tree.query(xyz, k=min(K_NN + 1, n))
        pad = idx[:, 1:]
        m = np.ones_like(pad, bool)
        K = pad.shape[1]

    zeros = ~rgb.any(axis=1)                               # 6 相机都没看到 → 填黑
    m &= ~zeros[pad]                                       # 剔除填黑邻居
    m &= ~zeros[:, None]                                   # 自己也是填黑就不参与
    nb_cnt = m.sum(axis=1)

    # 颜色不连续性：与**有效**邻居的 RGB 欧氏距离均值
    d = np.linalg.norm(rgb[pad] - rgb[:, None, :], axis=2)  # (N,K)
    c_disc = np.where(nb_cnt > 0,
                      np.where(m, d, 0.0).sum(1) / np.maximum(nb_cnt, 1), np.nan)

    # 对照：**只留色度、去掉亮度**。光照变化、阴影、图像边缘变暗都会产生 RGB 距离，
    # 但那不是材质。把每个点归一化到单位亮度后只剩色度，若色度 AUC 掉到 0.5，
    # 说明前面那个「颜色信号」其实只是明暗，与材质无关。这是 MDM 最该自证的一条。
    lit = np.linalg.norm(rgb, axis=1, keepdims=True) + 1e-6
    ch = rgb / lit
    k2 = np.linalg.norm(ch[pad] - ch[:, None, :], axis=2)
    k_disc = np.where(nb_cnt > 0,
                      np.where(m, k2, 0.0).sum(1) / np.maximum(nb_cnt, 1), np.nan)

    # 几何不连续性：邻域 PCA 形状量之差（与 mdm_signal_probe 同口径）
    nbx = xyz[pad]
    ctr = nbx.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', nbx - ctr, nbx - ctr) / max(K, 1)
    w, v = np.linalg.eigh(cov)
    l1, l2, l3 = w[:, 2], w[:, 1], w[:, 0]
    eps = 1e-12
    shape = np.stack([l3 / (l1 + eps), (l2 - l3) / (l1 + eps),
                      np.abs(v[:, 2, 2]), np.sqrt(l1)], axis=1)
    gd = np.abs(shape[pad] - shape[:, None, :]).sum(axis=2)
    g_disc = np.where(nb_cnt > 0,
                      np.where(m, gd, 0.0).sum(1) / np.maximum(nb_cnt, 1), np.nan)

    if boundary_key == 'sem':
        key = sem
    else:
        key = sem.astype(np.int64) * 100000 + np.clip(inst, -1, 99999)
    is_b = np.any((key[pad] != key[:, None]) & m, axis=1)

    thin = nb_cnt < MIN_NB                                 # 有效邻居太少的点弃掉
    for arr in (c_disc, g_disc, k_disc):
        arr[thin] = np.nan
    return is_b, c_disc, g_disc, zeros, k_disc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/nuscenes_seg_full_16k')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n_frames', type=int, default=300)
    ap.add_argument('--radius', type=float, default=0.0,
                    help='>0 用固定半径邻域（与 UR-MAT 探针同口径）；0 = 自适应 KNN')
    ap.add_argument('--out', default='results/nuscenes_discontinuity_val.json')
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.data, args.split, '*.npz')))[:args.n_frames]
    if not frames:
        raise SystemExit(f'没有找到 npz：{args.data}/{args.split}')
    print(f'[disc] nuScenes {args.split} 取 {len(frames)} 帧，3D KNN k={K_NN}（不经过投影）',
          flush=True)

    acc = {k: {'b': [], 'c': [], 'g': [], 'z': [], 'k': []} for k in ('inst', 'sem')}
    for i, p in enumerate(frames):
        d = np.load(p)
        xyz = d['xyz'].astype(np.float32)
        rgb = d['rgb'].astype(np.float32)
        sem, inst = d['sem'].astype(np.int64), d['inst'].astype(np.int64)
        for k in ('inst', 'sem'):
            b, c, g, z, kd = frame_stats(xyz, rgb, sem, inst, k, radius=args.radius)
            acc[k]['b'].append(b); acc[k]['c'].append(c)
            acc[k]['g'].append(g); acc[k]['z'].append(z); acc[k]['k'].append(kd)
        if (i + 1) % 50 == 0:
            print(f'  已处理 {i + 1}/{len(frames)} 帧', flush=True)

    res = {'n_frames': len(frames), 'k_nn': K_NN}
    print('\n' + '=' * 94)
    print('真实 nuScenes 上的外观不连续性信号（3D KNN，不经过图像投影，故无投影误差）')
    print('=' * 94)
    print(f"{'靶子':<24}{'边界占比':>10}{'颜色 AUC':>11}{'色度 AUC':>11}"
          f"{'几何 AUC':>11}{'填黑':>8}")
    print('-' * 94)
    for k, name in (('inst', '实例边界 (sem,inst)'), ('sem', '语义边界 (sem)')):
        b = np.concatenate(acc[k]['b']); c = np.concatenate(acc[k]['c'])
        g = np.concatenate(acc[k]['g']); z = np.concatenate(acc[k]['z'])
        kd = np.concatenate(acc[k]['k'])
        res[f'{k}_all'] = {'boundary_frac': round(float(b.mean()), 4),
                           'color_auc': round(auc(c, b), 4),
                           'chroma_auc': round(auc(kd, b), 4),
                           'geom_auc': round(auc(g, b), 4),
                           'black_frac': round(float(z.mean()), 4)}
        print(f'{name:<24}{b.mean() * 100:>9.1f}%{auc(c, b):>11.3f}{auc(kd, b):>11.3f}'
              f'{auc(g, b):>11.3f}{z.mean() * 100:>7.1f}%')
    print('  色度 = 去掉亮度后的纯颜色（RGB 归一化到单位亮度的距离）。'
          '若色度 AUC 塌到 0.5，')
    print('  说明「颜色不连续性」其实只是明暗/阴影/图像边缘变暗，与材质无关。')

    # 关键：掩掉填黑点、并且只看几何连续的子集 —— 这才是 MDM 声称独占的那部分信息
    print('\n' + '=' * 86)
    print('关键子集：掩掉填黑点 + 只保留几何连续（几何不连续性 ≤ 中位数）')
    print('=' * 86)
    print(f"{'靶子':<26}{'参与点数':>12}{'边界占比':>10}{'颜色 AUC':>11}")
    print('-' * 86)
    print(f"\n{'靶子':<24}{'参与点数':>12}{'边界占比':>10}{'颜色 AUC':>11}{'色度 AUC':>11}")
    print('-' * 94)
    for k, name in (('inst', '实例边界 (sem,inst)'), ('sem', '语义边界 (sem)')):
        b = np.concatenate(acc[k]['b']); c = np.concatenate(acc[k]['c'])
        g = np.concatenate(acc[k]['g']); z = np.concatenate(acc[k]['z'])
        kd = np.concatenate(acc[k]['k'])
        m = (~z) & np.isfinite(c) & np.isfinite(g)
        med = np.nanmedian(g[m])
        sub = m & (g <= med)
        # 子集外的点必须掩成 NaN：否则它们会以「负样本」身份混进 AUC，
        # 把子集上的可分性算成一个既非子集、也非全集的怪数。
        # 注意：这里**不能**再报子集内的几何 AUC —— 子集本身就是按几何分数截断定义的，
        # 在截断后重新排序几何分数是没有意义的数字（必然虚高）。
        a, ak = auc(np.where(sub, c, np.nan), b), auc(np.where(sub, kd, np.nan), b)
        res[f'{k}_geom_continuous'] = {'n_pts': int(sub.sum()),
                                       'boundary_frac': round(float(b[sub].mean()), 4),
                                       'color_auc': round(a, 4),
                                       'chroma_auc': round(ak, 4)}
        print(f'{name:<24}{sub.sum():>12}{b[sub].mean() * 100:>9.1f}%{a:>11.3f}{ak:>11.3f}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f'\n已写出：{args.out}')


if __name__ == '__main__':
    main()
