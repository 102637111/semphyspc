#!/usr/bin/env python3
"""材质任务的「颜色 vs 几何」判别力对照（纯 CPU、模型无关）。

动机：论文的路线判断需要一条**对称的证据**。已有的事实是——在 nuScenes 上，
只用 RGB 三通道预测 32 类语义的平衡精度仅 0.054（随机 0.031），pedestrian/car
召回全 0.000（见 `scripts/rgb_validity_probe.py`）：颜色对**语义**几乎没有判别力。
但材质分支（§3.3、UR-MAT）一直宣称「物理材质主要靠外观/颜色」，这个说法
在仓库里**从未被定量检验过**——`cfgs/seg_urmat_16k.yaml` 用的是 in_channels=6
（RGB+XYZ），单任务材质 mIoU 0.9406，但没有任何无颜色对照。

本脚本在 UR-MAT 的 16k 子采样帧上，用同一个模型无关分类器（标准化后的最近类均值，
即等方差对角高斯贝叶斯）比较两组特征对 **14 类材质** 的判别力：

  颜色   rgb（3 维）
  几何   离地高度 / 局部 PCA 形状（线性度、平面度、散射度）/ 竖直度 / 局部尺度 / 距离（6 维）
  合并   颜色 + 几何（9 维）

判据与语义探针一致：用**平衡精度**（各类召回的均值，等价于等先验下的平均精度）
对比随机基线 1/14=0.0714，并给出逐类召回。

预期（也就是待检验的假设）：
  - 颜色 → 材质 的平衡精度显著高于几何 → 「材质走 RGB」成立；
  - 几何 → 材质 明显更差 → 材质不能只靠几何；
  - 二者合并 ≥ 各自单独 → 互补而非冗余。

注意一个口径问题：UR-MAT 帧是合成渲染，材质与外观（反射率/光泽）是**被渲染参数
直接决定**的，所以颜色对材质的判别力在这里是**上界**，不能直接外推到真实 LiDAR。
这个上界本身仍然有意义：它说明「材质—颜色」这条通路在原理上成立，真实数据上的
差距属于域适应问题（见论文 §5 局限），而不是通路不成立。

用法：
    python3 scripts/material_modality_probe.py --n_frames 200 \
        --out results/material_modality_val.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_urmat_dataset import MATERIAL_NAMES as MAT_NAMES   # 14 类，单一来源，避免两处漂移

K_NN = 16            # 局部 PCA 邻域
MIN_CLS_PTS = 200    # 逐类召回只报告测试样本不少于这么多的类


def ground_normal_height(xyz):
    """用最低的 20% 点迭代最小二乘拟合地面，返回 (离地高度, 地面上方向)。

    UR-MAT 帧是相机坐标系（x 右 / y 下 / z 前），所以"上"是 −y。
    迭代两次：先取 y 最大的 20% 点（最低处）拟合，再按到平面的距离收紧。
    """
    h = np.ones((len(xyz), 1))
    q = np.c_[xyz, h]
    sel = xyz[:, 1] >= np.percentile(xyz[:, 1], 80)      # y 大 = 低
    n = None
    for _ in range(2):
        if sel.sum() < 50:
            break
        # 平面 a·x + b·y + c·z + d = 0，取最小奇异向量
        _, _, vt = np.linalg.svd(q[sel], full_matrices=False)
        n = vt[-1]
        d = -(q @ n)
        sel = np.abs(d) <= np.percentile(np.abs(d), 40)
    if n is None:
        return np.zeros(len(xyz), np.float32), np.array([0, -1, 0], np.float32)
    n = n[:3] / (np.linalg.norm(n[:3]) + 1e-9)
    if n[1] > 0:                                          # 统一朝"上"（−y）
        n = -n
    # 离地高度 = 沿法向到平面上一点（取内点均值）的带符号距离，地面为 0
    p0 = xyz[sel].mean(axis=0)
    return ((xyz - p0) @ n).astype(np.float32), n.astype(np.float32)


def geom_feats(xyz, tree):
    """逐点几何特征，分成两组返回。

    **必须分开**：合成街景里材质与「点在场景中的位置」强相关（鹅卵石=地面、
    玻璃=楼上窗户、塑料=近处物体），所以「离地高度 / 距离」这类**位置**特征
    会伪装成几何判别力，而它换一个场景就失效。只有**局部形状**量
    （线性度/平面度/散射度/竖直度/尺度）是平移不变的、真正描述表面性质的特征。

    返回 {'color': …, 'shape': …, 'position': …}。
    """
    h, _ = ground_normal_height(xyz)
    _, idx = tree.query(xyz, k=K_NN)                      # (N,K)
    nb = xyz[idx]                                         # (N,K,3)
    c = nb.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', nb - c, nb - c) / K_NN
    w, v = np.linalg.eigh(cov)                            # 升序
    l1, l2, l3 = w[:, 2], w[:, 1], w[:, 0]
    eps = 1e-12
    shape = np.stack([(l1 - l2) / (l1 + eps),             # 线性度
                      (l2 - l3) / (l1 + eps),             # 平面度
                      l3 / (l1 + eps),                    # 散射度
                      np.abs(v[:, 2, 2]),                 # 竖直度（法向 vs 上）
                      np.sqrt(l1)],                       # 局部尺度
                     axis=1).astype(np.float32)
    position = np.stack([h, np.linalg.norm(xyz, axis=1)], axis=1).astype(np.float32)
    return shape, position


def collect(frames, n_frames):
    """返回 {'color'/'shape'/'position': 拼接后的特征} 与标签。"""
    cols = {'color': [], 'shape': [], 'position': []}
    labels = []
    used = 0
    from scipy.spatial import cKDTree
    for p in frames[:n_frames]:
        d = np.load(p)
        xyz = d['xyz'].astype(np.float32)
        mat = d['mat'].astype(np.int64)
        shape, position = geom_feats(xyz, cKDTree(xyz))
        cols['color'].append(d['rgb'].astype(np.float32))
        cols['shape'].append(shape)
        cols['position'].append(position)
        labels.append(mat)
        used += 1
        if used % 50 == 0:
            print(f'  已处理 {used}/{min(n_frames, len(frames))} 帧', flush=True)
    return {k: np.concatenate(v) for k, v in cols.items()}, np.concatenate(labels), used


def eval_feats(X, y, rng, classes, clf='ncm'):
    """标准化 + 分类器。返回平衡精度与逐类召回。

    clf='ncm'：最近类均值（等协方差高斯）。与 `rgb_validity_probe.py` 口径一致，
               但它是**等权**的——多塞进几个没用的维度就会稀释判别力，所以
               「颜色+形状 < 颜色」这种结果多半是分类器假象，不能当作不互补的证据。
    clf='lda'：线性判别分析（闭式解，白化共享协方差）。它会给维度加权、压低
               无关方向，是检验「特征是否互补」的公平裁判。两者一起看才说明问题。
    """
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    perm = rng.permutation(len(y))
    n_tr = int(0.7 * len(y))
    tr, te = perm[:n_tr], perm[n_tr:]
    if clf == 'ncm':
        cent, keep = [], []
        for c in classes:
            m = tr[y[tr] == c]
            if len(m) == 0:
                continue
            cent.append(X[m].mean(0))
            keep.append(c)
        d = ((X[te][:, None, :] - np.stack(cent)[None]) ** 2).sum(-1)
        pred = np.asarray(keep)[d.argmin(1)]
    else:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        m = LinearDiscriminantAnalysis()
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        keep = classes
    yte = y[te]                                          # 测试集标签，与 pred 同序
    per, recalls = {}, []
    for c in keep:
        m_ = np.flatnonzero(yte == c)                    # pred 的下标，不是原数组下标
        if len(m_) == 0:
            continue                                     # 该类没进测试集
        r = float((pred[m_] == c).mean())
        recalls.append(r)
        if len(m_) >= MIN_CLS_PTS:
            per[MAT_NAMES[c]] = {'n': int(len(m_)), 'recall': round(r, 4)}
    bal = float(np.mean(recalls)) if recalls else 0.0
    # 只在样本充足的类上再算一次，避免个位数样本的类把均值搅乱
    bal_robust = float(np.mean([v['recall'] for v in per.values()])) if per else 0.0
    return bal, bal_robust, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/mnt/data/datasets/UR-MAT/seg_16k')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n_frames', type=int, default=200)
    ap.add_argument('--out', default='results/material_modality_val.json')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.data, args.split, '*.npz')))
    if not frames:
        raise SystemExit(f'没有找到 npz：{args.data}/{args.split}')
    print(f'[mat] {args.split} 共 {len(frames)} 帧，取前 {args.n_frames} 帧，纯 CPU', flush=True)

    feats, y, used = collect(frames, args.n_frames)
    c, s, p = feats['color'], feats['shape'], feats['position']
    classes = np.unique(y)
    print(f'\n合计 {len(y)} 点，{len(classes)} 类：{classes.tolist()}')
    cnt = {int(k): int((y == k).sum()) for k in classes}
    print('逐类点数：', {MAT_NAMES[k]: v for k, v in sorted(cnt.items())})

    rng = np.random.default_rng(args.seed)
    res = {'n_frames': used, 'n_pts': int(len(y)), 'n_classes': int(len(classes)),
           'per_class_points': {MAT_NAMES[k]: v for k, v in sorted(cnt.items())},
           'random_baseline': round(1.0 / len(classes), 4)}

    SUBSETS = [
        ('颜色 RGB', c),
        ('局部形状', s),
        ('位置(对照)', p),                 # 含场景布局混淆，只看它说明"位置也能猜材质"
        ('颜色+形状', np.c_[c, s]),
        ('颜色+形状+位置', np.c_[c, s, p]),
    ]
    print('\n' + '=' * 88)
    print('材质判别力：颜色 vs 几何（标准化，70/30 留出）')
    print(f'随机基线 = 1/{len(classes)} = {1.0 / len(classes):.4f}；'
          f'「局部形状」是平移不变的表面量，「位置」是场景布局对照')
    print('NCM = 最近类均值（等权，与语义探针口径一致）；'
          'LDA = 线性判别（会加权，用来判互补性）')
    print('=' * 88)
    print(f"{'特征':<18}{'维数':>5}{'NCM 平衡精度':>14}{'相对随机':>9}"
          f"{'LDA 平衡精度':>14}{'相对随机':>9}")
    print('-' * 88)
    for name, X in SUBSETS:
        row = {'dim': int(X.shape[1])}
        for clf in ('ncm', 'lda'):
            bal, bal_r, per = eval_feats(X, y, rng, classes, clf=clf)
            row[clf] = {'balanced_acc': round(bal, 4),
                        'balanced_acc_robust': round(bal_r, 4), 'per_class': per}
            row[f'{clf}_bal'] = bal
        res[name] = row
        print(f"{name:<18}{X.shape[1]:>5}"
              f"{row['ncm_bal']:>14.4f}{row['ncm_bal'] / (1.0 / len(classes)):>8.1f}×"
              f"{row['lda_bal']:>14.4f}{row['lda_bal'] / (1.0 / len(classes)):>8.1f}×")

    print('\n' + '=' * 88)
    print(f'逐类召回（测试样本 ≥{MIN_CLS_PTS} 的类；NCM / LDA 两个分类器并排）')
    print('=' * 88)
    keys = sorted(set().union(*[set(res[n]['ncm']['per_class']) for n, _ in SUBSETS],
                              *[set(res[n]['lda']['per_class']) for n, _ in SUBSETS]))
    print(f"{'材质':<12}{'测试点数':>9}" + ''.join(f'{n:>13}' for n, _ in SUBSETS))
    print(f"{'':<21}" + ''.join(f"{'NCM / LDA':>13}" for _ in SUBSETS))
    print('-' * 88)
    for k in keys:
        cells, n_pts = [], 0
        for n, _ in SUBSETS:
            a = res[n]['ncm']['per_class'].get(k)
            b = res[n]['lda']['per_class'].get(k)
            n_pts = max(n_pts, (a or b or {}).get('n', 0))
            cells.append(f"{(a or {}).get('recall', float('nan')):.2f} / "
                         f"{(b or {}).get('recall', float('nan')):.2f}"
                         if (a or b) else '—')
        print(f'{k:<12}{n_pts:>9}' + ''.join(f'{x:>13}' for x in cells))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(f'\n已写出：{args.out}')


if __name__ == '__main__':
    main()
