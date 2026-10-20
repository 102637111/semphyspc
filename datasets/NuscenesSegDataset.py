"""
NuscenesSegDataset - nuScenes 逐点语义分割数据集（懒加载 per-sample .npz）。

与 scripts/build_nuscenes_seg_dataset.py 的输出配套：
  DATA_PATH/<subset>/<lidar_token>.npz
    每个 npz: {xyz (N,3), rgb (N,3) in [0,1], sem (N,) int16, inst (N,) int16}

Config 期望:
  DATA_PATH: 目录（含 train/ 和 val/ 子目录）
  N_POINTS: 每样本点数
  NUM_CATEGORY: 语义类别数（默认 32 = nuScenes-lidarseg 全类别）
  IN_CHANNELS: 3 或 6（6 = XYZ+RGB，默认 6）
"""

import os, json, numpy as np, torch
from pathlib import Path
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *

# P3：需要过采样的稀有类（nuScenes-lidarseg id）。
# 前 5 个是 §4.14 认定的「标注量不可约」类，其余是稀有 thing 类。
RARE_CLASSES = [3, 6, 7, 5, 8, 4, 14, 21, 12, 9]


@DATASETS.register_module()
class NuscenesSegDataset(Dataset):
    """nuScenes per-point semantic segmentation dataset."""

    def __init__(self, config):
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.in_channels = getattr(config, 'IN_CHANNELS', 6)
        data_dir = getattr(config, 'DATA_PATH', 'data/nuscenes_seg')

        split_dir = Path(data_dir) / self.subset
        if not split_dir.exists():
            raise FileNotFoundError(
                f'{split_dir} not found. Run build_nuscenes_seg_dataset.py first.')

        self.files = sorted(split_dir.glob('*.npz'))
        if len(self.files) == 0:
            raise FileNotFoundError(f'{split_dir} 下没有 .npz 文件')

        self.num_category = getattr(config, 'NUM_CATEGORY', 32)

        # 合并相似类：config.CLASS_MAP 给 32 类各一个「新类 id」的映射表。
        # 缺省 None = 不合并（原始 32 类）。映射表长度须等于 32（lidarseg 全类别数）。
        self.class_map = getattr(config, 'CLASS_MAP', None)
        if self.class_map is not None:
            self.class_map = np.asarray(self.class_map, dtype=np.int64)
            assert self.class_map.shape == (32,), 'CLASS_MAP 必须是 32 长度的列表'
            assert self.class_map.max() < self.num_category, \
                f'CLASS_MAP 映射越界：max={self.class_map.max()} >= num_category={self.num_category}'

        # P3：每帧权重（供 RareAwareSampler 用）；不需要时为 None
        self.rare_weights = None
        if getattr(config, 'RARE_AWARE_SAMPLING', False) and self.subset == 'train':
            self.rare_weights = self._build_rare_weights(
                max_rep=float(getattr(config, 'RARE_MAX_REP', 25.0)),
                exponent=float(getattr(config, 'RARE_EXPONENT', 0.5)))

        print_log(f'[NuscenesSegDataset] {self.subset}: {len(self.files)} samples, '
                  f'{self.num_category} classes, in_channels={self.in_channels}, '
                  f'rare_aware={self.rare_weights is not None}',
                  logger='NuscenesSeg')

    def _build_rare_weights(self, max_rep=25.0, exponent=0.5):
        """扫一遍每个 npz 的 sem/inst，按类实例数算出每帧的过采样权重。

        实例键必须用 (sem, inst)：`inst` 在同一帧内会跨类复用（见 §4.14 的坑）。
        结果缓存成 json，避免每次训练重新扫 2.8 万个 npz。
        """
        cache = Path(self.files[0]).parent.parent / f'rare_index_{self.subset}.json'
        tag = f'{len(self.files)}帧_v1'      # 口径变了就改这里，避免读到过期缓存
        d = None
        if cache.exists():
            try:
                with open(cache) as f:
                    d = json.load(f)
                if d.get('tag') != tag:
                    d = None
            except (OSError, ValueError):
                d = None
        if d is not None:
            counts = {int(k): v for k, v in d['class_instances'].items()}
            per_file = d['per_file']
            print_log(f'[NuscenesSegDataset] 复用稀有类索引 {cache}', logger='NuscenesSeg')
        else:
            counts = {c: 0 for c in RARE_CLASSES}
            per_file = {}
            for i, fp in enumerate(self.files):
                z = np.load(str(fp))
                sem = np.clip(z['sem'].astype(np.int64), 0, 31)
                inst = z['inst'].astype(np.int64)
                # inst<0 表示非实例点，必须排除，否则会凭空多算一个「实例」
                key = sem * 1000 + np.clip(inst, 0, 999)
                present = []
                for c in RARE_CLASSES:
                    sel = (sem == c) & (inst >= 0)
                    n = int(np.unique(key[sel]).size) if sel.any() else 0
                    if n > 0:
                        counts[c] += n
                        present.append(c)
                per_file[fp.stem] = present
                if (i + 1) % 5000 == 0:
                    print_log(f'  ... 稀有类索引 {i + 1}/{len(self.files)}', logger='NuscenesSeg')
            try:
                with open(cache, 'w') as f:
                    json.dump({'tag': tag,
                               'class_instances': {str(k): v for k, v in counts.items()},
                               'per_file': per_file}, f)
            except OSError as e:
                print_log(f'[warn] 稀有类索引缓存失败：{e}', logger='NuscenesSeg')

        nz = [v for v in counts.values() if v > 0]
        nmax = max(nz) if nz else 1
        # w_c = (N_max / N_c)^exponent，裁剪到 [1, max_rep]
        rep = {c: float(np.clip((nmax / max(1, counts[c])) ** exponent, 1.0, max_rep))
               for c in RARE_CLASSES}
        print_log('[NuscenesSegDataset] 稀有类实例数与过采样倍率：' + ', '.join(
            f'{c}:{counts[c]}(×{rep[c]:.1f})' for c in RARE_CLASSES),
            logger='NuscenesSeg')

        w = np.ones(len(self.files), dtype=np.float64)
        for i, fp in enumerate(self.files):
            for c in per_file.get(fp.stem, []):
                w[i] = max(w[i], rep[c])
        n_boost = int((w > 1.0).sum())
        print_log(f'[NuscenesSegDataset] {n_boost}/{len(self.files)} 帧被过采样，'
                  f'权重和 {w.sum():.0f}（均匀时 {len(self.files)}）',
                  logger='NuscenesSeg')
        return w

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        d = np.load(str(self.files[index]))
        xyz = d['xyz'].astype(np.float32)
        rgb = d['rgb'].astype(np.float32)
        sem = d['sem'].astype(np.int64)
        # 合并相似类：把 lidarseg 32 类 id 映射到粗粒度类 id（padding 前做，此时 sem ∈ [0,31]）
        if self.class_map is not None:
            sem = self.class_map[sem]

        N = xyz.shape[0]
        # 保证点数一致：不足则补零（用 -1 mask 忽略），超出则随机截取
        if N != self.npoints:
            if N > self.npoints:
                idx = np.random.choice(N, self.npoints, replace=False)
                xyz, rgb, sem = xyz[idx], rgb[idx], sem[idx]
            else:
                pad = self.npoints - N
                xyz = np.pad(xyz, ((0, pad), (0, 0)), mode='constant')
                rgb = np.pad(rgb, ((0, pad), (0, 0)), mode='constant')
                sem = np.pad(sem, (0, pad), mode='constant', constant_values=-1)

        # 训练增强：绕 Z 轴随机旋转（室外驾驶场景 yaw 任意）
        if self.subset == 'train':
            ang = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            xyz = xyz @ R.T

        if self.in_channels == 6:
            pts = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
        else:
            pts = xyz

        pts_tensor = torch.from_numpy(pts).float()
        sem_tensor = torch.from_numpy(sem).long()

        return 'NuscenesSeg', self.files[index].stem, (pts_tensor, sem_tensor)
