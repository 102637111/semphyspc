"""
UrmatSegDataset - UR-MAT 逐点材质分割数据集（懒加载 per-sample .npz）。

与 scripts/build_urmat_dataset.py 的输出配套：
  DATA_PATH/<subset>/<frame>.npz
    每个 npz: {xyz (N,3) [m], rgb (N,3) in [0,1], mat (N,) int16}

Config 期望:
  DATA_PATH: 目录（含 train/ 和 val/ 子目录）
  N_POINTS: 每样本点数
  NUM_CATEGORY: 材质类别数（UR-MAT = 14）
  IN_CHANNELS: 6（XYZ+RGB）

材质类别 (label -> name):
  0 Limestone, 1 Slate, 2 Glass, 3 Steel, 4 Brick, 5 Cobblestone,
  6 Plaster, 7 Concrete, 8 Gravel, 9 Tiles, 10 Wood, 11 Asphalt,
  12 Plastic, 13 Unknown
"""

import os, numpy as np, torch
from pathlib import Path
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *


@DATASETS.register_module()
class UrmatSegDataset(Dataset):
    """UR-MAT per-point material segmentation dataset."""

    def __init__(self, config):
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.in_channels = getattr(config, 'IN_CHANNELS', 6)
        data_dir = getattr(config, 'DATA_PATH', 'data/urmat_seg')

        split_dir = Path(data_dir) / self.subset
        if not split_dir.exists():
            raise FileNotFoundError(
                f'{split_dir} not found. Run build_urmat_dataset.py first.')

        self.files = sorted(split_dir.glob('*.npz'))
        if len(self.files) == 0:
            raise FileNotFoundError(f'{split_dir} 下没有 .npz 文件')

        self.num_category = getattr(config, 'NUM_CATEGORY', 14)

        print_log(f'[UrmatSegDataset] {self.subset}: {len(self.files)} samples, '
                  f'{self.num_category} classes, in_channels={self.in_channels}',
                  logger='UrmatSeg')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        d = np.load(str(self.files[index]))
        xyz = d['xyz'].astype(np.float32)
        rgb = d['rgb'].astype(np.float32)
        mat = d['mat'].astype(np.int64)

        # 裁剪远距离/天空点 (z > 100m)：避免 Hilbert 序列化 depth>16 崩溃
        # (序列化网格 0.01m, depth<=16 -> 坐标范围需 <655m)
        keep = xyz[:, 2] < 100.0
        xyz, rgb, mat = xyz[keep], rgb[keep], mat[keep]

        N = xyz.shape[0]
        if N != self.npoints:
            if N > self.npoints:
                idx = np.random.choice(N, self.npoints, replace=False)
                xyz, rgb, mat = xyz[idx], rgb[idx], mat[idx]
            else:
                pad = self.npoints - N
                xyz = np.pad(xyz, ((0, pad), (0, 0)), mode='constant')
                rgb = np.pad(rgb, ((0, pad), (0, 0)), mode='constant')
                mat = np.pad(mat, (0, pad), mode='constant', constant_values=-1)

        # 训练增强：相机坐标系下不做 yaw 旋转（无意义），只做小幅尺度/抖动扰动
        if self.subset == 'train':
            scale = np.random.uniform(0.9, 1.1)
            xyz = xyz * scale
            jitter = np.random.normal(0, 0.01, size=xyz.shape).astype(np.float32)
            xyz = xyz + jitter

        if self.in_channels == 6:
            pts = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
        else:
            pts = xyz

        pts_tensor = torch.from_numpy(pts).float()
        mat_tensor = torch.from_numpy(mat).long()

        return 'UrmatSeg', self.files[index].stem, (pts_tensor, mat_tensor)
