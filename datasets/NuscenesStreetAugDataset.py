"""
NuscenesStreetAugDataset - nuScenes 标注 + 街道级伪标注 混合语义分割数据集。

用途：自训练(self-training)域适配。street_pc 无真值语义标签，用语义模型伪标注后
（scripts/pseudolabel_street_pc.py，保留 conf>thresh 且非 noise 的点，已是 lidar z-up 系），
按 STREET_RATIO 概率在训练中过采样伪标注样本，与 nuScenes 真值混合训练。

Config 期望:
  DATA_PATH:   nuScenes 真值目录（含 train/val）
  STREET_PATH: street 伪标注目录（含 train/）
  STREET_RATIO: 每个 batch 样本来自 street 的概率（默认 0.3）
  N_POINTS / NUM_CATEGORY / IN_CHANNELS: 同 NuscenesSegDataset
"""

import os, numpy as np, torch
from pathlib import Path
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *


@DATASETS.register_module()
class NuscenesStreetAugDataset(Dataset):
    def __init__(self, config):
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.in_channels = getattr(config, 'IN_CHANNELS', 6)
        self.num_category = getattr(config, 'NUM_CATEGORY', 32)
        self.street_ratio = getattr(config, 'STREET_RATIO', 0.3)

        nu_dir = Path(getattr(config, 'DATA_PATH', 'data/nuscenes_seg_full_16k')) / self.subset
        st_dir = Path(getattr(config, 'STREET_PATH', '')) / self.subset

        if not nu_dir.exists():
            raise FileNotFoundError(f'{nu_dir} not found.')
        self.nu_files = sorted(nu_dir.glob('*.npz'))
        if len(self.nu_files) == 0:
            raise FileNotFoundError(f'{nu_dir} 下没有 .npz 文件')

        self.st_files = sorted(st_dir.glob('*.npz')) if st_dir.exists() else []
        print_log(f'[NuscenesStreetAug] {self.subset}: nu={len(self.nu_files)} '
                  f'street={len(self.st_files)} (ratio={self.street_ratio})',
                  logger='NuscenesStreetAug')

    def __len__(self):
        # 每个 epoch 覆盖全部 nuScenes 一次；street 通过 ratio 在 batch 内过采样
        return len(self.nu_files)

    def _load(self, f):
        d = np.load(str(f))
        xyz = d['xyz'].astype(np.float32)
        rgb = d['rgb'].astype(np.float32)
        sem = d['sem'].astype(np.int64)
        return xyz, rgb, sem

    def _resample(self, xyz, rgb, sem):
        N = xyz.shape[0]
        if N > self.npoints:
            idx = np.random.choice(N, self.npoints, replace=False)
            xyz, rgb, sem = xyz[idx], rgb[idx], sem[idx]
        elif N < self.npoints:
            pad = self.npoints - N
            xyz = np.pad(xyz, ((0, pad), (0, 0)), mode='constant')
            rgb = np.pad(rgb, ((0, pad), (0, 0)), mode='constant')
            sem = np.pad(sem, (0, pad), mode='constant', constant_values=-1)
        return xyz, rgb, sem

    def __getitem__(self, index):
        use_street = (len(self.st_files) > 0) and (np.random.rand() < self.street_ratio)
        if use_street:
            f = self.st_files[np.random.randint(len(self.st_files))]
            xyz, rgb, sem = self._load(f)
            name = 'street_' + f.stem
        else:
            f = self.nu_files[index % len(self.nu_files)]
            xyz, rgb, sem = self._load(f)
            name = 'nu_' + f.stem

        xyz, rgb, sem = self._resample(xyz, rgb, sem)

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
        return 'NuscenesStreetAug', name, (pts_tensor, sem_tensor)
