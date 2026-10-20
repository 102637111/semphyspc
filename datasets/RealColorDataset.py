"""
RealColorDataset - loads real colored point clouds from .npz files.

Compatible with output from:
  - scripts/build_rtabmap_dataset.py  (RTAB-Map / Orbbec PCD → .npz)
  - scripts/precompute_rgb_dataset.py  (ModelNet40 → .npz)

Config expects:
  DATA_PATH: directory containing train.npz and test.npz
  N_POINTS: number of points
  NUM_CATEGORY: number of classes (auto-detected if not specified)
"""

import os, numpy as np, torch
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *


@DATASETS.register_module()
class RealColorDataset(Dataset):
    """Generic real colored point cloud dataset."""

    def __init__(self, config):
        self.npoints = config.N_POINTS
        self.subset = config.subset
        data_dir = getattr(config, 'DATA_PATH', 'data/my_real_dataset')

        npz_path = os.path.join(data_dir, f'{self.subset}.npz')
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f'{npz_path} not found. Run build_rtabmap_dataset.py first.')

        print_log(f'[RealColorDataset] Loading {npz_path}...', logger='RealColor')
        data = np.load(npz_path)
        self.points = data['points']
        self.labels = data['labels']

        # Auto-detect num classes
        n_classes = getattr(config, 'NUM_CATEGORY', None)
        if n_classes is None:
            n_classes = int(self.labels.max()) + 1
        self.num_category = n_classes

        print_log(f'[RealColorDataset] {self.subset}: {len(self)} samples, '
                  f'{n_classes} classes, shape={self.points.shape}',
                  logger='RealColor')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        pts = self.points[index].copy()
        if self.subset == 'train':
            pts = pts[np.random.permutation(pts.shape[0])]
        pts_tensor = torch.from_numpy(pts).float()
        label = int(self.labels[index])
        return 'RealColor', 'sample', (pts_tensor, label)
