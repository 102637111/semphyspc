#!/usr/bin/env python3
"""构建 nuScenes LiDAR→6 相机投影索引，供 2D/3D 门控投票使用。

对 full_16k 数据集的每个样本（文件名 = lidar_token），反查 nuScenes 数据库，
计算并保存 6 个相机（CAM_FRONT / FRONT_LEFT / FRONT_RIGHT / BACK / BACK_LEFT /
BACK_RIGHT）各自的：
  - K:             相机内参 (6,3,3)
  - T_lidar_cam:   LiDAR → 该相机外参 (6,4,4)（cam <- lidar 变换）
  - cam_names:     (6,) 相机名顺序（bbox 缓存的 cam_idx 引用此顺序）

3D 点 xyz 已在 full_16k 的 .npz 里，门控投票时用 K、T 现场投影即可。

投影链:  lidar_frame --lidar_to_ego--> ego --ego_to_cam--> cam --K--> pixel(u,v)

用法:
    python scripts/build_projection_index.py \
        --dataroot /mnt/data/nuscenes --version v1.0-trainval \
        --seg_dir data/nuscenes_seg_full_16k \
        --out data/nuscenes_projection_index.npz \
        --max_samples 100          # 先小样本验证；0 = 全量
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

try:
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion
    HAS_NUSCENES = True
except ImportError:
    HAS_NUSCENES = False

CAM_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
IMG_W, IMG_H = 1600, 900  # nuScenes 相机分辨率（6 相机一致）


def get_projection_multi(nusc, lidar_token):
    """返回 (K, T_lidar_cam) 均为 (6,3,3) / (6,4,4)，对应 CAM_CHANNELS 顺序。"""
    sd_lidar = nusc.get('sample_data', lidar_token)
    cs_lidar = nusc.get('calibrated_sensor', sd_lidar['calibrated_sensor_token'])
    lidar_to_ego = np.eye(4, dtype=np.float64)
    lidar_to_ego[:3, :3] = Quaternion(cs_lidar['rotation']).rotation_matrix
    lidar_to_ego[:3, 3] = cs_lidar['translation']

    sample = nusc.get('sample', sd_lidar['sample_token'])
    Ks, Ts = [], []
    for ch in CAM_CHANNELS:
        cam_token = sample['data'][ch]
        sd_cam = nusc.get('sample_data', cam_token)
        cs_cam = nusc.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])
        cam_to_ego = np.eye(4, dtype=np.float64)
        cam_to_ego[:3, :3] = Quaternion(cs_cam['rotation']).rotation_matrix
        cam_to_ego[:3, 3] = cs_cam['translation']
        ego_to_cam = np.linalg.inv(cam_to_ego)
        T = ego_to_cam @ lidar_to_ego
        K = np.array(cs_cam['camera_intrinsic'], dtype=np.float64)  # 3x3
        Ks.append(K); Ts.append(T)
    return np.array(Ks), np.array(Ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', required=True)
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--seg_dir', required=True, help='full_16k 目录（含 train/、val/）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_samples', type=int, default=0, help='0=全量')
    args = ap.parse_args()

    if not HAS_NUSCENES:
        print('ERROR: pip install nuscenes-devkit pyquaternion')
        return

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # 收集所有 npz 文件名（= lidar_token）
    tokens = set()
    for split in ['train', 'val']:
        d = os.path.join(args.seg_dir, split)
        if os.path.isdir(d):
            tokens.update(f[:-4] for f in os.listdir(d) if f.endswith('.npz'))
    tokens = sorted(tokens)
    if args.max_samples:
        tokens = tokens[:args.max_samples]
    print(f'共 {len(tokens)} 个样本构建投影索引', flush=True)

    Ks, Ts = [], []
    skip = 0
    for tok in tqdm(tokens, desc='building 6-cam projection index'):
        try:
            K, T = get_projection_multi(nusc, tok)
        except Exception:
            skip += 1
            continue
        Ks.append(K); Ts.append(T)

    Ks = np.array(Ks, dtype=np.float32)          # M,6,3,3
    Ts = np.array(Ts, dtype=np.float32)          # M,6,4,4
    tokens = np.array(tokens[:len(Ks)])          # 对齐（去掉 skip 的）

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(args.out, tokens=tokens, K=Ks, T=Ts,
                        cam_names=np.array(CAM_CHANNELS))
    print(f'\n保存 -> {args.out}  (tokens={len(tokens)}, skip={skip})', flush=True)
    print(f'相机顺序: {CAM_CHANNELS}', flush=True)


if __name__ == '__main__':
    main()
