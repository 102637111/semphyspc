#!/usr/bin/env python3
"""
从 nuScenes 构建逐点语义分割数据集（LiDAR + 相机RGB + panoptic 标签）。

流程:
  1. 加载 LIDAR_TOP 关键帧点云 (xyz)
  2. 用 6 个相机投影给点云上色 (rgb)
  3. 加载 panoptic 逐点标签 (sem = label//1000, inst = label%1000)
  4. FPS 下采样到固定点数
  5. 保存为 per-sample .npz: {xyz, rgb, sem, inst}

用法:
    # mini 冒烟测试
    python scripts/build_nuscenes_seg_dataset.py \
        --dataroot ./data --version v1.0-mini \
        --panoptic_dir /tmp/panop/nuScenes-panoptic-v1.0-all \
        --output_dir ./data/nuscenes_seg_mini --num_points 4096 --max_samples 50

    # 全量 trainval
    python scripts/build_nuscenes_seg_dataset.py \
        --dataroot /mnt/data/nuscenes --version v1.0-trainval \
        --panoptic_dir /path/to/nuScenes-panoptic-v1.0-all \
        --output_dir /mnt/data/nuscenes_seg --num_points 4096
"""

import os, sys, argparse, json, numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    HAS_NUSCENES = True
except ImportError:
    HAS_NUSCENES = False
    print("Warning: nuscenes-devkit not installed. pip install nuscenes-devkit")

# 6 个相机通道
CAM_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def color_lidar_by_camera(xyz, nusc, lidar_token, cam_channel='CAM_FRONT'):
    """把 LiDAR 点云投影到指定相机，给可见点赋予 RGB（[0,1]）。

    Returns:
        rgb:   (N,3) float32
        valid: (N,) bool
    """
    from PIL import Image
    from pyquaternion import Quaternion

    N = len(xyz)
    rgb = np.zeros((N, 3), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)

    sd_lidar = nusc.get('sample_data', lidar_token)
    sample = nusc.get('sample', sd_lidar['sample_token'])
    if cam_channel not in sample['data']:
        return rgb, valid
    sd_cam = nusc.get('sample_data', sample['data'][cam_channel])
    cs_lidar = nusc.get('calibrated_sensor', sd_lidar['calibrated_sensor_token'])
    cs_cam = nusc.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])

    # LiDAR -> ego -> cam（关键帧 LiDAR 与相机同步，略去 global/ego_pose 往返）
    lidar_to_ego = np.eye(4, dtype=np.float64)
    lidar_to_ego[:3, :3] = Quaternion(cs_lidar['rotation']).rotation_matrix
    lidar_to_ego[:3, 3] = cs_lidar['translation']
    cam_to_ego = np.eye(4, dtype=np.float64)
    cam_to_ego[:3, :3] = Quaternion(cs_cam['rotation']).rotation_matrix
    cam_to_ego[:3, 3] = cs_cam['translation']
    T = np.linalg.inv(cam_to_ego) @ lidar_to_ego          # LiDAR -> cam

    # 投影到像素
    xyz_h = np.concatenate([xyz.astype(np.float64),
                            np.ones((N, 1), dtype=np.float64)], axis=1).T  # (4,N)
    cam = T @ xyz_h                                        # (3,N)
    z = cam[2]
    K = np.array(cs_cam['camera_intrinsic'], dtype=np.float64)
    u = K[0, 0] * cam[0] / z + K[0, 2]
    v = K[1, 1] * cam[1] / z + K[1, 2]

    img_path = os.path.join(nusc.dataroot, sd_cam['filename'])
    if not os.path.exists(img_path):
        return rgb, valid
    img = np.array(Image.open(img_path))
    H, W = img.shape[:2]

    in_front = z > 0.5
    ui = np.floor(u).astype(np.int64)
    vi = np.floor(v).astype(np.int64)
    in_img = in_front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    rgb[in_img] = img[vi[in_img], ui[in_img]][:, :3].astype(np.float32) / 255.0
    valid = in_img
    return rgb, valid


def color_lidar_multi_cam(xyz, nusc, lidar_token):
    """用 6 个相机给 LiDAR 点云上色，合并所有可见点。

    Returns: rgb (N,3) in [0,1], valid (N,) bool
    """
    N = len(xyz)
    rgb_all = np.zeros((N, 3), dtype=np.float32)
    valid_all = np.zeros(N, dtype=bool)
    for cam in CAM_CHANNELS:
        try:
            c, v = color_lidar_by_camera(xyz, nusc, lidar_token, cam_channel=cam)
            rgb_all[v] = c[v]
            valid_all |= v
        except Exception:
            continue
    return rgb_all, valid_all


def load_panoptic_label(panoptic_dir, version, lidar_token):
    """加载某帧的 panoptic 逐点标签，返回 (sem, inst) 均为 (N,) int16。"""
    path = Path(panoptic_dir) / 'panoptic' / version / f'{lidar_token}_panoptic.npz'
    if not path.exists():
        return None, None
    label = np.load(str(path))['data'].astype(np.int32)
    sem = (label // 1000).astype(np.int16)
    inst = (label % 1000).astype(np.int16)
    return sem, inst


def fps_downsample(points, labels, num_points):
    """GPU FPS 下采样 points (N,C) 和 labels (N,*)，返回对齐后的子集。

    points 前 3 列必须是 XYZ。
    """
    import torch
    from pointnet2_ops import pointnet2_utils

    N = points.shape[0]
    if N <= num_points:
        return points, labels

    xyz = torch.from_numpy(points[:, :3].astype(np.float32)).cuda().unsqueeze(0)  # 1 N 3
    idx = pointnet2_utils.furthest_point_sample(xyz, num_points)  # 1 num_points
    idx = idx.squeeze(0).cpu().numpy()  # num_points
    return points[idx], labels[idx]


def build_dataset(args):
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    output_dir = Path(args.output_dir)
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # 官方 train/val 划分（nuscenes.utils.splits，700 train + 150 val）
    name2token = {s['name']: s['token'] for s in nusc.scene}
    split_path = args.split_file
    if split_path and os.path.exists(split_path):
        sp = json.load(open(split_path))
        val_scene_tokens = {name2token[n] for n in sp['val'] if n in name2token}
        print(f"使用官方划分: val={len(val_scene_tokens)} scenes (来自 {split_path})")
    else:
        scenes = nusc.scene
        n_val_scenes = max(1, int(len(scenes) * args.val_ratio))
        val_scene_tokens = {s['token'] for s in scenes[-n_val_scenes:]}  # 回退: 最后15%

    counts = defaultdict(int)
    sem_class_count = defaultdict(int)

    for sample in tqdm(nusc.sample, desc='Building seg dataset'):
        if args.max_samples and counts['total'] >= args.max_samples:
            break

        scene_token = sample['scene_token']
        split = 'val' if scene_token in val_scene_tokens else 'train'

        lidar_token = sample['data']['LIDAR_TOP']
        sd_record = nusc.get('sample_data', lidar_token)

        # 1. LiDAR 点云
        pcl_path = os.path.join(args.dataroot, sd_record['filename'])
        try:
            pc = LidarPointCloud.from_file(pcl_path)
        except Exception:
            continue
        xyz = pc.points[:3, :].T.astype(np.float32)  # N 3

        # 2. 相机上色
        rgb, color_valid = color_lidar_multi_cam(xyz, nusc, lidar_token)

        # 3. panoptic 标签
        sem, inst = load_panoptic_label(args.panoptic_dir, args.version, lidar_token)
        if sem is None:
            continue
        if len(sem) != len(xyz):
            # 标签与点云点数不一致（极少数情况），截断/补齐到 min
            n = min(len(sem), len(xyz))
            sem, inst, xyz, rgb = sem[:n], inst[:n], xyz[:n], rgb[:n]

        # 4. 合并 + FPS 下采样
        points = np.concatenate([xyz, rgb], axis=1).astype(np.float32)  # N 6
        labels = np.stack([sem, inst], axis=1).astype(np.int16)  # N 2
        points, labels = fps_downsample(points, labels, args.num_points)

        # 5. 保存
        sem_ds = labels[:, 0]
        inst_ds = labels[:, 1]
        out = {
            'xyz': points[:, :3].astype(np.float32),
            'rgb': points[:, 3:6].astype(np.float32),
            'sem': sem_ds.astype(np.int16),
            'inst': inst_ds.astype(np.int16),
        }
        out_path = (val_dir if split == 'val' else train_dir) / f'{lidar_token}.npz'
        np.savez_compressed(str(out_path), **out)

        counts[split] += 1
        counts['total'] += 1
        for s in sem_ds:
            sem_class_count[int(s)] += 1

    print(f"\n完成: train={counts['train']}, val={counts['val']}")
    print("各类别点数统计 (sem_class: count):")
    for c in sorted(sem_class_count):
        print(f"  {c:>3}: {sem_class_count[c]:,}")


def main():
    parser = argparse.ArgumentParser(description='Build nuScenes per-point seg dataset')
    parser.add_argument('--dataroot', required=True, help='nuScenes dataroot (含 <version>/ 与 samples/)')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--panoptic_dir', required=True, help='解压后的 nuScenes-panoptic-v1.0-all 目录')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_points', type=int, default=4096)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--split_file', type=str, default='data/nuscenes_official_split.json',
                        help='官方 train/val scene 划分 json（{train:[names], val:[names]}），不存在则回退 val_ratio')
    parser.add_argument('--max_samples', type=int, default=0, help='0=全部，>0=只处理前 N 帧（冒烟测试）')
    args = parser.parse_args()

    if not HAS_NUSCENES:
        print("ERROR: pip install nuscenes-devkit")
        sys.exit(1)

    build_dataset(args)
    print(f"\nDone! 数据保存到 {args.output_dir}")


if __name__ == '__main__':
    main()
