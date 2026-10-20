#!/usr/bin/env python3
"""SemanticKITTI -> 12 类逐点语义分割 npz（nuScenes 之外的第二真实驾驶数据集）。

用途：
  作为 nuScenes 之外的第二个真实室外驾驶数据集，用于
  ① 跨数据集预训练 -> nuScenes 微调（迁移学习，救 12 类体系的样本量与泛化）；
  ② 验证 XYZ+RGB 早期融合 Mamba 的跨域一致性（论文加分点，呼应 rgb-channel-validity）。

输出格式与 NuscenesSegDataset 完全一致（可直接喂同一个 dataset/模型）：
  {xyz (N,3) float32, rgb (N,3) float32 [0,1], sem (N,) int16}
  sem 为 12 类 COARSE id（见 class-merge-coarse），-1 = 忽略（unlabeled/outlier）。

----------------------------- 数据下载（链接 2026-09-21 实测有效） -----------------------------
两类数据分两处下（语义标签不在 KITTI 官网）：

  A. KITTI Odometry 原始数据（cvlibs.net，需注册邮箱后下载）：
     - 点云  (~80 GB)  http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip
                      -> sequences/XX/velodyne/*.bin
     - 标定  (~1 MB)   http://www.cvlibs.net/download.php?file=data_odometry_calib.zip
                      -> sequences/XX/calib.txt
     - 图像  (~65 GB, 可选，仅 --rgb_mode project 要真 RGB)
                      http://www.cvlibs.net/download.php?file=data_odometry_color.zip
                      -> sequences/XX/image_2/*.png
  B. SemanticKITTI 语义标签（semantic-kitti.org，核心，只有这里有 .label）：
     - 标签  (~179 MB) http://semantic-kitti.org/assets/data_odometry_labels.zip
                      -> sequences/XX/labels/*.label
  官网下载页：http://www.semantic-kitti.org/dataset.html （注：不是 dataset.php，后者已 404）

目录结构（解压后）：
  dataset/sequences/
    00/ velodyne/*.bin   labels/*.label   calib.txt   image_2/*.png
    01/ ...   ...  （共 22 个序列 00-21）

序列划分（官方）：
  train = 00,01,02,03,04,05,06,07,09,10
  val   = 08
  test  = 11-21（无公开标签，本脚本跳过）

RGB 模式（--rgb_mode）：
  intensity : 用 LiDAR intensity 归一化后复制 3 通道（默认，零额外下载）
  project   : 用 calib + image_2 把点投影到相机取真 RGB（需下载 color 数据）
  none      : RGB 全 0（纯 XYZ 几何，消融用）

用法:
    # 全量 train（intensity 兜底，无需相机图像）
    python scripts/build_semantickitti_dataset.py \
        --src /mnt/data/datasets/SemanticKITTI --dst /mnt/data/datasets/semantickitti_seg_16k \
        --split train --n_points 16384 --rgb_mode intensity

    # 真 RGB 投影（需先下载 color 数据）
    python scripts/build_semantickitti_dataset.py \
        --src /mnt/data/datasets/SemanticKITTI --dst /mnt/data/datasets/semantickitti_seg_16k \
        --split train --n_points 16384 --rgb_mode project

    # 冒烟（每序列只取前 50 帧）
    python scripts/build_semantickitti_dataset.py --src ... --dst ... --max_samples 50
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# nuScenes 12 类 COARSE id（与 class-merge-coarse 一致）
COARSE_12 = ['noise', 'animal', 'person', 'car', 'bus', 'truck',
             'two_wheeler', 'ground', 'manmade', 'vegetation', 'frangible', 'ego']

# SemanticKITTI 原始标签(0-33) -> 12 类 COARSE id（-1 = 忽略）。
# 注意：不经官方 learning_map（官方把 bus 并入 other-vehicle），
# 直接用原始标签，使 bus(13) 落在 12 类的 bus(4)，与 nuScenes 口径一致。
# 语义依据：bicyclist/motorcyclist = 骑车/骑摩托的「人」-> person(致命)；
#           fence/pole/traffic-sign -> manmade(硬刚性)；trunk -> vegetation(可变形)。
RAW_TO_COARSE = {
    0: -1, 1: -1,               # unlabeled, outlier
    10: 3,                       # car
    11: 6,                       # bicycle -> two_wheeler
    13: 4,                       # bus -> bus
    15: 6,                       # motorcycle -> two_wheeler
    16: 5,                       # on-rails -> truck(重型)
    18: 5,                       # truck
    20: 5,                       # other-vehicle -> truck
    30: 2, 31: 2, 32: 2,         # person / bicyclist / motorcyclist -> person
    40: 7, 44: 7, 48: 7, 49: 7,  # road / parking / sidewalk / other-ground -> ground
    50: 8, 51: 8, 52: 8,         # building / fence / other-structure -> manmade
    60: 7,                       # lane-marking -> ground
    70: 9, 71: 9,                # vegetation / trunk -> vegetation
    72: 7,                       # terrain -> ground
    80: 8, 81: 8,                # pole / traffic-sign -> manmade
    99: -1,                      # other-object
    252: 3,                      # moving-car -> car
    253: 2, 254: 2, 255: 2,      # moving-bicyclist/person/motorcyclist -> person
    256: 5, 257: 4, 258: 5, 259: 5,  # moving on-rails/bus/truck/other-vehicle
}

# 官方序列划分（test 序列无公开标签，跳过）
SEQ_TRAIN = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
SEQ_VAL = ['08']


def read_calib(calib_path):
    """读 calib.txt，返回 P2(3x4), R0_rect(3x3), Tr_velo_to_cam(3x4)。"""
    data = {}
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, val = line.split(':', 1)
            data[key] = np.array([float(x) for x in val.split()], dtype=np.float64)
    P2 = data['P2'].reshape(3, 4)
    R0 = data['R0_rect'].reshape(3, 3)
    Tr = data['Tr_velo_to_cam'].reshape(3, 4)
    return P2, R0, Tr


def project_rgb(pts, calib_path, img_path):
    """把 velodyne 点投影到 cam2 彩色图取 RGB（[0,1]），不可见点返回 0。

    pts: (N,3) float32 velodyne 坐标。
    """
    from PIL import Image

    P2, R0, Tr = read_calib(calib_path)
    N = len(pts)
    rgb = np.zeros((N, 3), dtype=np.float32)

    # velo -> cam0 -> rectified -> cam2 像素
    xyz_h = np.concatenate([pts.astype(np.float64),
                            np.ones((N, 1), dtype=np.float64)], axis=1).T  # (4,N)
    cam = Tr @ xyz_h                        # (3,N) cam0
    rect = R0 @ cam                         # (3,N) rectified cam0
    uv = P2 @ np.vstack([rect, np.ones((1, N))])  # (3,N)
    z = uv[2]
    valid = z > 0.5                         # 相机前方
    u = (uv[0] / z).astype(np.int64)
    v = (uv[1] / z).astype(np.int64)

    img = np.asarray(Image.open(img_path))   # (H, W, 3) uint8
    H, W = img.shape[:2]
    in_img = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if in_img.any():
        rgb[in_img] = img[v[in_img], u[in_img]][:, :3].astype(np.float32) / 255.0
    return rgb


def load_frame(seq_dir, frame_id, rgb_mode):
    """读一帧，返回 xyz(N,3), rgb(N,3), sem(N,)（12 类，-1 忽略）。"""
    velo = np.fromfile(str(seq_dir / 'velodyne' / f'{frame_id}.bin'),
                       dtype=np.float32).reshape(-1, 4)
    xyz = velo[:, :3].astype(np.float32)
    intensity = velo[:, 3].astype(np.float32)

    label = np.fromfile(str(seq_dir / 'labels' / f'{frame_id}.label'),
                        dtype=np.uint32)
    sem_raw = (label & 0xFFFF).astype(np.int64)

    # 原始标签 -> 12 类
    sem = np.array([RAW_TO_COARSE.get(int(r), -1) for r in sem_raw], dtype=np.int16)

    # RGB
    if rgb_mode == 'project':
        rgb = project_rgb(xyz, str(seq_dir / 'calib.txt'),
                          str(seq_dir / 'image_2' / f'{frame_id}.png'))
    elif rgb_mode == 'intensity':
        # intensity 归一化到 [0,1] 复制 3 通道（保持 6 通道结构，第 4-6 通道=反射率）
        i = np.clip(intensity, 0, None)
        imax = i.max() if i.size else 1.0
        i = (i / max(imax, 1e-6)).astype(np.float32)
        rgb = np.stack([i, i, i], axis=1)
    else:  # none
        rgb = np.zeros((len(xyz), 3), dtype=np.float32)

    return xyz, rgb, sem


def sample_n(xyz, rgb, sem, n_points, rng):
    """随机下采样到固定 n_points（SemanticKITTI 每帧 ~12 万点，远超 16k）。"""
    N = len(xyz)
    if N == n_points:
        return xyz, rgb, sem
    if N > n_points:
        idx = rng.choice(N, n_points, replace=False)
        return xyz[idx], rgb[idx], sem[idx]
    # 不足则补零（sem 补 -1 忽略）
    pad = n_points - N
    xyz = np.pad(xyz, ((0, pad), (0, 0)))
    rgb = np.pad(rgb, ((0, pad), (0, 0)))
    sem = np.pad(sem, (0, pad), constant_values=-1)
    return xyz, rgb, sem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='SemanticKITTI 根目录（含 sequences/）')
    ap.add_argument('--dst', required=True, help='输出 npz 目录')
    ap.add_argument('--split', choices=['train', 'val'], default='train')
    ap.add_argument('--n_points', type=int, default=16384)
    ap.add_argument('--rgb_mode', choices=['intensity', 'project', 'none'],
                    default='intensity')
    ap.add_argument('--max_samples', type=int, default=None, help='每序列最多转多少帧（冒烟用）')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    src = Path(args.src)
    seqs = SEQ_TRAIN if args.split == 'train' else SEQ_VAL
    rng = np.random.default_rng(args.seed)

    # 校验 project 模式依赖
    if args.rgb_mode == 'project':
        probe = src / 'sequences' / seqs[0]
        if not (probe / 'image_2').exists() or not (probe / 'calib.txt').exists():
            print('[warn] --rgb_mode project 需要 image_2/ 与 calib.txt，'
                  '请先下载 color 数据；若缺失会报错。')

    out_dir = Path(args.dst) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    kept_classes = {}
    for seq in seqs:
        seq_dir = src / 'sequences' / seq
        velo_dir = seq_dir / 'velodyne'
        if not velo_dir.exists():
            print(f'[skip] {seq_dir} 不存在（未下载？）')
            continue
        frames = sorted(p.stem for p in velo_dir.glob('*.bin'))
        if args.max_samples:
            frames = frames[:args.max_samples]

        for fid in tqdm(frames, desc=f'seq {seq}'):
            try:
                xyz, rgb, sem = load_frame(seq_dir, fid, args.rgb_mode)
            except FileNotFoundError as e:
                print(f'[warn] {seq}/{fid} 缺文件：{e}')
                continue
            xyz, rgb, sem = sample_n(xyz, rgb, sem, args.n_points, rng)
            # 统计有效类分布（忽略 -1）
            vals, cnts = np.unique(sem[sem >= 0], return_counts=True)
            for v, c in zip(vals.tolist(), cnts.tolist()):
                kept_classes[v] = kept_classes.get(v, 0) + int(c)
            np.savez_compressed(str(out_dir / f'{seq}_{fid}.npz'),
                                xyz=xyz, rgb=rgb, sem=sem)
            total += 1

    print(f'\n[完成] {args.split} 共 {total} 帧 -> {out_dir}')
    print('[有效类分布（12 类 COARSE，排除 -1 忽略）]：')
    for c in range(12):
        if c in kept_classes:
            print(f'  {c:>2} {COARSE_12[c]:<12} {kept_classes[c]:>12,} 点')


if __name__ == '__main__':
    main()
