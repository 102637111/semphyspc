#!/usr/bin/env python3
"""把 nuScenes 3D 物体框投影成 6 相机的 2D bbox（罕见类），作门控投票的 GT 缓存。

nuScenes 的 sample_annotation.json 存的是 **3D bbox**（global 坐标系），
门控投票需要 **2D bbox**（图像坐标系）。本脚本对每个罕见类 3D box：
  global --ego_pose^-1--> ego --cam_extrinsic^-1--> cam --K--> pixel(u,v)
取 8 个角点的 2D 包围盒，对每个可见相机各产生一个 bbox，按 lidar_token 分组保存。

罕见类（long-tail，3D 分割弱 / 2D 检测强）及其 32 类 id：
  pedestrian(2..8), barrier(9), trafficcone(12), bicycle(14), motorcycle(21)

6 相机: CAM_FRONT / FRONT_LEFT / FRONT_RIGHT / BACK / BACK_LEFT / BACK_RIGHT

输出 data/nuscenes_2d_bbox_cache.npz：
  tokens:        (M,)    lidar_token（与 full_16k 的 .npz 文件名对齐）
  offsets:       (M+1,)  该帧的 bbox 区间 = boxes[offsets[i]:offsets[i+1]]
  boxes:         (K,8)   [x, y, w, h, sem_id, depth_min, depth_max, cam_idx]
                        (x,y 左上角, w,h 宽高, sem_id 32 类,
                         depth_min/max 该 3D box 在对应 cam 系的深度范围，
                         cam_idx 索引 CAM_CHANNELS 顺序)
  category_name: (K,)    bbox 对应的 nuScenes 类别名（诊断）
  cam_names:     (6,)    相机名顺序（cam_idx 引用）

用法:
    python scripts/build_2d_bbox_cache.py \
        --dataroot /mnt/data/nuscenes --version v1.0-trainval \
        --seg_dir data/nuscenes_seg_full_16k \
        --out data/nuscenes_2d_bbox_cache.npz \
        --max_samples 100          # 0=全量
"""

import os
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import view_points
    from pyquaternion import Quaternion
    HAS_NUSCENES = True
except ImportError:
    HAS_NUSCENES = False

CAM_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
IMG_W, IMG_H = 1600, 900

# 罕见类：nuScenes 检测类名 -> 32 类分割 id
RARE_CATEGORY_TO_SEM = {
    'human.pedestrian.adult': 2,
    'human.pedestrian.child': 3,
    'human.pedestrian.construction_worker': 4,
    'human.pedestrian.personal_mobility': 5,
    'human.pedestrian.police_officer': 6,
    'human.pedestrian.stroller': 7,
    'human.pedestrian.wheelchair': 8,
    'movable_object.barrier': 9,
    'movable_object.trafficcone': 12,
    'vehicle.bicycle': 14,
    'vehicle.motorcycle': 21,
}


def make_transform(rotation, translation, invert=False):
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = Quaternion(rotation).rotation_matrix
    m[:3, 3] = translation
    return np.linalg.inv(m) if invert else m


def project_box_to_cam(nusc, ann, sample, cam_idx):
    """3D box (sample_annotation) -> 单个相机 2D bbox [x,y,w,h,dmin,dmax] 或 None。"""
    box = Box(center=ann['translation'], size=ann['size'],
              orientation=Quaternion(ann['rotation']))
    corners = box.corners()  # 3x8 global 坐标

    sd_lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', sd_lidar['ego_pose_token'])
    global_to_ego = make_transform(ego_pose['rotation'], ego_pose['translation'], invert=True)

    cam_token = sample['data'][CAM_CHANNELS[cam_idx]]
    sd_cam = nusc.get('sample_data', cam_token)
    cs_cam = nusc.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])
    ego_to_cam = make_transform(cs_cam['rotation'], cs_cam['translation'], invert=True)

    global_to_cam = ego_to_cam @ global_to_ego

    corners_h = np.concatenate([corners.T, np.ones((8, 1))], axis=1)  # 8,4
    pts_cam = (global_to_cam @ corners_h.T).T  # 8,4
    depth = pts_cam[:, 2]
    front = depth > 0
    if not front.any():
        return None
    K = np.array(cs_cam['camera_intrinsic'], dtype=np.float64)
    uv = view_points(pts_cam[front, :3].T, K, normalize=True)  # 3,N
    u, v = uv[0], uv[1]
    x0, y0 = u.min(), v.min()
    x1, y1 = u.max(), v.max()
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(IMG_W - 1, x1), min(IMG_H - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    depth_min = float(depth[front].min())
    depth_max = float(depth[front].max())
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0), depth_min, depth_max]


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

    seg_tokens = set()
    for split in ['train', 'val']:
        d = os.path.join(args.seg_dir, split)
        if os.path.isdir(d):
            seg_tokens.update(f[:-4] for f in os.listdir(d) if f.endswith('.npz'))

    rare_cat = {}
    for cat in nusc.category:
        if cat['name'] in RARE_CATEGORY_TO_SEM:
            rare_cat[cat['token']] = (cat['name'], RARE_CATEGORY_TO_SEM[cat['name']])

    inst2cat = {inst['token']: inst['category_token'] for inst in nusc.instance}

    bbox_by_lidar = defaultdict(list)
    n_kept = 0
    anns = nusc.sample_annotation
    if args.max_samples:
        anns = anns[:args.max_samples * 40]

    for ann in tqdm(anns, desc='projecting 3D boxes to 6-cam 2D'):
        cat = inst2cat.get(ann['instance_token'])
        if cat not in rare_cat:
            continue
        name, sem_id = rare_cat[cat]

        sample = nusc.get('sample', ann['sample_token'])
        lidar_token = sample['data']['LIDAR_TOP']
        if lidar_token not in seg_tokens:
            continue

        for cam_idx in range(len(CAM_CHANNELS)):
            bbox = project_box_to_cam(nusc, ann, sample, cam_idx)
            if bbox is None:
                continue
            bbox_by_lidar[lidar_token].append(bbox + [sem_id, cam_idx, name])
            n_kept += 1

    lidar_tokens = sorted(bbox_by_lidar)
    offsets = np.zeros(len(lidar_tokens) + 1, dtype=np.int64)
    boxes_list, names_list = [], []
    for i, tok in enumerate(lidar_tokens):
        items = bbox_by_lidar[tok]
        offsets[i + 1] = offsets[i] + len(items)
        for (x, y, w, h, dmin, dmax, sem_id, cam_idx, name) in items:
            boxes_list.append([x, y, w, h, sem_id, dmin, dmax, cam_idx])
            names_list.append(name)

    boxes = np.array(boxes_list, dtype=np.float32) if boxes_list else np.zeros((0, 8), np.float32)
    names = np.array(names_list)
    tokens = np.array(lidar_tokens)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(args.out, tokens=tokens, offsets=offsets,
                        boxes=boxes, category_name=names,
                        cam_names=np.array(CAM_CHANNELS))
    print(f'\n保存 -> {args.out}', flush=True)
    print(f'帧数={len(tokens)}  罕见类 bbox 总数={n_kept}', flush=True)
    if len(boxes):
        from collections import Counter
        cnt = Counter(boxes[:, 4].astype(int))
        print('各类别 bbox 数 (sem_id: count):', flush=True)
        for sem_id in sorted(cnt):
            print(f'  {sem_id:>2}: {cnt[sem_id]:,}', flush=True)
        # 各相机 bbox 分布
        cam_cnt = Counter(boxes[:, 7].astype(int))
        print('各相机 bbox 数 (cam_idx: count):', flush=True)
        for ci in sorted(cam_cnt):
            print(f'  {ci} {CAM_CHANNELS[ci]:<16}: {cam_cnt[ci]:,}', flush=True)


if __name__ == '__main__':
    main()
