#!/usr/bin/env python3
"""S3 第 1 步：用 nuScenes **train split** 的 3D 真值框投影出 6 相机的 2D 标签，
供微调 RT-DETR 使用。

动机（§4.13）：COCO 零样本检测器只覆盖 person / bicycle / motorcycle，
barrier 与 trafficcone 无对应类，且召回仅 0.205（oracle 0.861）。
在**训练集**上用真值标注做有监督微调是标准做法，不构成泄漏——
泄漏只发生在**评测时**使用评测集的真值（那正是 S1/oracle 设置，本文已显式降级标注）。

类别表：沿用 COCO 的 80 类 id，在末尾**追加** barrier=80、trafficcone=81。
这样微调时前 80 行的权重可完整继承 COCO 预训练，只有两行是新初始化的，
同时 car/bus/truck 等 COCO 类继续被监督（提供背景判别力，避免把车误判成护栏）。

输出 data/s3_det_labels_train.npz：
  tokens:  (M,)    lidar_token（与该帧 6 张相机图对应）
  offsets: (M+1,)
  boxes:   (K,9)   [x, y, w, h, cls_id, depth_min, depth_max, cam_idx, n_lidar_pts]
  cls_names: (82,) 类别名（诊断）

用法:
    python3 scripts/build_s3_det_labels.py --dataroot /mnt/data/nuscenes \
        --seg_dir data/nuscenes_seg_full --split train --out data/s3_det_labels_train.npz
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

# nuScenes lidarseg 32 类 id -> COCO 80 类 id（80/81 为新增的 barrier / trafficcone）
SEM_TO_COCO = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,   # 7 个行人子类 -> person
               14: 1,                                       # bicycle
               17: 2,                                       # car
               21: 3,                                       # motorcycle
               15: 5, 16: 5,                                # bus.bendy / bus.rigid -> bus
               23: 7,                                       # truck
               9: 80,                                       # barrier（新增）
               12: 81}                                      # trafficcone（新增）
NUM_CLASSES = 82
MIN_BOX_PX = 2.0        # 丢弃投影后过小的框（1~2 像素的框对训练只有噪声）


def cls_name(cid):
    coco = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    return coco.get(cid, {80: 'barrier', 81: 'trafficcone'}.get(cid, f'id{cid}'))


def make_transform(rotation, translation, invert=False):
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = Quaternion(rotation).rotation_matrix
    m[:3, 3] = translation
    return np.linalg.inv(m) if invert else m


def project_box_to_cam(nusc, ann, sample, cam_idx, cache):
    """3D box -> 单相机 2D bbox [x,y,w,h,dmin,dmax]，不可见返回 None。"""
    box = Box(center=ann['translation'], size=ann['size'],
              orientation=Quaternion(ann['rotation']))
    corners = box.corners()

    key = ('lidar_pose', sample['data']['LIDAR_TOP'])
    if key not in cache:
        sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ep = nusc.get('ego_pose', sd['ego_pose_token'])
        cache[key] = make_transform(ep['rotation'], ep['translation'], invert=True)
    global_to_ego = cache[key]

    cam_token = sample['data'][CAM_CHANNELS[cam_idx]]
    key = ('cam', cam_token)
    if key not in cache:
        sd_cam = nusc.get('sample_data', cam_token)
        cs = nusc.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])
        cache[key] = (make_transform(cs['rotation'], cs['translation'], invert=True),
                      np.array(cs['camera_intrinsic'], dtype=np.float64))
    ego_to_cam, K = cache[key]

    global_to_cam = ego_to_cam @ global_to_ego
    corners_h = np.concatenate([corners.T, np.ones((8, 1))], axis=1)
    pts_cam = (global_to_cam @ corners_h.T).T
    depth = pts_cam[:, 2]
    front = depth > 0
    if not front.any():
        return None
    uv = view_points(pts_cam[front, :3].T, K, normalize=True)
    u, v = uv[0], uv[1]
    x0, y0 = max(0.0, u.min()), max(0.0, v.min())
    x1, y1 = min(IMG_W - 1.0, u.max()), min(IMG_H - 1.0, v.max())
    w, h = x1 - x0, y1 - y0
    if w < MIN_BOX_PX or h < MIN_BOX_PX:
        return None
    return [float(x0), float(y0), float(w), float(h),
            float(depth[front].min()), float(depth[front].max())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', required=True)
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--seg_dir', required=True, help='分割数据集目录（含 train/、val/）')
    ap.add_argument('--split', default='train', choices=['train', 'val'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--min_lidar_pts', type=int, default=1,
                    help='物体框内最少 LiDAR 点数；0 表示不过滤（被完全遮挡的物体标注不可靠）')
    ap.add_argument('--max_samples', type=int, default=0, help='0=全量（调试用）')
    args = ap.parse_args()

    if not HAS_NUSCENES:
        raise SystemExit('ERROR: pip install nuscenes-devkit pyquaternion')

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    split_dir = os.path.join(args.seg_dir, args.split)
    wanted = {f[:-4] for f in os.listdir(split_dir) if f.endswith('.npz')}
    print(f'[s3] split={args.split}  目标帧数 {len(wanted)}', flush=True)

    # 只保留类别映射表里有的 instance
    inst2cat = {i['token']: i['category_token'] for i in nusc.instance}
    cat2sem = {}
    for c in nusc.category:
        cat2sem[c['token']] = c['name']
    NAME_TO_SEM = {
        'human.pedestrian.adult': 2, 'human.pedestrian.child': 3,
        'human.pedestrian.construction_worker': 4, 'human.pedestrian.personal_mobility': 5,
        'human.pedestrian.police_officer': 6, 'human.pedestrian.stroller': 7,
        'human.pedestrian.wheelchair': 8, 'movable_object.barrier': 9,
        'movable_object.trafficcone': 12, 'vehicle.bicycle': 14,
        'vehicle.bus.bendy': 15, 'vehicle.bus.rigid': 16, 'vehicle.car': 17,
        'vehicle.motorcycle': 21, 'vehicle.truck': 23,
    }

    bbox_by_lidar = defaultdict(list)
    cache = {}
    n_skip_pts = 0
    anns = nusc.sample_annotation
    if args.max_samples:
        anns = anns[:args.max_samples * 60]

    for ann in tqdm(anns, desc=f'projecting 3D->2D ({args.split})'):
        name = cat2sem.get(inst2cat.get(ann['instance_token']))
        sem = NAME_TO_SEM.get(name)
        if sem is None:
            continue
        if ann['num_lidar_pts'] < args.min_lidar_pts:
            n_skip_pts += 1
            continue

        sample = nusc.get('sample', ann['sample_token'])
        lidar_token = sample['data']['LIDAR_TOP']
        if lidar_token not in wanted:
            continue

        cid = SEM_TO_COCO[sem]
        for cam_idx in range(len(CAM_CHANNELS)):
            bb = project_box_to_cam(nusc, ann, sample, cam_idx, cache)
            if bb is None:
                continue
            x, y, w, h, dmin, dmax = bb
            bbox_by_lidar[lidar_token].append(
                [x, y, w, h, cid, dmin, dmax, cam_idx, ann['num_lidar_pts']])

    tokens = sorted(bbox_by_lidar)
    offsets = np.zeros(len(tokens) + 1, dtype=np.int64)
    rows = []
    for i, tok in enumerate(tokens):
        items = bbox_by_lidar[tok]
        offsets[i + 1] = offsets[i] + len(items)
        rows.extend(items)
    boxes = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 9), np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, tokens=np.array(tokens), offsets=offsets, boxes=boxes,
                        cls_names=np.array([cls_name(i) for i in range(NUM_CLASSES)]))

    print(f'\n[s3] 输出 {args.out}')
    print(f'     帧数 {len(tokens):,}   框数 {len(boxes):,}   '
          f'（因框内 LiDAR 点 <{args.min_lidar_pts} 跳过 {n_skip_pts:,} 个标注）')
    if len(boxes):
        u, c = np.unique(boxes[:, 4].astype(int), return_counts=True)
        print('     每类框数：')
        for uu, cc in zip(u, c):
            print(f'       {uu:>3} {cls_name(uu):<14}{cc:>10,}')
        per_cam = np.bincount(boxes[:, 7].astype(int), minlength=6)
        print('     每相机框数：', per_cam.tolist())


if __name__ == '__main__':
    main()
