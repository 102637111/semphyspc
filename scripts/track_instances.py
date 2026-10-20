#!/usr/bin/env python3
"""
跨帧实例跟踪：对同一 nuScenes 场景的连续关键帧做实例级聚类，用匈牙利匹配
保持物体跨帧稳定 ID，并融合静态物体点云成地图级物体。

流程：
  1. nuScenes-devkit 取 scene → 连续 sample → LIDAR_TOP token（每帧 ~0.5s）。
  2. 每帧：加载 npz（sem + xyz）→ cluster_instances → 检测列表。
  3. InstanceTracker.update：静态物体融合点云，动态物体记轨迹。
  4. 导出 fused_map.npz（静态物体融合点云 + track_id + class_id）与 tracks.json。

用法：
    python scripts/track_instances.py --scene_name scene-0003 --num_frames 10 \
        --out_dir out_tracking
"""
import os, sys, argparse, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.instances import cluster_instances
from utils.tracker import InstanceTracker
from utils.materials import CLASS_STATIC, LIDARSEG_32

SPLIT_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'data', 'nuscenes_official_split.json')
NPZ_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'nuscenes_seg_full')


def scene_name_to_split():
    with open(SPLIT_JSON) as f:
        d = json.load(f)
    m = {}
    for split, names in d.items():
        for n in names:
            m[n] = split
    return m


def get_lidar_tokens(nusc, scene_name, num_frames=None):
    """按时间顺序返回某 scene 的 LIDAR_TOP token 列表。"""
    scene = next(s for s in nusc.scene if s['name'] == scene_name)
    tokens, tok = [], scene['first_sample_token']
    while tok:
        sample = nusc.get('sample', tok)
        tokens.append(sample['data']['LIDAR_TOP'])
        tok = sample['next']
        if num_frames and len(tokens) >= num_frames:
            break
    return tokens


def frame_detections(sem, xyz, min_points=8):
    """一帧 → 检测列表（InstanceTracker.update 的输入）。"""
    instance_id, instances = cluster_instances(sem, xyz, min_points=min_points)
    dets = []
    for iid, info in instances.items():
        pts = xyz[instance_id == iid]
        dets.append(dict(
            class_id=info['class_id'], class_name=info['class_name'],
            center=info['center'], bbox_min=info['bbox_min'], bbox_max=info['bbox_max'],
            n_points=info['n_points'],
            points=pts,
            is_static=(info['class_id'] in CLASS_STATIC),
        ))
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default='/mnt/data/nuscenes')
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--scene_name', default='scene-0003')
    ap.add_argument('--num_frames', type=int, default=10)
    ap.add_argument('--min_points', type=int, default=8)
    ap.add_argument('--iou_thresh', type=float, default=0.2)
    ap.add_argument('--out_dir', default='out_tracking')
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    split_of = scene_name_to_split()
    split = split_of.get(args.scene_name)
    if split is None:
        sys.exit(f'场景 {args.scene_name} 不在 official split 中')

    os.makedirs(args.out_dir, exist_ok=True)
    tokens = get_lidar_tokens(nusc, args.scene_name, args.num_frames)
    print(f'场景 {args.scene_name}（{split}），处理 {len(tokens)} 帧')

    tracker = InstanceTracker(iou_thresh=args.iou_thresh)
    frame_stats = []
    for fi, tok in enumerate(tokens):
        npz = os.path.join(NPZ_ROOT, split, f'{tok}.npz')
        if not os.path.exists(npz):
            print(f'  [帧 {fi}] 缺 npz：{tok}，跳过')
            continue
        d = np.load(npz)
        xyz = d['xyz'].astype(np.float32)
        sem = d['sem'].astype(np.int64).reshape(-1)
        dets = frame_detections(sem, xyz, args.min_points)
        results = tracker.update(dets)
        n_static = sum(1 for t in tracker.tracks.values() if t['is_static'])
        n_dyn = len(tracker.tracks) - n_static
        frame_stats.append(dict(frame=fi, token=tok, detections=len(dets),
                                tracks=len(tracker.tracks),
                                static_tracks=n_static, dynamic_tracks=n_dyn))
        print(f'  [帧 {fi}] 检测 {len(dets):>3} | 跟踪 {len(tracker.tracks):>3} '
              f'(静态 {n_static}, 动态 {n_dyn})')

    # ---- 导出融合地图（静态物体点云 + track_id）----
    fused_xyz, fused_tid, fused_cls = [], [], []
    for tid, t in tracker.tracks.items():
        if t['is_static'] and t['fused_points'] is not None and len(t['fused_points']):
            p = t['fused_points']
            fused_xyz.append(p)
            fused_tid.append(np.full(len(p), tid, dtype=np.int32))
            fused_cls.append(np.full(len(p), t['class_id'], dtype=np.int32))
    if fused_xyz:
        fused_xyz = np.concatenate(fused_xyz, axis=0)
        fused_tid = np.concatenate(fused_tid, axis=0)
        fused_cls = np.concatenate(fused_cls, axis=0)
        np.savez_compressed(os.path.join(args.out_dir, 'fused_map.npz'),
                            xyz=fused_xyz, track_id=fused_tid, class_id=fused_cls)
        print(f'\n融合地图 → fused_map.npz：{len(fused_xyz):,} 点，'
              f'{len(np.unique(fused_tid))} 个静态物体')

    summary = dict(
        scene=args.scene_name, split=split, num_frames=len(tokens),
        iou_thresh=args.iou_thresh,
        frame_stats=frame_stats,
        tracks=tracker.summarize(),
    )
    with open(os.path.join(args.out_dir, 'tracks.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印若干条跟踪结果
    print('\n跟踪摘要（按帧数排序前 12）:')
    rows = sorted(tracker.summarize(), key=lambda r: -r['n_frames'])
    for r in rows[:12]:
        kind = '静态' if r['is_static'] else '动态'
        print(f"  #{r['track_id']:>3} {r['class_name']:<22} {kind} "
              f"帧={r['n_frames']:>2} 点={r['n_points']:>5} "
              f"轨迹={r['trajectory_len']:>2}")
    print(f'\ntracks.json 已导出 → {os.path.join(args.out_dir, "tracks.json")}')


if __name__ == '__main__':
    main()
