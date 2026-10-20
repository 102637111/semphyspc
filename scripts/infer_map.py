#!/usr/bin/env python3
"""地图级点云分块识别：完整 Semantic3D 场景 -> 彩色 PLY + 可选 RViz2 标注。

读 Semantic3D 原始 7z 场景（station 级，约 2000~4000 万点，即"整个地图"），
按 XY 平面 grid 分块，每块 FPS 采样到 16384 点用 PointMambaSeg 推理，再用 1-NN
把语义传播回块内全部点，拼出整图逐点语义。输出：
  彩色 PLY（真彩 / 语义 / 安全，带标量，CloudCompare/MeshLab 打开即彩色）
  可选 RViz2（降采样 ~50 万点发布 /semvis/cloud + /semvis/objects 框+标签）

语义体系：nuScenes 32 类（零样本，ckpt mIoU 0.2342）；Semantic3D 场景上模型
只会输出它认识的类（车/地面/植被/建筑等），rare 类识别不到属正常。

用法：
  # 完整地图识别 + 彩色 PLY（CloudCompare 查看）
  python scripts/infer_map.py \
      --archive /mnt/data/datasets/Semantic3D/train/domfountain_station3_xyz_intensity_rgb.7z \
      --config experiments/seg_nuscenes_full_16k/cfgs/singletask_sem/20260821-221520/config.yaml \
      --ckpts  experiments/seg_nuscenes_full_16k/cfgs/singletask_sem/20260821-221520/ckpt-best.pth \
      --output_dir paper/figures/map

  # 加 RViz2 交互标注（框+标签）
      --rviz

  # 快速小规模验证（只读前 20 万点）
      --max_points 200000

  # 通用：非 7z 的裸 txt / npz 大点云，用 --txt <file> 或 --npz <file> 代替 --archive
"""
import os
import sys
import argparse
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.materials import LIDARSEG_32, COLLISION_SAFETY, apply_materials
from export_paper_visuals import SEM_PALETTE, SAFETY_COLORS, write_ply_binary, _short_name

BLOCK = 16384  # PointMambaSeg 固定输入点数


# ---------------- 大点云读取 ----------------
def _parse_lines(buf):
    """把字节缓冲按行解析成 (M,7) float32：x y z intensity r g b。"""
    arr = np.fromstring(buf, sep=' ', dtype=np.float32)
    return arr.reshape(-1, 7) if arr.size else np.zeros((0, 7), np.float32)


def read_txt_stream(path, max_points=None):
    """流式读 7 列 txt（支持 .7z 自动流式解压），返回 xyz(N,3) float32, rgb(N,3) float32 [0,1]。"""
    if path.endswith('.7z'):
        pipe = subprocess.Popen(['7z', 'x', '-so', path],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    else:
        pipe = open(path, 'rb')
    xs, rs = [], []
    buf = b''
    total = 0
    while True:
        chunk = pipe.read(4 << 20)
        if not chunk:
            break
        buf += chunk
        *lines, buf = buf.split(b'\n')
        if lines:
            arr = _parse_lines(b'\n'.join(lines))
            if arr.shape[0]:
                xs.append(arr[:, :3])
                rs.append(arr[:, 4:7] / 255.0)
                total += arr.shape[0]
                if max_points and total >= max_points:
                    break
    if buf.strip() and not (max_points and total >= max_points):
        arr = _parse_lines(buf)
        if arr.shape[0]:
            xs.append(arr[:, :3])
            rs.append(arr[:, 4:7] / 255.0)
            total += arr.shape[0]
    xyz = np.concatenate(xs, axis=0)
    rgb = np.concatenate(rs, axis=0)
    if max_points:
        xyz, rgb = xyz[:max_points], rgb[:max_points]
    return xyz, rgb


def read_npz(path, max_points=None):
    d = np.load(path)
    xyz = d['xyz'].astype(np.float32).reshape(-1, 3)
    rgb = d['rgb'].astype(np.float32).reshape(-1, 3)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    if max_points:
        xyz, rgb = xyz[:max_points], rgb[:max_points]
    return xyz, rgb


# ---------------- 分块推理核心 ----------------
class MapSegmenter:
    """加载一次 PointMambaSeg，对任意大点云逐块推理 + 1-NN 传播。"""

    def __init__(self, config_path, ckpt_path):
        import torch
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for _ext in ('chamfer_dist', 'emd'):
            _p = os.path.join(_root, 'extensions', _ext)
            if _p not in sys.path:
                sys.path.insert(0, _p)
        from tools import builder
        from utils.config import get_config

        class _A:
            pass
        args = _A()
        args.config = config_path
        args.local_rank = 0
        args.use_gpu = torch.cuda.is_available()
        args.distributed = False
        args.resume = False
        args.experiment_path = '/tmp/infer_map_exp'
        os.makedirs(args.experiment_path, exist_ok=True)
        args.log_name = 'infer_map'
        os.environ.setdefault('LOCAL_RANK', '0')

        from utils.logger import get_root_logger
        logger = get_root_logger(name='infer_map')
        config = get_config(args, logger=logger)
        self.in_channels = getattr(config.model, 'in_channels', 6)
        model = builder.model_builder(config.model)
        builder.load_model(model, ckpt_path, logger=logger)
        self.model = torch.nn.DataParallel(model).cuda().eval()
        self.npoints = BLOCK

    def _predict_block(self, xyz_b, rgb_b):
        """输入块 (n,3)/(n,3)，返回块内每点语义 (n,) int64。"""
        import torch
        from pointnet2_ops import pointnet2_utils
        n = len(xyz_b)
        if n >= self.npoints:
            t = torch.from_numpy(xyz_b.astype(np.float32)).cuda().unsqueeze(0)
            idx = pointnet2_utils.furthest_point_sample(t, self.npoints).squeeze(0).cpu().numpy()
            xyz_s, rgb_s = xyz_b[idx], rgb_b[idx]
            subsampled = True
        else:
            # 重复采样到 npoints：补零会引入远离点云的 (0,0,0)，干扰 FPS 采 center +
            # Hilbert 序列化（极端时 grid_coord.max()=0 -> depth=0 崩溃）
            reps = np.tile(np.arange(n), (self.npoints + n - 1) // n)[:self.npoints]
            xyz_s, rgb_s = xyz_b[reps], rgb_b[reps]
            subsampled = False

        pts = np.concatenate([xyz_s, rgb_s], axis=1) if self.in_channels >= 6 else xyz_s
        if pts.shape[1] < self.in_channels:
            pts = np.concatenate([pts, np.zeros((len(pts), self.in_channels - pts.shape[1]), np.float32)], axis=1)
        batch = torch.from_numpy(pts).float().unsqueeze(0).cuda()
        with torch.no_grad():
            ret = self.model(batch)
        logits = ret[0] if isinstance(ret, (tuple, list)) else ret
        sem_s = logits.argmax(-1).squeeze(0).cpu().numpy().astype(np.int64)
        if not subsampled:
            return sem_s[:n]
        # 1-NN 把采样点语义传播回块内全部点
        from scipy.spatial import cKDTree
        _, nn = cKDTree(xyz_s).query(xyz_b, k=1)
        return sem_s[nn]

    def segment(self, xyz, rgb, cell_size=30.0, min_cell_points=256, verbose=True):
        """整图分块推理，返回 sem (N,) int64（点数过少的 cell 标 -1 未识别）。"""
        n = len(xyz)
        gx = np.floor(xyz[:, 0] / cell_size).astype(np.int64)
        gy = np.floor(xyz[:, 1] / cell_size).astype(np.int64)
        _, inv, counts = np.unique(np.stack([gx, gy], axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
        order = np.argsort(inv, kind='stable')
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        ncell = len(counts)
        sem = np.full(n, -1, dtype=np.int32)

        if verbose:
            print(f'  分块：{ncell} 个 cell，cell={cell_size}m，总数 {n:,} 点', flush=True)
        skipped = 0
        for c in range(ncell):
            if counts[c] < min_cell_points:
                skipped += 1
                continue
            idx = order[starts[c]:starts[c] + counts[c]]
            sem[idx] = self._predict_block(xyz[idx], rgb[idx]).astype(np.int32)
            if verbose and (c + 1) % max(1, ncell // 20) == 0:
                print(f'    推理 {c + 1}/{ncell} 块', flush=True)
        if verbose:
            if skipped:
                print(f'  跳过 {skipped} 个点数 < {min_cell_points} 的小 cell（标为未识别）', flush=True)
            print(f'  推理完成，有效语义 {int((sem >= 0).sum()):,} / {n:,} 点', flush=True)
        return sem


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--archive', help='Semantic3D 场景 .7z')
    src.add_argument('--txt', help='裸 7 列 txt（x y z intensity r g b）')
    src.add_argument('--npz', help='大点云 npz（含 xyz/rgb）')
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpts', required=True)
    ap.add_argument('--output_dir', default='paper/figures/map')
    ap.add_argument('--cell_size', type=float, default=30.0, help='XY 分块边长（m）')
    ap.add_argument('--min_cell_points', type=int, default=256, help='点数少于该值的 cell 跳过（标为未识别）')
    ap.add_argument('--max_points', type=int, default=0, help='>0 时只读前 N 点（快速验证）')
    ap.add_argument('--rviz', action='store_true', help='额外降采样发布 RViz2（框+标签）')
    ap.add_argument('--rviz_points', type=int, default=500000, help='RViz2 降采样点数')
    ap.add_argument('--frame', default='map')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.archive or args.txt:
        src_path = args.archive or args.txt
        xyz, rgb = read_txt_stream(src_path, args.max_points or None)
    else:
        xyz, rgb = read_npz(args.npz, args.max_points or None)

    stem = os.path.splitext(os.path.basename(args.archive or args.txt or args.npz))[0]
    if stem.endswith('_xyz_intensity_rgb') or stem.endswith('_intensity_rgb'):
        stem = stem.replace('_xyz_intensity_rgb', '').replace('_intensity_rgb', '')
    print(f'[{stem}] 读入 {len(xyz):,} 点', flush=True)

    seg = MapSegmenter(args.config, args.ckpts)
    sem = seg.segment(xyz, rgb, cell_size=args.cell_size, min_cell_points=args.min_cell_points)

    # ---- 彩色 PLY（完整，CloudCompare 查看）----
    mat = apply_materials(sem)
    sem_color = np.zeros((len(xyz), 3), np.float32)
    for c in range(len(LIDARSEG_32)):
        sem_color[sem == c] = SEM_PALETTE[c]
    safety_color = np.zeros((len(xyz), 3), np.float32)
    for si, sname in enumerate(COLLISION_SAFETY):
        safety_color[mat['safety_id'] == si] = SAFETY_COLORS[sname]

    print(f'  写 PLY -> {args.output_dir}/', flush=True)
    write_ply_binary(os.path.join(args.output_dir, f'{stem}_map_rgb.ply'), xyz, rgb)
    write_ply_binary(os.path.join(args.output_dir, f'{stem}_map_semantic.ply'), xyz, sem_color,
                     scalars={'semantic_id': ('int', sem.astype(np.int32))})
    write_ply_binary(os.path.join(args.output_dir, f'{stem}_map_safety.ply'), xyz, safety_color,
                     scalars={'safety_id': ('int', mat['safety_id'].astype(np.int32)),
                              'physical_id': ('int', mat['physical_id'].astype(np.int32)),
                              'lethality': ('float', mat['lethality']),
                              'destructibility': ('float', mat['destructibility']),
                              'friction': ('float', mat['friction'])})

    present = np.unique(sem[sem >= 0])
    print(f'  识别类别 {len(present)} 个: ' +
          ', '.join(_short_name(LIDARSEG_32[c]) for c in present), flush=True)

    # ---- 可选 RViz2 降采样发布 ----
    if args.rviz:
        from rviz_semvis import build_cloud_msg, build_markers, cluster_instances
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Header
        from sensor_msgs.msg import PointCloud2
        from visualization_msgs.msg import MarkerArray
        import torch
        from pointnet2_ops import pointnet2_utils

        n = min(len(xyz), args.rviz_points)
        t = torch.from_numpy(xyz.astype(np.float32)).cuda().unsqueeze(0)
        idx = pointnet2_utils.furthest_point_sample(t, n).squeeze(0).cpu().numpy()
        xyz_v, rgb_v, sem_v = xyz[idx], rgb[idx], sem[idx]

        inst = cluster_instances(xyz_v, sem_v, eps=1.0, min_points=30)
        print(f'  RViz2 降采样 {n:,} 点，{len(inst)} 个物体实例', flush=True)

        rclpy.init()
        node = Node('infer_map_node')
        header = Header()
        header.frame_id = args.frame
        cloud_msg = build_cloud_msg(header, xyz_v, rgb_v)
        markers_msg = build_markers(header, inst)
        pub_cloud = node.create_publisher(PointCloud2, 'semvis/cloud', 10)
        pub_obj = node.create_publisher(MarkerArray, 'semvis/objects', 10)

        def publish_once():
            header.stamp = node.get_clock().now().to_msg()
            cloud_msg.header.stamp = header.stamp
            for m in markers_msg.markers:
                m.header.stamp = header.stamp
            pub_cloud.publish(cloud_msg)
            pub_obj.publish(markers_msg)

        publish_once()
        node.create_timer(1.0, publish_once)
        node.get_logger().info(
            f'地图识别发布中：/semvis/cloud ({n:,} 点) + /semvis/objects ({len(inst)} 实例)，frame={args.frame}')
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
