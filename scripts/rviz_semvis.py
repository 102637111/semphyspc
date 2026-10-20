#!/usr/bin/env python3
"""RViz2 语义点云可视化节点（离线）：彩色 PointCloud2 + 物体包围盒 + 类别小字标签。

读 npz（xyz + rgb + sem），在 RViz2 里展示：
  /semvis/cloud    sensor_msgs/PointCloud2       RGB 彩色点云（真实还原）
  /semvis/objects  visualization_msgs/MarkerArray  thing 类实例：CUBE 包围盒 + TEXT 类别名

语义来源：默认用 npz 自带 sem（真值）；给 --config + --ckpts 时用 PointMambaSeg 推理。

运行（先 source ROS，再用带 rclpy 的 python）：
  source /opt/ros/humble/setup.bash
  /home/saw/ros2_ws/.venv/bin/python scripts/rviz_semvis.py \
      --npz data/nuscenes_seg_full_16k/val/17bd604f6828423680434cd2e522180a.npz \
      --config experiments/seg_nuscenes_full_16k/cfgs/singletask_sem/20260821-221520/config.yaml \
      --ckpts experiments/seg_nuscenes_full_16k/cfgs/singletask_sem/20260821-221520/ckpt-best.pth

然后另开终端：
  rviz2 -d <本目录>/rviz_semvis.rviz   （或手动 add：PointCloud2 订阅 /semvis/cloud，MarkerArray 订阅 /semvis/objects，
   固定 Frame 设为 --frame 指定的值，默认 map）
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.materials import LIDARSEG_32

# 复用可视化脚本的推理函数（含 chamfer sys.path 修复 + forward tuple 解包）
from export_paper_visuals import run_seg_inference, SEM_PALETTE, _short_name

# ---------------- ROS 依赖（惰性，离线可先测聚类）----------------
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Header
    from sensor_msgs.msg import PointCloud2, PointField
    from visualization_msgs.msg import Marker, MarkerArray
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

# thing 类（物体）语义 id：1..23；0(noise) 与 24..31(stuff/背景) 不框
THING_IDS = set(range(1, 24))


# ---------------- 实例聚类（对 thing 类点做 DBSCAN）----------------
def cluster_instances(xyz, sem, eps=1.0, min_points=10):
    """把 thing 类点按空间聚成实例，返回 [(cls, pts_idx), ...]。"""
    from sklearn.cluster import DBSCAN
    thing_mask = np.isin(sem, list(THING_IDS))
    if thing_mask.sum() < min_points:
        return []
    xyz_t = xyz[thing_mask]
    sem_t = sem[thing_mask]
    lab = DBSCAN(eps=eps, min_samples=min_points).fit_predict(xyz_t)
    out = []
    for c in np.unique(lab):
        if c < 0:  # 噪声
            continue
        idx = np.where(lab == c)[0]
        if len(idx) < min_points:
            continue
        cls = int(np.bincount(sem_t[idx]).argmax())  # 该实例众数类别
        out.append((cls, xyz_t[idx]))
    return out


# ---------------- PointCloud2 构造（RGB bit-packed）----------------
def build_cloud_msg(header, xyz, rgb):
    n = len(xyz)
    rgb = np.clip(np.asarray(rgb, dtype=np.float32).reshape(n, 3), 0.0, 1.0)
    r = (rgb[:, 0] * 255).astype(np.uint32)
    g = (rgb[:, 1] * 255).astype(np.uint32)
    b = (rgb[:, 2] * 255).astype(np.uint32)
    packed = ((r << 16) | (g << 8) | b).astype(np.uint32).view(np.float32)

    data = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')])
    data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data['rgb'] = packed

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.data = data.tobytes()
    return msg


def _color_from_palette(rgb01):
    from std_msgs.msg import ColorRGBA
    c = ColorRGBA()
    c.r, c.g, c.b = float(rgb01[0]), float(rgb01[1]), float(rgb01[2])
    c.a = 0.35  # 包围盒半透明
    return c


def build_markers(header, instances):
    """为每个实例生成 CUBE 包围盒 + TEXT 类别标签。"""
    ma = MarkerArray()
    for i, (cls, pts) in enumerate(instances):
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        center = (lo + hi) / 2.0
        size = np.maximum(hi - lo, 0.3)  # 避免退化成零厚度

        # CUBE 包围盒
        box = Marker()
        box.header = header
        box.ns = 'bbox'
        box.id = i
        box.type = Marker.CUBE
        box.action = Marker.ADD
        box.pose.position.x = float(center[0])
        box.pose.position.y = float(center[1])
        box.pose.position.z = float(center[2])
        box.pose.orientation.w = 1.0
        box.scale.x = float(size[0])
        box.scale.y = float(size[1])
        box.scale.z = float(size[2])
        box.color = _color_from_palette(SEM_PALETTE[cls])
        ma.markers.append(box)

        # TEXT 小字标签（包围盒顶部上方）
        name = _short_name(LIDARSEG_32[cls])
        txt = Marker()
        txt.header = header
        txt.ns = 'label'
        txt.id = i
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = float(center[0])
        txt.pose.position.y = float(center[1])
        txt.pose.position.z = float(hi[2] + 0.6)
        txt.pose.orientation.w = 1.0
        txt.scale.z = 1.0  # 字高 1m
        txt.color.r = 1.0
        txt.color.g = 1.0
        txt.color.b = 1.0
        txt.color.a = 1.0
        txt.text = name
        ma.markers.append(txt)
    return ma


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--ckpts', default=None)
    ap.add_argument('--npoints', type=int, default=16384)
    ap.add_argument('--frame', default='map')
    ap.add_argument('--eps', type=float, default=1.0)
    ap.add_argument('--min_points', type=int, default=10)
    ap.add_argument('--rate', type=float, default=1.0)
    args = ap.parse_args()

    d = np.load(args.npz)
    xyz = d['xyz'].astype(np.float32)
    rgb = d.get('rgb', None)
    if rgb is not None:
        rgb = rgb.astype(np.float32).reshape(len(xyz), 3)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
    gt_sem = d.get('sem', None)
    if gt_sem is not None:
        gt_sem = gt_sem.astype(np.int64).reshape(-1)

    if args.config and args.ckpts:
        if rgb is None:
            raise ValueError('模型推理需要 rgb（6 通道输入）')
        xyz, rgb, sem = run_seg_inference(xyz, rgb, args.config, args.ckpts, args.npoints)
        sem_src = 'pred'
    elif gt_sem is not None:
        sem = gt_sem
        sem_src = 'gt'
    else:
        raise ValueError('npz 需含 sem，或提供 --config + --ckpts 做推理')

    instances = cluster_instances(xyz, sem, eps=args.eps, min_points=args.min_points)
    print(f'语义来源={sem_src}  点数={len(xyz):,}  实例数={len(instances)}')
    for cls, pts in instances:
        print(f'  - {_short_name(LIDARSEG_32[cls]):<16s} {len(pts):>6d} 点')

    if not HAS_ROS:
        print('\n[无 ROS 环境] 仅完成聚类测试，跳过发布。')
        return

    rclpy.init()
    node = Node('semvis_node')

    header = Header()
    header.frame_id = args.frame

    cloud_msg = build_cloud_msg(header, xyz, rgb) if rgb is not None else None
    markers_msg = build_markers(header, instances)

    pub_cloud = node.create_publisher(PointCloud2, 'semvis/cloud', 10)
    pub_obj = node.create_publisher(MarkerArray, 'semvis/objects', 10)

    def publish_once():
        header.stamp = node.get_clock().now().to_msg()
        if cloud_msg is not None:
            cloud_msg.header.stamp = header.stamp
            pub_cloud.publish(cloud_msg)
        markers_msg.markers = [m for m in markers_msg.markers]
        for m in markers_msg.markers:
            m.header.stamp = header.stamp
        pub_obj.publish(markers_msg)

    publish_once()
    node.create_timer(1.0 / args.rate, publish_once)
    node.get_logger().info(
        f'semvis 发布中：/semvis/cloud (RGB 彩色) + /semvis/objects ({len(instances)} 实例框+标签)，'
        f'frame={args.frame}')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
