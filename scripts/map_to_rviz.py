#!/usr/bin/env python3
"""地图识别结果 -> RViz2 交互查看（读 infer_map.py 生成的 PLY，不重新推理）。

读 rgb PLY（真彩）+ semantic PLY（semantic_id 标量），FPS 降采样到 ~50 万点，
发布 /semvis/cloud（彩色）+ /semvis/objects（thing 类框+标签）。

用法：
  bash scripts/run_map_rviz.sh   # 或手动：
  python scripts/map_to_rviz.py \
      --rgb      paper/figures/map/domfountain_station3_map_rgb.ply \
      --semantic paper/figures/map/domfountain_station3_map_semantic.ply
  # 另开终端 rviz2 -d scripts/rviz_semvis.rviz
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rviz_semvis import build_cloud_msg, build_markers, cluster_instances, LIDARSEG_32, _short_name

_DT = {'float': '<f4', 'uchar': 'u1', 'int': '<i4', 'uint': '<u4',
       'short': '<i2', 'ushort': '<u2'}


def read_ply(path):
    """读二进制 PLY（本项目的 write_ply_binary 格式），返回结构化数组。"""
    with open(path, 'rb') as f:
        n_vertex = 0
        props = []
        while True:
            raw = f.readline()
            line = raw.decode('ascii', 'ignore').strip()
            if line.startswith('element vertex'):
                n_vertex = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                props.append((parts[-1], parts[1]))
            elif line == 'end_header':
                break
        dtype = np.dtype([(name, _DT[t]) for name, t in props])
        return np.fromfile(f, dtype=dtype, count=n_vertex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb', required=True, help='真彩 PLY（_map_rgb.ply）')
    ap.add_argument('--semantic', required=True, help='语义 PLY（_map_semantic.ply）')
    ap.add_argument('--npoints', type=int, default=500000, help='RViz2 降采样点数')
    ap.add_argument('--fps', action='store_true', help='用 FPS 降采样（空间均匀但大点云慢）')
    ap.add_argument('--frame', default='map')
    ap.add_argument('--eps', type=float, default=1.0)
    ap.add_argument('--min_points', type=int, default=30)
    ap.add_argument('--rate', type=float, default=1.0)
    args = ap.parse_args()

    drgb = read_ply(args.rgb)
    dsem = read_ply(args.semantic)
    n = len(drgb)
    xyz = np.stack([drgb['x'], drgb['y'], drgb['z']], axis=1).astype(np.float32)
    rgb = np.stack([drgb['red'], drgb['green'], drgb['blue']], axis=1).astype(np.float32) / 255.0
    sem = dsem['semantic_id'].astype(np.int64).reshape(-1)
    print(f'读入 {n:,} 点（rgb={len(drgb)} / semantic={len(dsem)}）', flush=True)

    # 降采样（默认随机，秒级；--fps 用 FPS 保空间均匀但大点云很慢）
    k = min(n, args.npoints)
    if args.fps:
        import torch
        from pointnet2_ops import pointnet2_utils
        t = torch.from_numpy(xyz).cuda().unsqueeze(0)
        idx = pointnet2_utils.furthest_point_sample(t, k).squeeze(0).cpu().numpy()
    else:
        idx = np.random.default_rng(0).choice(n, k, replace=False)
    xyz_v, rgb_v, sem_v = xyz[idx], rgb[idx], sem[idx]

    inst = cluster_instances(xyz_v, sem_v, eps=args.eps, min_points=args.min_points)
    print(f'降采样 {k:,} 点，{len(inst)} 个物体实例', flush=True)
    for cls, pts in inst:
        print(f'  - {_short_name(LIDARSEG_32[cls]):<16s} {len(pts):>6d} 点', flush=True)

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Header
    from sensor_msgs.msg import PointCloud2
    from visualization_msgs.msg import MarkerArray

    rclpy.init()
    node = Node('map_to_rviz')
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
    node.create_timer(1.0 / args.rate, publish_once)
    node.get_logger().info(
        f'地图发布中：/semvis/cloud ({k:,} 点彩色) + /semvis/objects ({len(inst)} 框+标签)，frame={args.frame}')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
