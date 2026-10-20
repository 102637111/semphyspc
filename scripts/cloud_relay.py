#!/usr/bin/env python3
"""
点云降采样中继节点 — 运行在机器人上，将大点云缩小后发布。

机器人端: 订阅 /camera/depth_registered/points (95k点, ~3MB)
         → 体素降采样 (0.01m) → /camera/depth_registered/points_sparse (~8k点, ~250KB)
         → 带宽降低 ~12x

用法 (在机器人上):
    scp scripts/cloud_relay.py intel:~/ros2_ws/src/semphyspc/scripts/
    ssh intel
    cd ~/ros2_ws
    source install/setup.bash
    python3 src/semphyspc/scripts/cloud_relay.py
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


class CloudRelay(Node):
    def __init__(self):
        super().__init__('cloud_relay')
        self.sub = self.create_subscription(
            PointCloud2, '/camera/depth_registered/points', self._cb, 10)
        self.pub = self.create_publisher(
            PointCloud2, '/camera/depth_registered/points_sparse', 10)
        self.get_logger().info('Relay: /depth_registered/points → /depth_registered/points_sparse')

    def _cb(self, msg):
        # 读取所有点
        pts = list(point_cloud2.read_points(msg, ['x','y','z','rgb'], skip_nans=True))
        if not pts:
            return

        # Numpy 化
        xyz = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)
        rgb = np.array([p[3] for p in pts], dtype=np.float32)

        # 体素降采样 (0.01m grid)
        voxel_size = 0.01
        voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)

        xyz_ds = xyz[unique_idx]
        rgb_ds = rgb[unique_idx]

        # 如果还是太多，进一步随机采样
        max_pts = 15000
        if len(xyz_ds) > max_pts:
            idx = np.random.choice(len(xyz_ds), max_pts, replace=False)
            xyz_ds = xyz_ds[idx]
            rgb_ds = rgb_ds[idx]

        # 构建新消息
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # Pack: [x,y,z,rgb] as float32
        data = np.column_stack([xyz_ds, rgb_ds]).astype(np.float32).tobytes()

        out = PointCloud2(
            header=header,
            height=1, width=len(xyz_ds),
            fields=fields,
            is_bigendian=False,
            point_step=16,
            row_step=16 * len(xyz_ds),
            data=data,
            is_dense=True,
        )
        self.pub.publish(out)


def main():
    rclpy.init()
    node = CloudRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
