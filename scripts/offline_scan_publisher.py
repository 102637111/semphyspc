#!/usr/bin/env python3
"""离线导航仿真用假激光发布器：发布 /scan（全 max_range，表示开阔无遮挡）。

让 nav2_amcl 有扫描数据可读、生命周期正常运行；AMCL 在收到 initialpose 后广播
map→odom，之后假 scan 全开阔（粒子不更新，保持 initialpose 位姿）。

真实机器人接入后应停用此节点（use_fake_scan:=false），改用真实雷达。
"""
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class OfflineScanPublisher(Node):
    def __init__(self):
        super().__init__('offline_scan_publisher')
        self.pub = self.create_publisher(LaserScan, 'scan', 10)
        self.timer = self.create_timer(0.1, self.publish_scan)  # 10Hz
        self.get_logger().info('发布假激光 /scan（全 max_range，表示开阔无遮挡）')

    def publish_scan(self):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = math.pi / 180.0  # 360 个点
        n = int(round((msg.angle_max - msg.angle_min) / msg.angle_increment)) + 1
        msg.range_min = 0.1
        msg.range_max = 30.0
        msg.ranges = [30.0] * n
        msg.intensities = [0.0] * n
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = OfflineScanPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
