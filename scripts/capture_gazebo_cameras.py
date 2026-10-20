#!/usr/bin/env python3
"""订阅 Gazebo 相机图像，持续把最新帧存为 PNG（供论文截图）。

配合 semantic_track.world 里的 camera_top / camera_persp 两个相机。
持续运行，每收到一帧就覆盖写 /tmp/cam_top_latest.png、/tmp/cam_persp_latest.png，
外部在机器人到达后再复制到最终路径。结束用 Ctrl-C 或 kill。

用法：
  python3 scripts/capture_gazebo_cameras.py
"""
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from PIL import Image as PILImage


class CameraSaver(Node):
    def __init__(self):
        super().__init__('capture_gazebo_cameras')
        self.create_subscription(
            Image, '/camera_top/image_raw', self._make_cb('/tmp/cam_top_latest.png'), 5)
        self.create_subscription(
            Image, '/camera_persp/image_raw', self._make_cb('/tmp/cam_persp_latest.png'), 5)
        self.get_logger().info('订阅 /camera_top/image_raw 与 /camera_persp/image_raw')

    def _make_cb(self, path):
        def cb(msg):
            if msg.encoding.lower().strip() != 'rgb8':
                return
            try:
                img = PILImage.frombytes('RGB', (msg.width, msg.height), bytes(msg.data))
                img.save(path)
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f'save {path} 失败: {e}')
        return cb


def main():
    rclpy.init()
    node = CameraSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
