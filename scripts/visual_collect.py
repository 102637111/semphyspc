#!/usr/bin/env python3
"""
可视化彩色点云数据集采集工具 v3

RViz2 显示 3D 点云（相机已启动即自动显示），本脚本提供终端控制面板。

用法:
    # 终端1 (机器人): ros2 launch orbbec_camera gemini.launch.py
    # 终端2 (本地):   rviz2  (添加 PointCloud2 display, topic=/camera/depth_registered/points)
    # 终端3 (本地):   python scripts/visual_collect.py

操作:
  1-9,0,-,=  切换预设类别
  ENTER      采集当前帧
  A          自动采集模式（每2秒一帧，手动旋转物体）
  S          停止自动采集
  N          输入新类别名
  C          显示统计
  Q          退出
"""

import os, sys, time, argparse, threading, signal, numpy as np
from collections import defaultdict
from pathlib import Path

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

PRESET_CLASSES = [
    'cup', 'bottle', 'bowl', 'keyboard', 'laptop',
    'chair', 'monitor', 'plant', 'book', 'phone',
    'lamp', 'others'
]
KEY_CLASS = {
    '1':0, '2':1, '3':2, '4':3, '5':4,
    '6':5, '7':6, '8':7, '9':8, '0':9,
    '-':10, '=':11,
}


class CollectorNode(Node):
    """ROS2 订阅 + 采集控制"""

    def __init__(self, output_dir, camera_name='camera'):
        super().__init__('collector')
        self.output_dir = Path(output_dir)
        self.current_class = PRESET_CLASSES[0]
        self.counts = defaultdict(int)
        self.latest_cloud = None  # (xyz, rgb)
        self.auto_mode = False
        self.auto_interval = 2.0
        self.auto_last = 0
        self.running = True

        topic = f'/{camera_name}/depth_registered/points'
        self.sub = self.create_subscription(PointCloud2, topic, self._cb, 10)
        self.get_logger().info(f'Ready. Topic: {topic}')

    def _cb(self, msg):
        pts, cols = [], []
        for p in point_cloud2.read_points(msg, ['x','y','z','rgb'], skip_nans=True):
            pts.append([p[0], p[1], p[2]])
            rgb_b = np.frombuffer(np.array(p[3], dtype=np.float32).tobytes(), dtype=np.uint8)
            cols.append([rgb_b[2]/255., rgb_b[1]/255., rgb_b[0]/255.])
        self.latest_cloud = (np.array(pts, dtype=np.float32),
                            np.array(cols, dtype=np.float32))

    def save(self):
        if self.latest_cloud is None:
            return 0
        xyz, rgb = self.latest_cloud

        class_dir = self.output_dir / self.current_class
        class_dir.mkdir(parents=True, exist_ok=True)
        n = self.counts[self.current_class] + 1
        self.counts[self.current_class] = n

        pc = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
        np.save(str(class_dir / f'{self.current_class}_{n:04d}.npy'), pc)
        return len(pc)

    def auto_tick(self):
        if not self.auto_mode:
            return
        now = time.time()
        if now - self.auto_last >= self.auto_interval:
            self.auto_last = now
            n = self.save()
            if n > 0:
                cnt = self.counts[self.current_class]
                print(f"\r  [{self.current_class}] #{cnt}: {n} pts  ", end='', flush=True)

    def set_class(self, key):
        if key in KEY_CLASS:
            self.current_class = PRESET_CLASSES[KEY_CLASS[key]]
            return True
        return False

    def total(self):
        return sum(self.counts.values())


def print_header(node):
    _ = os.system('clear 2>/dev/null') or True
    print("""
╔══════════════════════════════════════════════════════╗
║     Color-PointMamba 可视化数据集采集工具 v3         ║
╠══════════════════════════════════════════════════════╣
║  RViz2 显示实时点云  |  终端控制采集                 ║
╠══════════════════════════════════════════════════════╣
║  预设类别 (按数字键切换):                             ║
║    1:cup   2:bottle   3:bowl   4:keyboard  5:laptop ║
║    6:chair 7:monitor  8:plant  9:book      0:phone  ║
║    -:lamp  =:others                                 ║
╠══════════════════════════════════════════════════════╣
║  操作:                                               ║
║    ENTER      采集当前帧                             ║
║    A + ENTER  自动采集 (每2秒)                       ║
║    S + ENTER  停止自动采集                           ║
║    N + ENTER  自定义类别名                           ║
║    C + ENTER  显示统计                               ║
║    Q + ENTER  退出                                   ║
╚══════════════════════════════════════════════════════╝
""")


def print_status(node):
    status = f"""
┌──────────────────────────────────────────┐
│ 当前类别: {node.current_class:<20s}           │
│ 已采集:   {node.counts[node.current_class]:<4d} 帧                       │
│ 模式:     {'AUTO (自动)' if node.auto_mode else 'MANUAL (手动)'}           │
│ 总计:     {node.total():<4d} 帧                       │
│ 输出:     {node.output_dir}/
└──────────────────────────────────────────┘
"""
    print(status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./my_raw_dataset')
    parser.add_argument('--camera_name', default='camera')
    parser.add_argument('--auto_interval', type=float, default=2.0)
    args = parser.parse_args()

    rclpy.init()
    node = CollectorNode(args.output_dir, args.camera_name)
    node.auto_interval = args.auto_interval

    # 等待第一帧
    print("等待相机数据...", end=' ', flush=True)
    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.latest_cloud is not None:
            break
    else:
        print("\nERROR: 检查 ros2 launch orbbec_camera gemini.launch.py")
        sys.exit(1)
    print("OK")

    print_header(node)
    print("  ⚠️  请打开 RViz2 并添加 PointCloud2 Display:")
    print("       topic = /camera/depth_registered/points")
    print()

    # ── 使用 select 实现非阻塞键盘输入 ──
    import select, termios, tty
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())
        node.get_logger().info('开始采集 (按 Q 退出)')

        while node.running and rclpy.ok():
            # 处理 ROS2
            rclpy.spin_once(node, timeout_sec=0.05)

            # 自动采集 tick
            if node.auto_mode:
                node.auto_tick()

            # 非阻塞读取键盘
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                key = sys.stdin.read(1)

                if key in ('q', 'Q'):
                    break
                elif key in KEY_CLASS:
                    node.set_class(key)
                    print_status(node)
                elif key in ('\r', '\n'):  # ENTER
                    n = node.save()
                    if n > 0:
                        cnt = node.counts[node.current_class]
                        print(f"  ✓ [{node.current_class}] #{cnt}: {n} pts saved")
                elif key in ('a', 'A'):
                    if node.auto_mode:
                        node.auto_mode = False
                        print("\n  ■ AUTO STOPPED")
                    else:
                        node.auto_mode = True
                        node.auto_last = time.time()
                        print(f"\n  ▶ AUTO STARTED ({node.auto_interval}s interval)")
                        print(f"    请在 RViz2 中观察，缓慢旋转物体...")
                elif key in ('s', 'S'):
                    node.auto_mode = False
                    print("\n  ■ STOPPED")
                elif key in ('n', 'N'):
                    name = input("\n  新类别名: ").strip().lower()
                    if name:
                        node.current_class = name
                        print(f"  → 切换到: {name}")
                        print_status(node)
                elif key in ('c', 'C'):
                    print_status(node)

    finally:
        termios.tcsetattr(sys.stdin, old_settings)
        total = node.total()
        print(f"\n\n{'='*60}")
        print(f"  采集完成! {total} 帧 → {node.output_dir}/")
        if total > 0:
            print(f"\n  下一步:")
            print(f"    python scripts/build_rtabmap_dataset.py \\")
            print(f"        --raw_dir {node.output_dir} \\")
            print(f"        --output_dir ./data/my_real_dataset")
        print(f"{'='*60}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
