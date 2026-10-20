#!/usr/bin/env python3
"""
Interactive dataset collection tool for Orbbec Gemini camera.

Captures colored point clouds and lets you label them in real-time.
Builds a dataset suitable for Color-PointMamba training.

Usage:
    # Interactive mode - type class name for each object
    python scripts/label_and_capture.py \
        --output_dir ./my_raw_dataset \
        --frames_per_object 10

    # Then process into trainable dataset:
    python scripts/build_rtabmap_dataset.py \
        --raw_dir ./my_raw_dataset \
        --output_dir ./data/my_real_dataset

    # Train on real data:
    python main.py \
        --config cfgs/finetune_modelnet_real.yaml \
        --scratch_model
"""

import os, sys, time, argparse, numpy as np, threading
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from rclpy.qos import qos_profile_sensor_data


class InteractiveCapture(Node):
    """Capture colored point clouds with manual labeling."""

    def __init__(self, output_dir, camera_name='camera', suffix=''):
        super().__init__('interactive_capture')
        self.output_dir = Path(output_dir)
        self.latest_cloud = None
        self.lock = threading.Lock()

        topic = f'/{camera_name}/depth_registered/points'
        self.sub = self.create_subscription(
            PointCloud2, topic, self._cb, qos_profile_sensor_data)
        self.get_logger().info(f'Subscribed to {topic}')

    def _cb(self, msg):
        with self.lock:
            self.latest_cloud = msg

    def wait_for_cloud(self, timeout=10.0):
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            with self.lock:
                if self.latest_cloud is not None:
                    return True
        return False

    def extract(self, msg, sparse=1):
        """Extract XYZ+RGB from PointCloud2. Returns [N, 6] numpy."""
        pts, cols = [], []
        for i, p in enumerate(point_cloud2.read_points(msg, ['x','y','z','rgb'], skip_nans=True)):
            if sparse > 1 and i % sparse != 0:
                continue
            x, y, z, rgb = p[0], p[1], p[2], p[3]
            rgb_b = np.frombuffer(np.array(rgb, dtype=np.float32).tobytes(), dtype=np.uint8)
            pts.append([x, y, z])
            cols.append([rgb_b[2]/255.0, rgb_b[1]/255.0, rgb_b[0]/255.0])
        return np.concatenate([np.array(pts, dtype=np.float32),
                               np.array(cols, dtype=np.float32)], axis=1)


def main():
    parser = argparse.ArgumentParser(description='Interactive labeled point cloud collection')
    parser.add_argument('--output_dir', default='./my_raw_dataset')
    parser.add_argument('--camera_name', default='camera')
    parser.add_argument('--frames_per_object', type=int, default=10,
                       help='Frames to capture per object (vary viewpoint slightly between frames)')
    parser.add_argument('--sparse', type=int, default=2,
                       help='Take every Nth point (2 = half resolution, faster)')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between frames (seconds)')
    args = parser.parse_args()

    rclpy.init()
    node = InteractiveCapture(args.output_dir, args.camera_name)

    print("\n" + "="*60)
    print("  Interactive Dataset Collection Tool")
    print("="*60)
    print(f"  Output: {args.output_dir}")
    print(f"  Frames per object: {args.frames_per_object}")
    print(f"  Sparse factor: {args.sparse}")
    print()
    print("  Instructions:")
    print("    1. Place object in front of camera")
    print("    2. Type class name and press ENTER")
    print("    3. Slightly rotate/move object between frames")
    print("    4. Type 'q' to quit when done")
    print("="*60 + "\n")

    print("Waiting for camera...")
    if not node.wait_for_cloud(timeout=15.0):
        print("ERROR: No point cloud received. Is the camera running?")
        print("  ros2 launch orbbec_camera gemini.launch.py")
        node.destroy_node(); rclpy.shutdown(); sys.exit(1)
    print("Camera connected!\n")

    total_frames = 0
    classes_captured = {}

    try:
        while True:
            class_name = input("Class name (or 'q' to quit): ").strip().lower()
            if class_name == 'q':
                break
            if not class_name:
                print("  Please enter a class name.")
                continue

            class_dir = Path(args.output_dir) / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            existing = len(list(class_dir.glob('*.npy')))
            start_idx = existing + 1

            print(f"  Capturing {args.frames_per_object} frames for '{class_name}' "
                  f"(starting at #{start_idx})...")

            for i in range(args.frames_per_object):
                print(f"    Frame {i+1}/{args.frames_per_object}...", end=' ', flush=True)

                # Slight delay for user to adjust viewpoint
                if i > 0:
                    time.sleep(args.delay)

                node.wait_for_cloud(timeout=5.0)
                with node.lock:
                    if node.latest_cloud is None:
                        print("SKIP (no data)")
                        continue

                # Get latest cloud
                with node.lock:
                    msg = node.latest_cloud

                colored_pc = node.extract(msg, sparse=args.sparse)

                if len(colored_pc) < 100:
                    print(f"SKIP (only {len(colored_pc)} points)")
                    continue

                filename = class_dir / f'{class_name}_{start_idx + i:04d}.npy'
                np.save(str(filename), colored_pc)
                print(f"OK ({len(colored_pc)} pts)")
                total_frames += 1

            classes_captured[class_name] = classes_captured.get(class_name, 0) + args.frames_per_object
            print(f"  ✓ '{class_name}' done. Total frames: {total_frames}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    print("\n" + "="*60)
    print("  Collection Summary")
    print("="*60)
    for cls, count in classes_captured.items():
        print(f"  {cls}: {count} frames")
    print(f"  Total: {total_frames} frames")
    print(f"  Saved to: {args.output_dir}")
    print()
    print("  Next step - build training dataset:")
    print(f"    python scripts/build_rtabmap_dataset.py \\")
    print(f"        --raw_dir {args.output_dir} \\")
    print(f"        --output_dir ./data/my_real_dataset")
    print("="*60)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
