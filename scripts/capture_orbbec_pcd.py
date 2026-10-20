#!/usr/bin/env python3
"""
Capture colored point clouds from Orbbec Gemini camera via ROS2.

Subscribes to:
    /camera/depth_registered/points  (sensor_msgs/PointCloud2 with x,y,z,r,g,b)

Saves colored point clouds as .npy files [N, 6] (XYZ + RGB normalized to [0,1])
and optionally as .ply files for visualization.

Usage:
    # Capture a single frame and save
    ros2 run point_mamba capture_orbbec_pcd --output_dir ./captured_pcd

    # Capture multiple frames with a delay
    ros2 run point_mamba capture_orbbec_pcd --output_dir ./captured_pcd --num_frames 10 --delay 2.0

    # Capture continuously on keypress
    ros2 run point_mamba capture_orbbec_pcd --output_dir ./captured_pcd --interactive

    # Specify camera namespace
    ros2 run point_mamba capture_orbbec_pcd --camera_name my_camera
"""

import os
import sys
import time
import argparse
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from rclpy.qos import qos_profile_sensor_data


class OrbbecPointCloudCapture(Node):
    """ROS2 node for capturing colored point clouds from Orbbec Gemini camera."""

    def __init__(self, output_dir, camera_name='camera', save_ply=False):
        super().__init__('orbbec_pcd_capture')

        self.output_dir = output_dir
        self.save_ply = save_ply
        self.latest_cloud = None
        self.lock = threading.Lock()
        self.frame_count = 0

        os.makedirs(self.output_dir, exist_ok=True)

        # Subscribe to colored point cloud topic
        topic = f'/{camera_name}/depth_registered/points'
        self.subscription = self.create_subscription(
            PointCloud2,
            topic,
            self.cloud_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(f'Subscribed to {topic}')
        self.get_logger().info(f'Waiting for point cloud messages...')

    def cloud_callback(self, msg):
        """Store latest point cloud."""
        with self.lock:
            self.latest_cloud = msg

    def wait_for_cloud(self, timeout=10.0):
        """Wait until a point cloud is received."""
        start = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            with self.lock:
                if self.latest_cloud is not None:
                    return True
            if time.time() - start > timeout:
                self.get_logger().error(f'Timeout waiting for point cloud after {timeout}s')
                return False
        return False

    def extract_colored_points(self, cloud_msg):
        """
        Extract XYZ+RGB from PointCloud2 message.
        Returns numpy array of shape [N, 6] with XYZ in meters and RGB in [0, 1].
        """
        # Read points from PointCloud2
        gen = point_cloud2.read_points(
            cloud_msg,
            field_names=['x', 'y', 'z', 'rgb'],
            skip_nans=True
        )

        points_list = []
        colors_list = []

        for point in gen:
            x, y, z, rgb_packed = point[0], point[1], point[2], point[3]

            # Unpack RGB from float32 (standard PCL packing)
            # rgb_packed is a float32 whose bytes represent [R, G, B, _] as uint8
            rgb_bytes = np.frombuffer(np.array(rgb_packed, dtype=np.float32).tobytes(), dtype=np.uint8)
            r = rgb_bytes[2].astype(np.float32) / 255.0  # Red is at byte 2
            g = rgb_bytes[1].astype(np.float32) / 255.0  # Green is at byte 1
            b = rgb_bytes[0].astype(np.float32) / 255.0  # Blue is at byte 0

            points_list.append([x, y, z])
            colors_list.append([r, g, b])

        points = np.array(points_list, dtype=np.float32)
        colors = np.array(colors_list, dtype=np.float32)

        # Combine into [N, 6] array
        colored_pc = np.concatenate([points, colors], axis=1)

        return colored_pc

    def save_cloud(self, cloud_msg, filename_prefix='colored_pc'):
        """Save the point cloud to .npy and optionally .ply."""
        colored_pc = self.extract_colored_points(cloud_msg)
        self.frame_count += 1

        # Save as .npy
        npy_path = os.path.join(self.output_dir, f'{filename_prefix}_{self.frame_count:04d}.npy')
        np.save(npy_path, colored_pc)
        self.get_logger().info(f'Saved {colored_pc.shape[0]} colored points to {npy_path}')

        # Optionally save as .ply
        if self.save_ply:
            self._save_ply(colored_pc, filename_prefix)

        return colored_pc

    def _save_ply(self, colored_pc, filename_prefix):
        """Save colored point cloud as PLY file."""
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(colored_pc[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(colored_pc[:, 3:6])

            ply_path = os.path.join(self.output_dir, f'{filename_prefix}_{self.frame_count:04d}.ply')
            o3d.io.write_point_cloud(ply_path, pcd)
            self.get_logger().info(f'Saved PLY to {ply_path}')
        except ImportError:
            self.get_logger().warn('open3d not available, skipping PLY save')
            # Fallback: write simple ASCII PLY
            ply_path = os.path.join(self.output_dir, f'{filename_prefix}_{self.frame_count:04d}.ply')
            self._write_ascii_ply(colored_pc, ply_path)

    def _write_ascii_ply(self, colored_pc, filename):
        """Write a simple ASCII PLY file as fallback."""
        N = colored_pc.shape[0]
        with open(filename, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {N}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            for i in range(N):
                x, y, z = colored_pc[i, 0], colored_pc[i, 1], colored_pc[i, 2]
                r = int(colored_pc[i, 3] * 255)
                g = int(colored_pc[i, 4] * 255)
                b = int(colored_pc[i, 5] * 255)
                f.write(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n')
        self.get_logger().info(f'Saved ASCII PLY to {filename}')


def interactive_capture(node):
    """Capture frames on Enter key press. Press 'q' to quit."""
    print("\n=== Interactive Capture Mode ===")
    print("Press ENTER to capture a frame")
    print("Type 'q' + ENTER to quit")
    print("================================\n")

    frame_num = 0
    while rclpy.ok():
        user_input = input(f"[Frame {frame_num + 1}] Press ENTER to capture... ")

        if user_input.lower() == 'q':
            print("Quitting...")
            break

        # Spin to get latest frame
        rclpy.spin_once(node, timeout_sec=0.5)

        with node.lock:
            if node.latest_cloud is not None:
                node.save_cloud(node.latest_cloud)
                frame_num += 1
            else:
                print("  No point cloud received yet. Waiting...")
                if node.wait_for_cloud(timeout=5.0):
                    node.save_cloud(node.latest_cloud)
                    frame_num += 1
                else:
                    print("  Still no point cloud. Check camera connection.")


def batch_capture(node, num_frames, delay):
    """Capture multiple frames with a delay between captures."""
    print(f"\n=== Batch Capture: {num_frames} frames with {delay}s delay ===")

    for i in range(num_frames):
        print(f"Capturing frame {i + 1}/{num_frames}...")

        # Spin to get latest frame
        rclpy.spin_once(node, timeout_sec=0.5)

        with node.lock:
            if node.latest_cloud is not None:
                node.save_cloud(node.latest_cloud)
            else:
                print("  Waiting for point cloud...")
                if node.wait_for_cloud(timeout=5.0):
                    node.save_cloud(node.latest_cloud)
                else:
                    print("  Skipping frame - no data")

        if i < num_frames - 1:
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description='Capture colored point clouds from Orbbec Gemini camera')
    parser.add_argument('--output_dir', type=str, default='./captured_pcd',
                        help='Output directory for saved point clouds')
    parser.add_argument('--camera_name', type=str, default='camera',
                        help='Camera namespace (default: camera)')
    parser.add_argument('--num_frames', type=int, default=1,
                        help='Number of frames to capture (batch mode)')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Delay between captures in seconds (batch mode)')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode: press ENTER to capture each frame')
    parser.add_argument('--save_ply', action='store_true',
                        help='Also save as PLY format (requires open3d)')
    args = parser.parse_args()

    rclpy.init(args=None)

    node = OrbbecPointCloudCapture(
        output_dir=args.output_dir,
        camera_name=args.camera_name,
        save_ply=args.save_ply
    )

    try:
        if args.interactive:
            interactive_capture(node)
        elif args.num_frames > 1:
            batch_capture(node, args.num_frames, args.delay)
        else:
            # Single frame capture
            print(f"\n=== Capturing single frame to {args.output_dir} ===")
            print("Waiting for point cloud... (Ctrl+C to cancel)")
            if node.wait_for_cloud(timeout=10.0):
                node.save_cloud(node.latest_cloud)
                print("Done!")
            else:
                print("Failed to capture point cloud.")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\nCapture interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
