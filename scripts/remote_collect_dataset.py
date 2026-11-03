#!/usr/bin/env python3
"""
Remote dataset collection via ROS2 service — works over WiFi without data loss.

Principle:
  Camera is on robot, we are on local machine.
  PointCloud2 messages (~3MB) drop over WiFi.
  BUT ROS2 service calls (a few bytes) work reliably.
  Orbbec camera has a `/camera/save_point_cloud` service that saves .ply to disk.
  We call it remotely → robot saves file → we SCP/rsync the file.

Usage (run on LOCAL machine):
    python scripts/remote_collect_dataset.py \
        --class_name cup \
        --frames 10 \
        --robot_ip 10.0.0.5 \
        --output_dir ./my_raw_dataset

    # Or interactive mode:
    python scripts/remote_collect_dataset.py --interactive
"""

import os, sys, time, argparse, subprocess
from pathlib import Path


def call_save_service(camera_name='camera', timeout=5):
    """Call save_point_cloud service. Returns True on success."""
    import subprocess
    cmd = (
        f'ros2 service call /{camera_name}/save_point_cloud '
        f'std_srvs/srv/Empty "{{}}"'
    )
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def pull_files(robot_ip, robot_user, remote_dir, local_dir, class_name, max_files=None):
    """Pull .ply files from robot via SCP."""
    local_class_dir = Path(local_dir) / class_name
    local_class_dir.mkdir(parents=True, exist_ok=True)

    # List remote files
    list_cmd = f'ssh {robot_user}@{robot_ip} "ls -t {remote_dir}/colored_points_*.ply 2>/dev/null"'
    try:
        result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True, timeout=10)
        remote_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
    except Exception:
        print("  ERROR: Cannot list remote files. Check SSH connection.")
        return 0

    if max_files:
        remote_files = remote_files[:max_files]

    count = 0
    for rf in remote_files:
        local_path = local_class_dir / os.path.basename(rf)
        scp_cmd = f'scp -q {robot_user}@{robot_ip}:{rf} {local_path}'
        try:
            subprocess.run(scp_cmd, shell=True, check=True, timeout=15)
            count += 1
            # Delete remote file after successful transfer
            subprocess.run(
                f'ssh {robot_user}@{robot_ip} "rm -f {rf}"',
                shell=True, timeout=5)
        except Exception as e:
            print(f"  WARN: Failed to pull {rf}: {e}")

    return count


def interactive_collect(args):
    """Interactive collection mode."""
    print("\n" + "="*60)
    print("  Remote Dataset Collection (ROS2 Service + SCP)")
    print("="*60)
    print(f"  Robot: {args.robot_user}@{args.robot_ip}")
    print(f"  Remote PCD dir: {args.remote_pcd_dir}")
    print(f"  Local output: {args.output_dir}")
    print(f"  Frames per object: {args.frames}")
    print(f"  Camera: {args.camera_name}")
    print()
    print("  Instructions:")
    print("    1. Place object in front of robot's camera")
    print("    2. Type class name → auto-capture N frames")
    print("    3. Slightly rotate object between frames")
    print("    4. Type 'q' to finish, 'build' to build dataset")
    print("="*60 + "\n")

    # Test service
    print("Testing save_point_cloud service...")
    if call_save_service(args.camera_name):
        print("  Service OK. Cleaning up test file...\n")
    else:
        print("  WARN: Service call failed. Is camera running?")
        print("  Continue anyway? (y/n): ", end='')
        if input().lower() != 'y':
            sys.exit(1)

    total = 0
    while True:
        class_name = input("Class name (q=quit, build=build dataset): ").strip().lower()
        if class_name == 'q':
            break
        if class_name == 'build':
            build_dataset(args)
            continue
        if not class_name:
            continue

        print(f"\n  Collecting {args.frames} frames for '{class_name}'...")
        print(f"  Place object and press ENTER...", end='')
        input()

        for i in range(args.frames):
            print(f"    Frame {i+1}/{args.frames}...", end=' ', flush=True)
            ok = call_save_service(args.camera_name, timeout=5)
            if ok:
                print("OK")
            else:
                print("FAIL (service timeout)")
            # Wait between frames (user rotates object)
            if i < args.frames - 1:
                time.sleep(2.0)

        # Pull files
        print(f"  Pulling files from robot...")
        count = pull_files(
            args.robot_ip, args.robot_user, args.remote_pcd_dir,
            args.output_dir, class_name, max_files=args.frames)

        print(f"  ✓ '{class_name}': {count} frames saved\n")
        total += count

    print(f"\n  Total: {total} frames across all classes")
    print(f"  Local dataset: {args.output_dir}/")

    if total > 0:
        print(f"\n  Ready to build → run:")
        print(f"    python scripts/build_rtabmap_dataset.py \\")
        print(f"        --raw_dir {args.output_dir} \\")
        print(f"        --output_dir ./data/my_real_dataset")


def build_dataset(args):
    """Build .npz dataset from collected raw files."""
    print("\n  Building dataset...")
    cmd = [
        'python3', 'scripts/build_rtabmap_dataset.py',
        '--raw_dir', args.output_dir,
        '--output_dir', './data/my_real_dataset',
        '--npoints', str(args.npoints),
    ]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description='Remote colored point cloud collection via ROS2 service')
    parser.add_argument('--robot_ip', default='192.168.1.100', help='Robot IP')
    parser.add_argument('--robot_user', default='saw', help='SSH user')
    parser.add_argument('--remote_pcd_dir', default='~/point_cloud',
                       help='Remote directory where save_point_cloud writes files')
    parser.add_argument('--camera_name', default='camera')
    parser.add_argument('--output_dir', default='./my_raw_dataset')
    parser.add_argument('--frames', type=int, default=10,
                       help='Frames per object')
    parser.add_argument('--npoints', type=int, default=1024)
    parser.add_argument('--class_name', default=None,
                       help='Single class capture (non-interactive)')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    args = parser.parse_args()

    if args.class_name:
        print(f"Collecting {args.frames} frames for '{args.class_name}'...")
        for i in range(args.frames):
            print(f"  Frame {i+1}/{args.frames}...", end=' ', flush=True)
            call_save_service(args.camera_name)
            time.sleep(2.0)
        count = pull_files(args.robot_ip, args.robot_user, args.remote_pcd_dir,
                          args.output_dir, args.class_name, max_files=args.frames)
        print(f"Done: {count} frames")
    else:
        interactive_collect(args)


if __name__ == '__main__':
    main()
