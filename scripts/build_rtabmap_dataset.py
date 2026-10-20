#!/usr/bin/env python3
"""
Build a real colored point cloud dataset from RTAB-Map exports or Orbbec captures.

Expected directory structure:
    raw_dataset/
        airplane/           # Class name = folder name
            cloud_001.pcd   # RTAB-Map exported colored PCD
            cloud_002.pcd
            cloud_003.npy   # Or from Orbbec capture script
            ...
        chair/
            cloud_001.pcd
            ...
        cup/
            ...

This script:
  1. Loads all .pcd/.ply/.npy files from class subdirectories
  2. FPS-samples to target N points (GPU accelerated)
  3. Normalizes XYZ to unit sphere (RGB preserved)
  4. Saves as train/test .npz files

Usage:
    # Build dataset from raw captures
    python scripts/build_rtabmap_dataset.py \
        --raw_dir ./my_raw_dataset \
        --output_dir ./data/my_real_dataset \
        --npoints 1024

    # With train/test split ratio
    python scripts/build_rtabmap_dataset.py \
        --raw_dir ./my_raw_dataset \
        --output_dir ./data/my_real_dataset \
        --test_split 0.2
"""

import os, sys, argparse, numpy as np, torch, random
from tqdm import tqdm
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__) + '/..')
from pointnet2_ops import pointnet2_utils


def load_point_cloud_raw(filepath):
    """Load any supported point cloud format, return [N, C] numpy array."""
    ext = Path(filepath).suffix.lower()

    if ext == '.npy':
        data = np.load(filepath).astype(np.float32)
        return data[:, :6] if data.shape[1] >= 6 else data[:, :3]

    elif ext in ('.pcd', '.ply'):
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(filepath)
            xyz = np.asarray(pcd.points, dtype=np.float32)
            colors = np.asarray(pcd.colors, dtype=np.float32)

            if len(colors) == len(xyz) and colors.max() > 0:
                return np.concatenate([xyz, colors], axis=1)
            return xyz
        except ImportError:
            pass

        # Fallback: ASCII parser
        return _parse_ascii_pointcloud(filepath, ext)

    raise ValueError(f"Unsupported format: {ext}")


def _parse_ascii_pointcloud(filepath, ext):
    """Fallback ASCII PCD/PLY parser."""
    points, colors, in_header, in_data = [], [], True, False
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if in_header:
                if line.startswith('end_header'): in_header = False; in_data = True
                continue
            if in_data and line:
                parts = line.split()
                if len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    if len(parts) >= 6:
                        c = [float(x)/255.0 if float(x) > 1 else float(x) for x in parts[3:6]]
                        colors.append(c)

    pts = np.array(points, dtype=np.float32)
    if len(colors) == len(pts):
        return np.concatenate([pts, np.array(colors, dtype=np.float32)], axis=1)
    return pts


def process_cloud_gpu(pts, npoints, device):
    """GPU FPS + norm on a single cloud. Returns [npoints, C]."""
    # Normalize RGB to [0,1]
    if pts.shape[1] >= 6 and pts[:, 3:6].max() > 1.5:
        pts[:, 3:6] = pts[:, 3:6] / 255.0

    # Normalize XYZ to unit sphere
    xyz = pts[:, :3] - pts[:, :3].mean(0)
    m = np.sqrt((xyz**2).sum(1)).max()
    if m > 0: xyz /= m
    pts[:, :3] = xyz

    # GPU FPS
    pts_t = torch.from_numpy(pts).float().to(device).unsqueeze(0)  # [1, N, C]
    if pts_t.shape[1] > npoints:
        fps_idx = pointnet2_utils.furthest_point_sample(
            pts_t[:, :, :3].contiguous(), npoints)
        pts_t = pointnet2_utils.gather_operation(
            pts_t.transpose(1, 2).contiguous(), fps_idx
        ).transpose(1, 2).contiguous()
    elif pts_t.shape[1] < npoints:
        pad = torch.zeros(1, npoints - pts_t.shape[1], pts_t.shape[2], device=device)
        pts_t = torch.cat([pts_t, pad], dim=1)

    return pts_t.squeeze(0).cpu().numpy()


def build_dataset(raw_dir, output_dir, npoints, test_split, seed=42):
    """Build the full dataset from raw captures."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Discover classes
    class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"ERROR: No class subdirectories found in {raw_dir}")
        print(f"Expected structure: {raw_dir}/className/*.pcd")
        sys.exit(1)

    classes = {}
    class_files = {}

    for d in class_dirs:
        files = sorted(
            [str(f) for f in d.iterdir()
             if f.suffix.lower() in ('.pcd', '.ply', '.npy') and f.is_file()]
        )
        if files:
            classes[d.name] = len(classes)
            class_files[d.name] = files

    if not classes:
        print("ERROR: No point cloud files found in class directories!")
        sys.exit(1)

    print(f"Found {len(classes)} classes, {sum(len(v) for v in class_files.values())} total files:")
    for cls_name, files in class_files.items():
        print(f"  {cls_name}: {len(files)} files")

    # Save class mapping
    class_names = sorted(classes.keys(), key=lambda k: classes[k])
    with open(output_dir / 'classes.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"\nClass mapping saved to {output_dir / 'classes.txt'}")

    # Process all files
    all_train_pts, all_train_labels = [], []
    all_test_pts, all_test_labels = [], []

    random.seed(seed)
    np.random.seed(seed)

    for cls_name, files in tqdm(class_files.items(), desc="Processing classes"):
        cls_idx = classes[cls_name]

        for filepath in tqdm(files, desc=f"  {cls_name}", leave=False):
            try:
                pts = load_point_cloud_raw(filepath)
                processed = process_cloud_gpu(pts, npoints, device)

                if random.random() < test_split:
                    all_test_pts.append(processed)
                    all_test_labels.append(cls_idx)
                else:
                    all_train_pts.append(processed)
                    all_train_labels.append(cls_idx)
            except Exception as e:
                print(f"\n  ERROR processing {filepath}: {e}")

    # Save
    for split, pts_list, lbl_list in [
        ('train', all_train_pts, all_train_labels),
        ('test', all_test_pts, all_test_labels)]:

        if not pts_list:
            print(f"\n  WARNING: No samples in {split} set!")
            continue

        pts_arr = np.stack(pts_list).astype(np.float32)
        lbl_arr = np.array(lbl_list).astype(np.int64)
        out_path = output_dir / f'{split}.npz'
        np.savez_compressed(out_path, points=pts_arr, labels=lbl_arr)
        print(f"\n  {split}: {pts_arr.shape}, saved to {out_path}")
        print(f"  Size: {os.path.getsize(out_path)/1024/1024:.1f} MB")
        print(f"  Class distribution: {dict(zip(*np.unique(lbl_arr, return_counts=True)))}")

    print(f"\nDone! Dataset ready at {output_dir}")
    print(f"  Classes: {class_names}")
    print(f"  Train: {len(all_train_pts)}, Test: {len(all_test_pts)}")
    return class_names


def main():
    parser = argparse.ArgumentParser(description='Build colored dataset from RTAB-Map/Orbbec PCD captures')
    parser.add_argument('--raw_dir', required=True,
                       help='Directory with class subdirectories containing .pcd/.ply/.npy files')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for processed .npz files')
    parser.add_argument('--npoints', type=int, default=1024, help='Target points per cloud')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    build_dataset(args.raw_dir, args.output_dir, args.npoints, args.test_split, args.seed)


if __name__ == '__main__':
    main()
