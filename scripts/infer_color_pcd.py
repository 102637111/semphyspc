#!/usr/bin/env python3
"""
Unified colored point cloud inference script for Color-PointMamba.

Supports:
  - RTAB-Map .pcd files (colored or uncolored)
  - .npy files from Orbbec capture script ([N, 3] or [N, 6])
  - .ply files from Orbbec SDK save service
  - Directories containing multiple files

Usage:
    # Single file inference
    python scripts/infer_color_pcd.py \\
        --config cfgs/finetune_modelnet_rgb.yaml \\
        --ckpts finetune_modelnet.pth \\
        --input /path/to/colored_cloud.pcd

    # Directory inference
    python scripts/infer_color_pcd.py \\
        --config cfgs/finetune_modelnet_rgb.yaml \\
        --ckpts finetune_modelnet.pth \\
        --input /path/to/pcd_folder/

    # With custom number of points
    python scripts/infer_color_pcd.py \\
        --config cfgs/finetune_modelnet_rgb.yaml \\
        --ckpts finetune_modelnet.pth \\
        --input ./captured_pcd/colored_pc_0001.npy \\
        --npoints 1024

    # Sparse inference (take 1/Nth of points): useful for dense Orbbec clouds
    python scripts/infer_color_pcd.py \\
        --config cfgs/finetune_modelnet_rgb.yaml \\
        --ckpts finetune_modelnet.pth \\
        --input ./captured_pcd/ --sparse 4
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools import builder
from utils import misc
from utils.config import get_config

CLASS_NAMES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox'
]


def pc_normalize(pc):
    """Normalize XYZ coordinates to zero-mean and unit sphere."""
    xyz = pc[:, :3]
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    if m > 0:
        xyz = xyz / m
    pc[:, :3] = xyz
    return pc


def farthest_point_sample_numpy(point, npoint):
    """FPS sampling on XYZ coordinates, returns all channels."""
    N = point.shape[0]
    xyz = point[:, :3]
    centroids = np.zeros((npoint,), dtype=np.int64)
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)

    return point[centroids.astype(np.int32)]


def load_pcd_file(pcd_path):
    """Load a point cloud from .pcd file (supports colored and uncolored)."""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points).astype(np.float32)
        colors = np.asarray(pcd.colors).astype(np.float32)

        if len(points) == 0:
            raise ValueError(f"No points found in {pcd_path}")

        if len(colors) == len(points) and colors.max() > 0:
            # Has valid colors
            colored_pc = np.concatenate([points, colors], axis=1)
            has_color = True
        else:
            colored_pc = points
            has_color = False

        print(f"Loaded {len(points)} points from {pcd_path} (color: {has_color})")
        return colored_pc, has_color

    except ImportError:
        print("open3d not available, trying to read PCD as ASCII...")
        return _load_pcd_ascii(pcd_path)


def _load_pcd_ascii(pcd_path):
    """Fallback: parse ASCII PCD file manually."""
    points = []
    colors = []
    in_data = False
    has_color = False
    fields = []

    with open(pcd_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('FIELDS'):
                fields = line.split()[1:]
                if 'rgb' in fields or 'r' in fields:
                    has_color = True
            elif line.startswith('DATA'):
                in_data = True
                continue
            elif in_data and line:
                vals = line.split()
                if len(vals) >= 3:
                    x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                    points.append([x, y, z])
                    if has_color and len(vals) >= 6:
                        r = float(vals[3]) / 255.0 if float(vals[3]) > 1 else float(vals[3])
                        g = float(vals[4]) / 255.0 if float(vals[4]) > 1 else float(vals[4])
                        b = float(vals[5]) / 255.0 if float(vals[5]) > 1 else float(vals[5])
                        colors.append([r, g, b])

    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32) if colors else np.zeros_like(points)

    if len(colors) != len(points):
        colors = np.zeros_like(points)
        has_color = False

    colored_pc = np.concatenate([points, colors], axis=1) if has_color else points
    print(f"Loaded {len(points)} points from {pcd_path} (color: {has_color})")
    return colored_pc, has_color


def load_npy_file(npy_path):
    """Load point cloud from .npy file [N, 3] or [N, 6]."""
    data = np.load(npy_path).astype(np.float32)
    has_color = data.shape[1] >= 6

    if data.shape[1] == 3:
        print(f"Loaded {len(data)} XYZ points from {npy_path} (no color)")
    elif data.shape[1] >= 6:
        # Ensure RGB in [0, 1] range
        if data[:, 3:6].max() > 1.0:
            data[:, 3:6] = data[:, 3:6] / 255.0
        print(f"Loaded {len(data)} colored points from {npy_path}")
    else:
        print(f"Loaded {len(data)} points from {npy_path} ({data.shape[1]} channels)")

    return data[:, :6] if data.shape[1] >= 6 else data, has_color


def load_ply_file(ply_path):
    """Load point cloud from .ply file."""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points).astype(np.float32)
        colors = np.asarray(pcd.colors).astype(np.float32)

        if len(colors) == len(points) and colors.max() > 0:
            colored_pc = np.concatenate([points, colors], axis=1)
            has_color = True
        else:
            colored_pc = points
            has_color = False

        print(f"Loaded {len(points)} points from {ply_path} (color: {has_color})")
        return colored_pc, has_color
    except ImportError:
        print("open3d not available for PLY loading")
        raise


def load_point_cloud(file_path):
    """Auto-detect format and load point cloud."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.npy':
        return load_npy_file(file_path)
    elif ext == '.pcd':
        return load_pcd_file(file_path)
    elif ext == '.ply':
        return load_ply_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def preprocess_point_cloud(points, npoints=1024, sparse=1, has_color=False, model_expects_color=True):
    """
    Preprocess point cloud for model inference.

    Args:
        points: [N, 3] or [N, 6] numpy array
        npoints: target number of points
        sparse: take every Nth point before FPS (for dense clouds)
        has_color: whether the loaded point cloud has RGB
        model_expects_color: whether the model expects 6-channel input

    Returns:
        torch tensor [1, npoints, C]
    """
    # Sparse sampling for dense clouds
    if sparse > 1:
        points = points[::sparse]

    # FPS downsampling
    if len(points) > npoints:
        points = farthest_point_sample_numpy(points, npoints)
    elif len(points) < npoints:
        # Pad with zeros
        pad_len = npoints - len(points)
        if points.shape[1] >= 6:
            padding = np.zeros((pad_len, points.shape[1]), dtype=np.float32)
        else:
            padding = np.zeros((pad_len, 3), dtype=np.float32)
        points = np.concatenate([points, padding])

    # Normalize XYZ
    points = pc_normalize(points)

    # Handle color channels
    if model_expects_color and not has_color:
        # Model needs 6 channels but data only has XYZ → pad with zeros
        if points.shape[1] < 6:
            color_pad = np.zeros((len(points), 6 - points.shape[1]), dtype=np.float32)
            points = np.concatenate([points, color_pad], axis=1)
            print("  Warning: Model expects color but input has none. Padding with zeros.")
    elif not model_expects_color and has_color:
        # Model expects 3 channels but data has color → strip to XYZ
        points = points[:, :3]
        print("  Info: Stripping color channels (model uses XYZ only).")
    elif model_expects_color and has_color and points.shape[1] < 6:
        # Partial color data
        color_pad = np.zeros((len(points), 6 - points.shape[1]), dtype=np.float32)
        points = np.concatenate([points, color_pad], axis=1)

    # Ensure exactly right channels
    target_channels = 6 if model_expects_color else 3
    points = points[:, :target_channels]

    points = torch.from_numpy(points).float().unsqueeze(0)
    return points


def load_model(args, config):
    """Load Color-PointMamba model."""
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts)
    base_model = base_model.cuda()
    base_model = torch.nn.DataParallel(base_model)
    base_model.eval()
    return base_model


def get_top5(logits):
    """Get top-5 predictions from logits."""
    top5_idx = np.argsort(logits)[-5:][::-1]
    top5 = []
    for idx in top5_idx:
        top5.append((CLASS_NAMES[idx], logits[idx]))
    return top5


def predict_single(base_model, points, config, use_gpu=True):
    """Run inference on a single point cloud."""
    npoints = config.npoints

    if points.size(1) > npoints:
        points = misc.fps(points, npoints)

    if use_gpu:
        points = points.cuda()

    with torch.no_grad():
        logits = base_model(points)
        pred = logits.argmax(-1).item()
        confidence = torch.softmax(logits, dim=-1).max().item()
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    return pred, confidence, probs


def process_file(base_model, file_path, config, args):
    """Process a single file and print results."""
    print(f"\n{'='*60}")
    print(f"File: {file_path}")
    print(f"{'='*60}")

    # Load
    points, has_color = load_point_cloud(file_path)
    print(f"  Original: {len(points)} points, {points.shape[1]} channels, color={has_color}")

    # Check if model expects color
    in_channels = getattr(config.model, 'in_channels', 3)
    model_expects_color = (in_channels >= 6)

    # Preprocess
    points_tensor = preprocess_point_cloud(
        points,
        npoints=config.npoints,
        sparse=args.sparse,
        has_color=has_color,
        model_expects_color=model_expects_color
    )
    print(f"  Preprocessed: {points_tensor.shape[1]} points, {points_tensor.shape[2]} channels")

    # Predict
    pred, confidence, probs = predict_single(base_model, points_tensor, config, args.use_gpu)

    # Results
    print(f"\n  Predicted: {CLASS_NAMES[pred]} (class {pred})")
    print(f"  Confidence: {confidence:.4f}")

    print(f"\n  Top-5 predictions:")
    top5 = get_top5(probs)
    for i, (name, prob) in enumerate(top5):
        bar = '█' * int(prob * 40) + '░' * (40 - int(prob * 40))
        print(f"    {i + 1}. {name:<15s}  {prob:.4f}  {bar}")

    return {
        'filename': os.path.basename(file_path),
        'num_points': len(points),
        'has_color': has_color,
        'predicted_class': CLASS_NAMES[pred],
        'predicted_index': pred,
        'confidence': confidence,
        'top5': [(name, float(prob)) for name, prob in top5],
    }


def main():
    parser = argparse.ArgumentParser(description='Color-PointMamba inference on point cloud files')
    parser.add_argument('--config', type=str, required=True,
                        help='YAML config file (use *_rgb.yaml for color model)')
    parser.add_argument('--ckpts', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file (.pcd/.ply/.npy) or directory')
    parser.add_argument('--npoints', type=int, default=1024,
                        help='Number of points for FPS sampling')
    parser.add_argument('--sparse', type=int, default=1,
                        help='Take every Nth point (for dense clouds)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output results to file (JSON)')
    args = parser.parse_args()

    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.local_rank = 0

    config = get_config(args)
    config.npoints = args.npoints

    # Check channels match
    in_channels = getattr(config.model, 'in_channels', 3)
    print(f"\nModel config: in_channels={in_channels}")
    print(f"GPU available: {args.use_gpu}")
    print(f"Target points: {config.npoints}")

    # Load model
    print(f"\nLoading model from {args.ckpts}...")
    base_model = load_model(args, config)
    print("Model loaded.")

    results = []

    if os.path.isfile(args.input):
        result = process_file(base_model, args.input, config, args)
        results.append(result)

    elif os.path.isdir(args.input):
        print(f"\n=== Processing directory: {args.input} ===")
        files = sorted([
            f for f in os.listdir(args.input)
            if any(f.endswith(ext) for ext in ['.pcd', '.ply', '.npy'])
        ])

        if not files:
            print(f"No point cloud files found in {args.input}")
            sys.exit(1)

        print(f"Found {len(files)} files")
        for f in files:
            try:
                result = process_file(base_model, os.path.join(args.input, f), config, args)
                results.append(result)
            except Exception as e:
                print(f"  Error processing {f}: {e}")

        # Summary
        print(f"\n{'='*60}")
        print(f"Summary: {len(results)}/{len(files)} files processed successfully")
        print(f"{'='*60}")
        for r in results:
            print(f"  {r['filename']:<30s} → {r['predicted_class']:<15s} ({r['confidence']:.3f})")

    else:
        print(f"Error: {args.input} not found.")
        sys.exit(1)

    # Save results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
