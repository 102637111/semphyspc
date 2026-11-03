#!/usr/bin/env python3
"""
Generate colored ModelNet40 dataset from existing XYZ+normal data.

ModelNet40 .txt files already contain [N, 6] data: XYZ + normals.
This script maps normals to RGB colors, producing [N, 6] = XYZ + RGB.

Color mapping strategies:
  1. 'normal':  Map surface normals → RGB (n_x→R, n_y→G, n_z→B)
  2. 'coord':   Map XYZ coordinates → RGB (position-based, ignores normals)
  3. 'hybrid':  Blend normal (60%) + position (40%) colors

Usage:
    python scripts/generate_colored_modelnet.py --strategy normal
    python scripts/generate_colored_modelnet.py --strategy hybrid
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count


def normals_to_rgb(normals):
    """Map normals to RGB: n ∈ [-1,1]³ → RGB ∈ [0,1]³"""
    # Ensure normals are unit vectors
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1e-8
    normals = normals / norms
    rgb = (normals + 1.0) / 2.0
    return np.clip(rgb, 0.0, 1.0)


def coords_to_rgb(xyz):
    """Map XYZ coordinates to RGB by normalizing to [0,1] range."""
    min_vals = xyz.min(axis=0)
    max_vals = xyz.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals < 1e-8] = 1e-8
    rgb = (xyz - min_vals) / range_vals
    return np.clip(rgb, 0.0, 1.0)


def process_single_file(args_tuple):
    """Process one .txt file → save as colored .npy"""
    txt_path, output_path, strategy = args_tuple

    try:
        # Load data [N, 6] = XYZ + normals (or [N, 3] = XYZ only)
        data = np.loadtxt(txt_path, delimiter=',').astype(np.float32)

        xyz = data[:, :3]
        has_normals = (data.shape[1] >= 6)

        if strategy == 'normal':
            if has_normals:
                rgb = normals_to_rgb(data[:, 3:6])
            else:
                # Fallback: estimate from coordinates
                rgb = coords_to_rgb(xyz)
        elif strategy == 'coord':
            rgb = coords_to_rgb(xyz)
        elif strategy == 'hybrid':
            if has_normals:
                rgb_n = normals_to_rgb(data[:, 3:6])
            else:
                rgb_n = coords_to_rgb(xyz)
            rgb_c = coords_to_rgb(xyz)
            rgb = 0.6 * rgb_n + 0.4 * rgb_c
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        colored_pc = np.concatenate([xyz, rgb], axis=1).astype(np.float32)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, colored_pc)

        return output_path, None

    except Exception as e:
        return None, f"{txt_path}: {e}"


def main():
    parser = argparse.ArgumentParser(description='Generate colored ModelNet40 from XYZ+normal data')
    parser.add_argument('--data_dir', type=str,
                        default='./data/ModelNet/modelnet40_normal_resampled',
                        help='Path to ModelNet40 data (with .txt files)')
    parser.add_argument('--output_dir', type=str,
                        default='./data/ModelNet/modelnet40_colored',
                        help='Output directory for colored .npy files')
    parser.add_argument('--strategy', type=str, default='normal',
                        choices=['normal', 'coord', 'hybrid'],
                        help='Color generation strategy (normal=use existing normals)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Parallel workers')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        print(f"Error: {data_dir} not found!")
        sys.exit(1)

    # Collect .txt files from class subdirectories
    txt_files = sorted(data_dir.glob('*/*.txt'))
    print(f"Found {len(txt_files)} .txt files in {data_dir}")
    print(f"Strategy: {args.strategy}")
    print(f"Output: {output_dir}")
    print()

    # Build task list
    tasks = []
    for txt_path in txt_files:
        rel_path = txt_path.relative_to(data_dir)
        output_path = output_dir / rel_path.with_suffix('.npy')
        tasks.append((str(txt_path), str(output_path), args.strategy))

    workers = args.workers or max(1, cpu_count() - 2)
    success = 0
    errors = 0

    with Pool(workers) as pool:
        results = list(tqdm(pool.imap(process_single_file, tasks), total=len(tasks),
                           desc=f"Generating ({args.strategy})"))

    for path, err in results:
        if err:
            print(f"  Error: {err}")
            errors += 1
        else:
            success += 1

    print(f"\nDone! Success: {success}, Errors: {errors}")

    if success > 0:
        sample_path = [p for p, e in results if e is None][0]
        sample = np.load(sample_path)
        print(f"\nSample: {sample_path}")
        print(f"  Shape: {sample.shape}")
        print(f"  XYZ: [{sample[:,:3].min():.3f}, {sample[:,:3].max():.3f}]")
        print(f"  RGB: [{sample[:,3:6].min():.3f}, {sample[:,3:6].max():.3f}]")


if __name__ == '__main__':
    main()
