#!/usr/bin/env python3
"""
Pre-compute ModelNet40RGB dataset into a single memory-mapped file using GPU FPS.

This creates:
  - train.npz: (points [N_train, 1024, 6], labels [N_train])
  - test.npz:  (points [N_test,  1024, 6], labels [N_test])

Usage:
    python scripts/precompute_rgb_dataset.py
"""

import os, sys, argparse, time, numpy as np, torch
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(__file__) + '/..')
from pointnet2_ops import pointnet2_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/ModelNet/modelnet40_colored')
    parser.add_argument('--output_dir', default='data/ModelNet/modelnet40_colored_processed')
    parser.add_argument('--npoints', type=int, default=1024)
    parser.add_argument('--batch_fps', type=int, default=128,
                       help='Batch size for GPU FPS')
    args = parser.parse_args()

    data_dir = args.data_dir
    modelnet_root = 'data/ModelNet/modelnet40_normal_resampled'
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load class names
    catfile = os.path.join(modelnet_root, 'modelnet40_shape_names.txt')
    cats = [line.rstrip() for line in open(catfile)]
    class_map = dict(zip(cats, range(len(cats))))
    print(f'{len(cats)} classes')

    for split in ['train', 'test']:
        print(f'\n=== Processing {split} set ===')

        # Read split file
        split_file = os.path.join(modelnet_root, f'modelnet40_{split}.txt')
        ids = [line.rstrip() for line in open(split_file)]
        names = ['_'.join(x.split('_')[:-1]) for x in ids]

        # Build file list
        files = [os.path.join(data_dir, n, i + '.npy') for n, i in zip(names, ids)]
        labels = [class_map[n] for n in names]

        # Filter existing
        valid = [(f, l) for f, l in zip(files, labels) if os.path.exists(f)]
        print(f'  {len(valid)}/{len(files)} files found')

        all_points = []
        all_labels = []

        for i in tqdm(range(0, len(valid), args.batch_fps), desc=f'  {split}'):
            batch_files = valid[i:i+args.batch_fps]
            batch_labels = [l for _, l in batch_files]

            # Load batch
            batch_data = []
            for f, _ in batch_files:
                pts = np.load(f).astype(np.float32)
                # Normalize RGB
                if pts[:, 3:6].max() > 1.5:
                    pts[:, 3:6] = pts[:, 3:6] / 255.0
                # Normalize XYZ
                xyz = pts[:, :3] - pts[:, :3].mean(0)
                m = np.sqrt((xyz**2).sum(1)).max()
                if m > 0:
                    xyz = xyz / m
                pts[:, :3] = xyz
                batch_data.append(pts)

            # Stack as tensors
            batch_tensors = [torch.from_numpy(d).float().cuda() for d in batch_data]
            # GPU FPS for each
            for j, pts_t in enumerate(batch_tensors):
                if pts_t.shape[0] > args.npoints:
                    pts_t = pts_t.unsqueeze(0)  # [1, N, 6]
                    xyz = pts_t[:, :, :3].contiguous()
                    fps_idx = pointnet2_utils.furthest_point_sample(xyz, args.npoints)
                    pts_t = pointnet2_utils.gather_operation(
                        pts_t.transpose(1, 2).contiguous(), fps_idx
                    ).transpose(1, 2).contiguous()
                    all_points.append(pts_t.squeeze(0).cpu().numpy())
                else:
                    all_points.append(batch_data[j][:args.npoints])

            all_labels.extend(batch_labels)

        # Save as compressed npz
        points_arr = np.stack(all_points).astype(np.float32)
        labels_arr = np.array(all_labels).astype(np.int64)
        out_path = os.path.join(args.output_dir, f'{split}.npz')
        np.savez_compressed(out_path, points=points_arr, labels=labels_arr)
        print(f'  Saved: {points_arr.shape}, {labels_arr.shape} → {out_path}')
        print(f'  Size: {os.path.getsize(out_path)/1024/1024:.1f} MB')


if __name__ == '__main__':
    main()
