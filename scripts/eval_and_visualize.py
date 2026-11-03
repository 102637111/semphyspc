#!/usr/bin/env python3
"""
Color-PointMamba evaluation and visualization toolkit.

Performs:
  1. Full test set evaluation (overall + per-class accuracy)
  2. Confusion matrix
  3. t-SNE feature visualization
  4. Inference on real point clouds (Orbbec camera captures / PCD files)
  5. Attention/feature heatmap on sample point clouds

Usage:
    # Full evaluation on test set
    python scripts/eval_and_visualize.py \
        --config cfgs/finetune_modelnet_rgb.yaml \
        --ckpts experiments/.../ckpt-best.pth \
        --eval

    # t-SNE visualization only
    python scripts/eval_and_visualize.py \
        --config cfgs/finetune_modelnet_rgb.yaml \
        --ckpts experiments/.../ckpt-best.pth \
        --tsne

    # Inference on real captures
    python scripts/eval_and_visualize.py \
        --config cfgs/finetune_modelnet_rgb.yaml \
        --ckpts experiments/.../ckpt-best.pth \
        --infer_dir ./captured_pcd/

    # Everything at once
    python scripts/eval_and_visualize.py \
        --config cfgs/finetune_modelnet_rgb.yaml \
        --ckpts experiments/.../ckpt-best.pth \
        --eval --tsne --confusion --infer_dir ./captured_pcd/
"""

import os, sys, argparse, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools import builder
from utils import misc
from utils.config import get_config
from datasets.ModelNet40RGBDataset import ModelNet40RGB

CLASS_NAMES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox'
]

OUTPUT_DIR = './eval_results'


def load_model(args, config):
    """Load trained Color-PointMamba model."""
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts)
    base_model = base_model.cuda()
    base_model = torch.nn.DataParallel(base_model)
    base_model.eval()
    return base_model


def evaluate_full(model, config, dataloader, device):
    """Full test set evaluation with per-class accuracy."""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for _, _, data in dataloader:
            pts = data[0].to(device)
            labels = data[1].to(device)
            logits = model(pts)
            preds = logits.argmax(-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Overall accuracy
    overall_acc = (all_preds == all_labels).float().mean().item() * 100

    # Per-class accuracy
    n_classes = config.model.cls_dim
    per_class_acc = []
    for c in range(n_classes):
        mask = (all_labels == c)
        if mask.sum() > 0:
            acc = (all_preds[mask] == c).float().mean().item() * 100
        else:
            acc = float('nan')
        per_class_acc.append(acc)

    return overall_acc, per_class_acc, all_preds, all_labels


def plot_confusion_matrix(all_preds, all_labels, n_classes, save_path):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy(),
                          labels=range(n_classes))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto')

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(CLASS_NAMES[:n_classes], rotation=90, fontsize=7)
    ax.set_yticklabels(CLASS_NAMES[:n_classes], fontsize=7)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (Overall Acc: {100*(all_preds==all_labels).float().mean():.1f}%)')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Confusion matrix saved to {save_path}')


def plot_tsne(model, dataloader, device, n_classes, save_path, max_samples=2000):
    """t-SNE visualization of learned features."""
    from sklearn.manifold import TSNE

    features = []
    labels_list = []

    # Hook to extract features before classification head
    def hook_fn(module, input, output):
        features.append(output.detach().cpu())

    # Register hook on the avg pooling output (before cls_head)
    # PointMamba outputs pooled features internally; we hook cls_head_finetune input
    if hasattr(model, 'module'):
        handle = model.module.cls_head_finetune[0].register_forward_hook(hook_fn)
    else:
        handle = model.cls_head_finetune[0].register_forward_hook(hook_fn)

    with torch.no_grad():
        count = 0
        for _, _, data in dataloader:
            pts = data[0].to(device)
            labels = data[1]
            model(pts)
            labels_list.append(labels)
            count += len(labels)
            if count >= max_samples:
                break

    handle.remove()
    features = torch.cat(features)[:max_samples].numpy()
    labels_arr = torch.cat(labels_list)[:max_samples].numpy()

    print(f'  Running t-SNE on {len(features)} samples...')
    t0 = time.time()
    tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=30)
    embedded = tsne.fit_transform(features)
    print(f'  t-SNE done in {time.time()-t0:.1f}s')

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.cm.tab20 if n_classes <= 20 else plt.cm.gist_ncar
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=labels_arr,
                         cmap=cmap, s=2, alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(0, n_classes, max(1, n_classes//20)))
    cbar.ax.set_yticklabels([CLASS_NAMES[i] for i in range(0, n_classes, max(1, n_classes//20))],
                            fontsize=6)
    ax.set_title(f't-SNE of Color-PointMamba Features ({max_samples} samples)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  t-SNE saved to {save_path}')


def infer_on_real_data(model, input_dir, config, device):
    """Run inference on real captured point clouds."""
    from scripts.infer_color_pcd import load_point_cloud, preprocess_point_cloud, predict_single

    results = []
    files = sorted([f for f in os.listdir(input_dir)
                   if any(f.endswith(ext) for ext in ['.pcd', '.ply', '.npy'])])

    print(f'\n  Found {len(files)} files in {input_dir}')
    in_channels = getattr(config.model, 'in_channels', 3)
    model_expects_color = (in_channels >= 6)

    for f in files:
        filepath = os.path.join(input_dir, f)
        try:
            points, has_color = load_point_cloud(filepath)

            # Preprocess
            if len(points) > config.npoints:
                from datasets.ModelNet40RGBDataset import _pc_normalize_xyz
                # Simple FPS + normalize
                from scripts.infer_color_pcd import farthest_point_sample_numpy
                points = farthest_point_sample_numpy(points, config.npoints)

            # Ensure correct channels
            if model_expects_color and not has_color:
                if points.shape[1] < 6:
                    pad = np.zeros((len(points), 6 - points.shape[1]), dtype=np.float32)
                    points = np.concatenate([points, pad], axis=1)
            elif not model_expects_color:
                points = points[:, :3]

            # Normalize
            from datasets.ModelNet40RGBDataset import _pc_normalize_xyz
            points_copy = points.copy()
            points_copy = _pc_normalize_xyz(points_copy)
            pts_tensor = torch.from_numpy(points_copy).float().unsqueeze(0).to(device)

            if pts_tensor.size(1) > config.npoints:
                pts_tensor = misc.fps(pts_tensor, config.npoints)

            with torch.no_grad():
                logits = model(pts_tensor)
                probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

            top5_idx = np.argsort(probs)[-5:][::-1]
            top5 = [(CLASS_NAMES[i], float(probs[i])) for i in top5_idx]

            pred = top5_idx[0]
            conf = float(probs[pred])
            print(f'    {f}: {CLASS_NAMES[pred]} ({conf:.3f})')

            results.append({
                'file': f,
                'prediction': CLASS_NAMES[pred],
                'confidence': conf,
                'top5': top5
            })
        except Exception as e:
            print(f'    {f}: ERROR - {e}')

    return results


def main():
    parser = argparse.ArgumentParser(description='Color-PointMamba evaluation & visualization')
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpts', required=True)
    parser.add_argument('--eval', action='store_true', help='Full test evaluation')
    parser.add_argument('--confusion', action='store_true', help='Confusion matrix')
    parser.add_argument('--tsne', action='store_true', help='t-SNE visualization')
    parser.add_argument('--infer_dir', type=str, default=None,
                       help='Directory with real point cloud files for inference')
    parser.add_argument('--all', action='store_true', help='Run everything')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    if not any([args.eval, args.confusion, args.tsne, args.infer_dir, args.all]):
        print("Please specify at least one action: --eval, --confusion, --tsne, --infer_dir, or --all")
        return

    if args.all:
        args.eval = args.confusion = args.tsne = True

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup - mimic main.py args
    cfg_args = argparse.Namespace(
        config=args.config, use_gpu=torch.cuda.is_available(),
        distributed=False, local_rank=0, resume=False,
        ckpts=args.ckpts, experiment_path=args.output_dir,
    )
    config = get_config(cfg_args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading model from {args.ckpts}...')
    model = load_model(cfg_args, config)
    print(f'Model loaded. Device: {device}\n')

    # ---- Full Evaluation ----
    if args.eval or args.confusion or args.tsne:
        print('=== Building test dataloader ===')
        cfg_args.num_workers = 4
        # Set batch size (same as main.py)
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs
        if config.dataset.get('val'):
            config.dataset.val.others.bs = config.total_bs
        (_, test_loader) = builder.dataset_builder(cfg_args, config.dataset.test)
        print(f'  {len(test_loader)} batches\n')

    if args.eval:
        print('=== Full Test Evaluation ===')
        t0 = time.time()
        overall_acc, per_class_acc, all_preds, all_labels = evaluate_full(
            model, config, test_loader, device)
        print(f'  Overall Accuracy: {overall_acc:.2f}%')
        print(f'  Time: {time.time()-t0:.1f}s')

        print(f'\n  Per-class Accuracy:')
        for i, (name, acc) in enumerate(zip(CLASS_NAMES, per_class_acc)):
            bar = '█' * int(acc/5) if not np.isnan(acc) else '---'
            print(f'    {i:2d} {name:<15s}: {acc:5.1f}% {bar}')
        print(f'  Mean Class Accuracy: {np.nanmean(per_class_acc):.2f}%')

        # Save
        results = {
            'overall_acc': float(overall_acc),
            'per_class_acc': {n: float(a) for n, a in zip(CLASS_NAMES, per_class_acc)},
            'mean_class_acc': float(np.nanmean(per_class_acc))
        }
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  Results saved to {args.output_dir}/results.json\n')

    if args.confusion:
        print('=== Confusion Matrix ===')
        if 'all_preds' not in dir():
            _, _, all_preds, all_labels = evaluate_full(model, config, test_loader, device)
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(all_preds, all_labels, config.model.cls_dim, cm_path)

    if args.tsne:
        print('=== t-SNE Visualization ===')
        tsne_path = os.path.join(args.output_dir, 'tsne_features.png')
        plot_tsne(model, test_loader, device, config.model.cls_dim, tsne_path)

    if args.infer_dir:
        print(f'=== Real Point Cloud Inference: {args.infer_dir} ===')
        results = infer_on_real_data(model, args.infer_dir, config, device)
        with open(os.path.join(args.output_dir, 'inference_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  Results saved to {args.output_dir}/inference_results.json')

    print('\nDone!')


if __name__ == '__main__':
    main()
