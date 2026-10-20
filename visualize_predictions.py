import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.switch_backend('Agg')
from tools import builder
from utils import misc
from utils.config import get_config
import argparse
import json

CLASS_NAMES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 
    'wardrobe', 'xbox'
]

COLOR_MAP = plt.cm.get_cmap('tab20', 40)

def load_model(args, config):
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts)
    if args.use_gpu:
        base_model = base_model.cuda()
    base_model = torch.nn.DataParallel(base_model)
    base_model.eval()
    return base_model

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def visualize_single_pointcloud(points, true_label, pred_label, confidence, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    title_color = 'green' if true_label == pred_label else 'red'
    title = f'True: {true_label} | Predicted: {pred_label} | Confidence: {confidence:.2f}'
    
    axes[0].scatter(points[:, 0], points[:, 1], c='blue', s=1)
    axes[0].set_title('XY View')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_aspect('equal')
    
    axes[1].scatter(points[:, 0], points[:, 2], c='red', s=1)
    axes[1].set_title('XZ View')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_aspect('equal')
    
    axes[2].scatter(points[:, 1], points[:, 2], c='green', s=1)
    axes[2].set_title('YZ View')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].set_aspect('equal')
    
    fig.suptitle(title, color=title_color, fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_with_o3d(points, true_label, pred_label, confidence, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    if points.shape[1] > 3:
        colors = plt.cm.viridis(points[:, 3])[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        color = np.array([0.5, 0.5, 0.5])
        if true_label == pred_label:
            color = np.array([0.2, 0.8, 0.2])
        else:
            color = np.array([0.8, 0.2, 0.2])
        pcd.paint_uniform_color(color)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600, visible=False)
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = 3.0
    
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    
    vis.capture_screen_image(save_path)
    vis.destroy_window()

def generate_sample_visualizations(base_model, test_dataloader, config, args):
    npoints = config.npoints
    os.makedirs(args.output_dir, exist_ok=True)
    
    sample_count = 0
    correct_count = 0
    wrong_count = 0
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
            
            points_fps = misc.fps(points, npoints)
            logits = base_model(points_fps)
            
            pred = logits.argmax(-1).view(-1)
            confidence = torch.softmax(logits, dim=-1).max(dim=-1)[0]
            
            for i in range(points.shape[0]):
                if sample_count >= args.num_samples:
                    return
                
                true_label = CLASS_NAMES[label[i].item()]
                pred_label = CLASS_NAMES[pred[i].item()]
                conf = confidence[i].item()
                
                point_np = points[i].cpu().numpy()
                if point_np.shape[1] > 3:
                    point_np = point_np[:, :3]
                
                is_correct = true_label == pred_label
                
                if is_correct and correct_count < args.num_samples // 2:
                    save_path = os.path.join(args.output_dir, f'correct_{correct_count:03d}_{true_label}.png')
                    visualize_single_pointcloud(point_np, true_label, pred_label, conf, save_path)
                    correct_count += 1
                    sample_count += 1
                elif not is_correct and wrong_count < args.num_samples // 2:
                    save_path = os.path.join(args.output_dir, f'wrong_{wrong_count:03d}_{true_label}_as_{pred_label}.png')
                    visualize_single_pointcloud(point_np, true_label, pred_label, conf, save_path)
                    wrong_count += 1
                    sample_count += 1
    
    print(f"\nGenerated {correct_count} correct and {wrong_count} wrong sample visualizations")

def generate_confusion_matrix_visualization(test_pred, test_label, args):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(test_label, test_pred)
    
    plt.figure(figsize=(25, 20))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("Generated confusion matrix visualization")

def generate_class_accuracy_bar(test_pred, test_label, args):
    class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        if (test_label == i).sum() > 0:
            class_acc[name] = (test_pred[test_label == i] == i).sum() / (test_label == i).sum() * 100.
        else:
            class_acc[name] = 0.
    
    sorted_acc = sorted(class_acc.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_acc]
    accs = [x[1] for x in sorted_acc]
    
    colors = ['green' if a >= 90 else 'orange' if a >= 70 else 'red' for a in accs]
    
    plt.figure(figsize=(15, 10))
    plt.barh(names, accs, color=colors)
    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.title('Per-class Accuracy', fontsize=16)
    plt.xlim(0, 100)
    plt.gca().invert_yaxis()
    
    for i, v in enumerate(accs):
        plt.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'class_accuracy_bar.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("Generated class accuracy bar chart")

def generate_prediction_histogram(test_pred, test_label, args):
    predictions = [CLASS_NAMES[p] for p in test_pred]
    labels = [CLASS_NAMES[l] for l in test_label]
    
    from collections import Counter
    pred_counts = Counter(predictions)
    label_counts = Counter(labels)
    
    plt.figure(figsize=(15, 8))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    
    pred_vals = [pred_counts.get(name, 0) for name in CLASS_NAMES]
    label_vals = [label_counts.get(name, 0) for name in CLASS_NAMES]
    
    plt.bar(x - width/2, label_vals, width, label='Ground Truth')
    plt.bar(x + width/2, pred_vals, width, label='Predictions')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Ground Truth vs Predictions Distribution', fontsize=16)
    plt.xticks(x, CLASS_NAMES, rotation=90, fontsize=8)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'prediction_histogram.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("Generated prediction distribution histogram")

def generate_3d_scatter_plot(base_model, test_dataloader, config, args):
    from sklearn.manifold import TSNE
    
    features = []
    labels = []
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            if idx >= 5:
                break
            points = data[0].cuda()
            label = data[1].cuda()
            
            points_fps = misc.fps(points, config.npoints)
            
            if hasattr(base_model.module, 'get_feature'):
                feat = base_model.module.get_feature(points_fps)
            else:
                logits = base_model(points_fps)
                feat = logits
            
            features.append(feat.cpu().numpy())
            labels.extend(label.cpu().numpy())
    
    features = np.concatenate(features)
    labels = np.array(labels)
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    for i, name in enumerate(CLASS_NAMES[:10]):
        mask = labels == i
        if mask.sum() > 0:
            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], label=name, alpha=0.7, s=50)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('t-SNE Feature Visualization', fontsize=16)
    plt.savefig(os.path.join(args.output_dir, 'tsne_features.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("Generated t-SNE feature visualization")

def test_and_visualize(base_model, test_dataloader, config, args):
    test_pred = []
    test_label = []
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
            
            points_fps = misc.fps(points, config.npoints)
            logits = base_model(points_fps)
            
            pred = logits.argmax(-1).view(-1)
            target = label.view(-1)
            
            test_pred.append(pred.detach().cpu().numpy())
            test_label.append(target.detach().cpu().numpy())
    
    test_pred = np.concatenate(test_pred)
    test_label = np.concatenate(test_label)
    
    acc = (test_pred == test_label).sum() / float(test_label.shape[0]) * 100.
    print(f"Overall Accuracy: {acc:.4f}%")
    
    generate_confusion_matrix_visualization(test_pred, test_label, args)
    generate_class_accuracy_bar(test_pred, test_label, args)
    generate_prediction_histogram(test_pred, test_label, args)
    generate_sample_visualizations(base_model, test_dataloader, config, args)
    
    try:
        generate_3d_scatter_plot(base_model, test_dataloader, config, args)
    except Exception as e:
        print(f"t-SNE visualization skipped: {e}")
    
    results = {
        'accuracy': float(acc),
        'total_samples': len(test_label),
        'correct': int((test_pred == test_label).sum()),
        'wrong': int((test_pred != test_label).sum())
    }
    
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='yaml config file')
    parser.add_argument('--ckpts', type=str, required=True, help='checkpoint path')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='output directory')
    parser.add_argument('--num_samples', type=int, default=20, help='number of sample visualizations')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--exp_name', type=str, default='visualize')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.local_rank = 0
    args.experiment_path = os.path.join('./experiments', args.exp_name)
    
    config = get_config(args)
    config.dataset.test.others.bs = 8
    
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    
    base_model = load_model(args, config)
    
    print(f"Generating visualizations with checkpoint: {args.ckpts}")
    print(f"Output directory: {args.output_dir}")
    print(f"GPU available: {args.use_gpu}")
    
    test_and_visualize(base_model, test_dataloader, config, args)
    
    print(f"\n=== Visualizations saved to {args.output_dir} ===")

if __name__ == '__main__':
    main()