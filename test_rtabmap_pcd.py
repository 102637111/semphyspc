import os
import torch
import numpy as np
import open3d as o3d
from tools import builder
from utils import misc
from utils.config import get_config
import argparse

CLASS_NAMES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 
    'wardrobe', 'xbox'
]

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

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def load_pcd_file(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    print(f"Loaded {len(points)} points from {pcd_path}")
    return points

def preprocess_point_cloud(points, npoints=1024):
    if len(points) > npoints:
        points = farthest_point_sample(points, npoints)
    elif len(points) < npoints:
        padding = np.zeros((npoints - len(points), 3))
        points = np.concatenate([points, padding])
    
    points = pc_normalize(points)
    points = torch.from_numpy(points).float().unsqueeze(0)
    return points

def load_model(args, config):
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts)
    if args.use_gpu:
        base_model = base_model.cuda()
    base_model = torch.nn.DataParallel(base_model)
    base_model.eval()
    return base_model

def predict_single_point_cloud(base_model, points, config):
    npoints = config.npoints
    
    if points.size(1) > npoints:
        points = misc.fps(points, npoints)
    
    if args.use_gpu:
        points = points.cuda()
    
    with torch.no_grad():
        logits = base_model(points)
        pred = logits.argmax(-1).item()
        confidence = torch.softmax(logits, dim=-1).max().item()
    
    return pred, confidence, logits.squeeze().cpu().numpy()

def predict_pcd_folder(base_model, folder_path, config, args):
    pcd_files = [f for f in os.listdir(folder_path) if f.endswith('.pcd')]
    results = []
    
    for pcd_file in pcd_files:
        pcd_path = os.path.join(folder_path, pcd_file)
        try:
            raw_points = load_pcd_file(pcd_path)
            points = preprocess_point_cloud(raw_points, config.npoints)
            pred, confidence, logits = predict_single_point_cloud(base_model, points, config)
            
            result = {
                'filename': pcd_file,
                'num_points': len(raw_points),
                'predicted_class': CLASS_NAMES[pred],
                'predicted_index': pred,
                'confidence': confidence,
                'top5': get_top5(logits)
            }
            results.append(result)
            
            print(f"\n{pcd_file}:")
            print(f"  Points: {len(raw_points)}")
            print(f"  Predicted: {CLASS_NAMES[pred]} (confidence: {confidence:.4f})")
            print(f"  Top 5: {result['top5']}")
            
        except Exception as e:
            print(f"Error processing {pcd_file}: {e}")
    
    return results

def get_top5(logits):
    top5_idx = np.argsort(logits)[-5:][::-1]
    top5 = []
    for idx in top5_idx:
        top5.append(f"{CLASS_NAMES[idx]} ({logits[idx]:.4f})")
    return top5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='yaml config file')
    parser.add_argument('--ckpts', type=str, required=True, help='checkpoint path')
    parser.add_argument('--input', type=str, required=True, help='input PCD file or folder')
    parser.add_argument('--npoints', type=int, default=1024, help='number of points')
    args = parser.parse_args()
    
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.local_rank = 0
    
    config = get_config(args)
    config.npoints = args.npoints
    
    base_model = load_model(args, config)
    
    print(f"Model loaded from: {args.ckpts}")
    print(f"Config: {args.config}")
    print(f"GPU available: {args.use_gpu}")
    print(f"Number of points: {config.npoints}")
    
    if os.path.isfile(args.input):
        print(f"\n=== Testing single PCD file ===")
        raw_points = load_pcd_file(args.input)
        points = preprocess_point_cloud(raw_points, config.npoints)
        pred, confidence, logits = predict_single_point_cloud(base_model, points, config)
        
        print(f"\nResults:")
        print(f"  File: {args.input}")
        print(f"  Points: {len(raw_points)}")
        print(f"  Predicted class: {CLASS_NAMES[pred]}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"\n  Top 5 predictions:")
        for i, item in enumerate(get_top5(logits)):
            print(f"    {i+1}. {item}")
    
    elif os.path.isdir(args.input):
        print(f"\n=== Testing PCD folder: {args.input} ===")
        results = predict_pcd_folder(base_model, args.input, config, args)
        
        print(f"\n=== Summary ===")
        print(f"Total files processed: {len(results)}")
        for result in results:
            print(f"  {result['filename']}: {result['predicted_class']} ({result['confidence']:.4f})")
    
    else:
        print(f"Error: {args.input} is not a valid file or folder")

if __name__ == '__main__':
    main()