"""
Convert HuggingFace ModelNet40 dataset to PointMamba format.
Creates:
  - modelnet40_shape_names.txt
  - modelnet40_train.txt / modelnet40_test.txt (lines like "airplane/airplane_0001")
  - modelnet40_train_8192pts_fps.dat / modelnet40_test_8192pts_fps.dat (pickle files)
"""
import os, sys, pickle
import numpy as np

os.environ['http_proxy'] = 'http://127.0.0.1:7897/'
os.environ['https_proxy'] = 'http://127.0.0.1:7897/'
os.environ.pop('all_proxy', None)
os.environ.pop('ALL_PROXY', None)

from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = "/home/saw/ros2_ws/src/semphyspc/data/ModelNet/modelnet40_normal_resampled"
NUM_POINTS = 8192
TRAIN_RATIO = 0.8  # roughly follow Point-BERT split

def fps(xyz, npoints):
    """Farthest point sampling."""
    N = xyz.shape[0]
    centroids = np.zeros(npoints, dtype=np.int64)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoints):
        centroids[i] = farthest
        centroid = xyz[farthest]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return centroids


def parse_point_cloud(txt_content):
    """Parse comma-separated x,y,z(,nx,ny,nz) point cloud."""
    points = []
    for line in txt_content.strip().split('\n'):
        if line.strip():
            values = line.strip().split(',')
            if len(values) >= 3:
                points.append([float(values[0]), float(values[1]), float(values[2])])
    return np.array(points, dtype=np.float32)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading HuggingFace dataset...")
    dataset = load_dataset("Pointcept/modelnet40_normal_resampled-compressed",
                           split="train", streaming=True)

    # Phase 1: collect metadata
    shape_names = set()
    train_files_list = set()
    test_files_list = set()
    all_samples = []

    print("Phase 1: scanning metadata...")
    for item in tqdm(dataset, total=12318, desc="Scanning"):
        key = item['__key__']  # e.g., ./airplane/airplane_0001
        parts = key.split('/')
        category = parts[1]

        if category in ['filelist', 'modelnet10_shape_names', 'modelnet10_test',
                        'modelnet10_train', 'modelnet40_shape_names',
                        'modelnet40_test', 'modelnet40_train']:
            if category == 'modelnet40_shape_names':
                for line in item['txt'].strip().split('\n'):
                    name = line.strip()
                    if name and not name.startswith('data/'):
                        shape_names.add(name)
            elif category == 'modelnet40_train':
                for line in item['txt'].strip().split('\n'):
                    name = line.strip()
                    if name:
                        train_files_list.add(name)
            elif category == 'modelnet40_test':
                for line in item['txt'].strip().split('\n'):
                    name = line.strip()
                    if name:
                        test_files_list.add(name)
            continue

        shape_names.add(category)
        all_samples.append(item)

    sorted_classes = sorted(shape_names)
    class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    print(f"Found {len(sorted_classes)} classes: {sorted_classes}")

    # Phase 2: process point clouds
    print("\nPhase 2: processing point clouds...")
    train_data, train_labels = [], []
    test_data, test_labels = [], []
    train_files, test_files = [], []

    for item in tqdm(all_samples, desc="Processing"):
        key = item['__key__']
        parts = key.split('/')
        category = parts[1]
        filename = '/'.join(parts[1:])  # airplane/airplane_0001

        if category in ['filelist', 'modelnet10_shape_names', 'modelnet10_test',
                        'modelnet10_train', 'modelnet40_shape_names',
                        'modelnet40_test', 'modelnet40_train']:
            continue

        points = parse_point_cloud(item['txt'])

        if len(points) < NUM_POINTS:
            pad = NUM_POINTS - len(points)
            points = np.vstack([points, points[:pad]])

        # FPS sample
        indices = fps(points, NUM_POINTS)
        points = points[indices]  # (8192, 3)

        label = np.array([class_to_idx[category]], dtype=np.int32)

        if filename in train_files_list:
            train_data.append(points)
            train_labels.append(label)
            train_files.append(filename)
        else:
            test_data.append(points)
            test_labels.append(label)
            test_files.append(filename)

    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Save shape_names.txt
    with open(os.path.join(OUTPUT_DIR, 'modelnet40_shape_names.txt'), 'w') as f:
        for c in sorted_classes:
            f.write(c + '\n')
    print(f"Saved modelnet40_shape_names.txt")

    # Save train.txt / test.txt
    with open(os.path.join(OUTPUT_DIR, 'modelnet40_train.txt'), 'w') as f:
        for name in train_files:
            f.write(name + '\n')
    print(f"Saved modelnet40_train.txt ({len(train_files)} entries)")

    with open(os.path.join(OUTPUT_DIR, 'modelnet40_test.txt'), 'w') as f:
        for name in test_files:
            f.write(name + '\n')
    print(f"Saved modelnet40_test.txt ({len(test_files)} entries)")

    # Save .dat pickle files
    train_save_path = os.path.join(OUTPUT_DIR, 'modelnet40_train_8192pts_fps.dat')
    with open(train_save_path, 'wb') as f:
        pickle.dump([train_data, train_labels], f)
    print(f"Saved {train_save_path} ({len(train_data)} samples)")

    test_save_path = os.path.join(OUTPUT_DIR, 'modelnet40_test_8192pts_fps.dat')
    with open(test_save_path, 'wb') as f:
        pickle.dump([test_data, test_labels], f)
    print(f"Saved {test_save_path} ({len(test_data)} samples)")

    print("\nAll done! Files in", OUTPUT_DIR)
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {fname} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
