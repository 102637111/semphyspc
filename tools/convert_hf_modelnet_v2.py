"""
Fast ModelNet40 conversion: download parquet files from HF, process locally.
Creates: modelnet40_train_8192pts_fps.dat, modelnet40_test_8192pts_fps.dat,
         modelnet40_shape_names.txt, modelnet40_train.txt, modelnet40_test.txt
"""
import os, sys, pickle, io, gzip
import numpy as np
import pyarrow.parquet as pq

os.environ['http_proxy'] = 'http://127.0.0.1:7897/'
os.environ['https_proxy'] = 'http://127.0.0.1:7897/'
os.environ.pop('all_proxy', None)
os.environ.pop('ALL_PROXY', None)

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

OUTPUT_DIR = "/home/saw/ros2_ws/src/semphyspc/data/ModelNet/modelnet40_normal_resampled"
NUM_POINTS = 8192
REPO = "Pointcept/modelnet40_normal_resampled-compressed"

def fps(xyz, npoints):
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

def parse_point_cloud(txt):
    points = []
    for line in txt.strip().split('\n'):
        if line.strip():
            vals = line.strip().split(',')
            if len(vals) >= 3:
                points.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return np.array(points, dtype=np.float32)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Download all parquet files
    print("Listing repo files...")
    files = list_repo_files(REPO, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith('.parquet')]
    print(f"Found {len(parquet_files)} parquet files")

    # Step 2: Download and parse all data
    all_samples = {}  # key -> (xyz, category)
    shape_names = set()
    train_files = set()
    test_files = set()

    for pf in tqdm(parquet_files, desc="Downloading parquets"):
        local_path = hf_hub_download(repo_id=REPO, filename=pf, repo_type="dataset")
        table = pq.read_table(local_path)
        df = table.to_pandas()
        for _, row in df.iterrows():
            key = row['__key__']
            txt = row['txt']
            parts = key.split('/')
            category = parts[1]

            if category in ['filelist', 'modelnet10_shape_names', 'modelnet10_test',
                            'modelnet10_train', 'modelnet40_shape_names',
                            'modelnet40_test', 'modelnet40_train']:
                if category == 'modelnet40_shape_names':
                    for line in txt.strip().split('\n'):
                        name = line.strip()
                        if name and not name.startswith('data/'):
                            shape_names.add(name)
                elif category == 'modelnet40_train':
                    for line in txt.strip().split('\n'):
                        if line.strip():
                            train_files.add(line.strip())
                elif category == 'modelnet40_test':
                    for line in txt.strip().split('\n'):
                        if line.strip():
                            test_files.add(line.strip())
            else:
                shape_names.add(category)
                all_samples[key] = (txt, category)

    sorted_classes = sorted(shape_names)
    class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    print(f"Classes ({len(sorted_classes)}): {sorted_classes}")

    # Step 3: Process point clouds
    train_data, train_labels, train_names = [], [], []
    test_data, test_labels, test_names = [], [], []

    for key, (txt, category) in tqdm(all_samples.items(), desc="Processing"):
        # Normalize key format: ./airplane/airplane_0001 -> airplane/airplane_0001
        filename = '/'.join(key.split('/')[1:])  # airplane/airplane_0001

        points = parse_point_cloud(txt)
        if len(points) < NUM_POINTS:
            pad = NUM_POINTS - len(points)
            points = np.vstack([points, points[:pad]])
        indices = fps(points, NUM_POINTS)
        points = points[indices]

        label = np.array([class_to_idx[category]], dtype=np.int32)

        if filename in train_files:
            train_data.append(points)
            train_labels.append(label)
            train_names.append(filename)
        else:
            test_data.append(points)
            test_labels.append(label)
            test_names.append(filename)

    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Save files
    for name, data in [('modelnet40_shape_names.txt', '\n'.join(sorted_classes) + '\n'),
                       ('modelnet40_train.txt', '\n'.join(train_names) + '\n'),
                       ('modelnet40_test.txt', '\n'.join(test_names) + '\n')]:
        with open(os.path.join(OUTPUT_DIR, name), 'w') as f:
            f.write(data)
        print(f"Saved {name}")

    for split, data, labels in [('train', train_data, train_labels),
                                 ('test', test_data, test_labels)]:
        path = os.path.join(OUTPUT_DIR, f'modelnet40_{split}_8192pts_fps.dat')
        with open(path, 'wb') as f:
            pickle.dump([data, labels], f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Saved {path} ({size_mb:.0f} MB)")

    print("Done!")

if __name__ == '__main__':
    main()
