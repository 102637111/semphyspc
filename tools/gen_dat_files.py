"""
Fast parallel generation of ModelNet40 .dat files using multiprocessing.
"""
import os, sys, pickle
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

ROOT = "/home/saw/ros2_ws/src/semphyspc/data/ModelNet/modelnet40_normal_resampled"
NUM_POINTS = 8192
SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'train'  # 'train' or 'test'

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

def process_one(args):
    cls_name, filename, cls_idx = args
    filepath = os.path.join(ROOT, cls_name, filename + '.txt')
    try:
        points = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

    if len(points) < NUM_POINTS:
        pad = NUM_POINTS - len(points)
        points = np.vstack([points, points[:pad]])

    indices = fps(points[:, :3], NUM_POINTS)
    points = points[indices]

    label = np.array([cls_idx], dtype=np.int32)
    return points, label

def main():
    print(f"Processing {SPLIT} split...")

    # Read class names
    with open(os.path.join(ROOT, 'modelnet40_shape_names.txt')) as f:
        classes = [line.strip() for line in f]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Read file list
    with open(os.path.join(ROOT, f'modelnet40_{SPLIT}.txt')) as f:
        entries = [line.strip() for line in f]

    # Build task list
    tasks = []
    for entry in entries:
        cls_name = '_'.join(entry.split('_')[0:-1])
        tasks.append((cls_name, entry, class_to_idx[cls_name]))

    print(f"Processing {len(tasks)} files with {cpu_count()} workers...")

    # Parallel processing
    results = []
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap(process_one, tasks, chunksize=50),
                          total=len(tasks), desc=f"Generating {SPLIT}.dat"):
            if result is not None:
                results.append(result)

    # Separate points and labels
    list_of_points = [r[0] for r in results]
    list_of_labels = [r[1] for r in results]

    # Save as .dat (pickle file)
    save_path = os.path.join(ROOT, f'modelnet40_{SPLIT}_8192pts_fps.dat')
    with open(save_path, 'wb') as f:
        pickle.dump([list_of_points, list_of_labels], f)

    size_mb = os.path.getsize(save_path) / (1024*1024)
    print(f"Saved {save_path} ({size_mb:.0f} MB, {len(results)} samples)")

if __name__ == '__main__':
    main()
