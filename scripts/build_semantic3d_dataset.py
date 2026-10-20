#!/usr/bin/env python3
"""Semantic3D (semantic-8) -> PointMamba npz 转换：大型室外真实 LiDAR 语义分割数据集。

背景：
  作为 nuScenes 之外的第二真实室外语义分割集，用于验证 RGB+XYZ 早期融合分割前置的
  跨数据集泛化。Semantic3D 每个场景约 10^7~10^8 点，需流式降采样到 N_POINTS。

格式（均在 7z 压缩内，本脚本用 `7z x -so` 流式解压，不落盘原始 txt）：
  点云  *.txt    : 每行 7 列 [x y z intensity r g b]，r/g/b 为 0-255 整数。
                  注意：文件命名不一致——多数场景叫 *_xyz_intensity_rgb.txt，
                  但 sg27/sg28 叫 *_intensity_rgb.txt，二者都是 7 列、都含 xyz。
  标签  *.labels : 每行一个整数 (0-8)，与点云行序一一对应。

类别（semantic-8，重映射为 0..7，0(未标注) -> -1 供训练时忽略）：
  1 man-made terrain -> 0   2 natural terrain -> 1   3 high vegetation -> 2
  4 low vegetation   -> 3   5 buildings       -> 4   6 hard scape     -> 5
  7 scanning artefacts-> 6   8 cars            -> 7   0 未标注 -> -1

采样模式（--sample_mode）：
  all     : 全点无偏采样，未标注点 sem=-1（默认，保留未标注区作上下文）
  labeled : 只在标注点(label!=0)采样，产出全为有效标签（无 -1，有效训练点 100%）

输出：与 NuscenesSegDataset 一致，每场景一个 npz：
  {xyz (N,3) float32, rgb (N,3) float32 [0,1], sem (N,) int16}，N=N_POINTS（默认 16384）

用法:
    python scripts/build_semantic3d_dataset.py \
        --src /mnt/data/datasets/Semantic3D \
        --dst /mnt/data/datasets/Semantic3D/seg_npz \
        --n_points 16384 --sample_mode all --seed 0
"""

import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np

# semantic-8 标签 -> 重映射 (0..7)，0(未标注) -> -1
LABEL2SEM = {0: -1, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
CLASS_NAMES = ['man-made terrain', 'natural terrain', 'high vegetation',
               'low vegetation', 'buildings', 'hard scape',
               'scanning artefacts', 'cars']


def stream_7z(archive, member=None):
    """流式解压 archive 中 member（缺省解压唯一文件），返回逐字节读的 file 对象。"""
    cmd = ['7z', 'x', '-so', archive]
    if member:
        cmd.append(member)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout


def stream_arrays(pipe, ncols):
    """把 pipe 的字节流按行解析成 (M, ncols) float32 数组块，逐块 yield。"""
    buf = b''
    while True:
        chunk = pipe.read(1 << 20)
        if not chunk:
            break
        buf += chunk
        *lines, buf = buf.split(b'\n')
        if lines:
            arr = np.fromstring(b'\n'.join(lines), sep=' ', dtype=np.float32)
            if arr.size:
                yield arr.reshape(-1, ncols)
    if buf.strip():
        arr = np.fromstring(buf, sep=' ', dtype=np.float32)
        if arr.size:
            yield arr.reshape(-1, ncols)


def reservoir_sample(arrays_stream, k, rng):
    """对数组块流做水塘采样，返回 (indices, rows)，indices 为全局行号。"""
    res_idx = np.empty(k, dtype=np.int64)
    res_rows = np.empty((k, 7), dtype=np.float32)
    seen = 0
    for block in arrays_stream:
        for i in range(block.shape[0]):
            if seen < k:
                res_idx[seen] = seen
                res_rows[seen] = block[i]
            else:
                j = rng.integers(0, seen + 1)
                if j < k:
                    res_idx[j] = seen
                    res_rows[j] = block[i]
            seen += 1
    if seen < k:
        res_idx = res_idx[:seen]
        res_rows = res_rows[:seen]
    return res_idx, res_rows, seen


def read_labels_at(labels_pipe, idx_set):
    """从 labels 字节流中取 idx_set 指定的行，返回 {index: label}。"""
    out = {}
    i = 0
    for raw in labels_pipe:
        try:
            val = int(raw.split()[0])
        except (ValueError, IndexError):
            continue
        if i in idx_set:
            out[i] = val
        i += 1
    return out


def sample_labeled_indices(labels_pipe, k, rng):
    """只在标注点(label!=0)上做水塘采样，返回 (indices, labels)。"""
    res_idx = np.empty(k, dtype=np.int64)
    res_lab = np.empty(k, dtype=np.int64)
    seen = 0
    i = 0
    for raw in labels_pipe:
        try:
            val = int(raw.split()[0])
        except (ValueError, IndexError):
            i += 1
            continue
        if val != 0:
            if seen < k:
                res_idx[seen] = i
                res_lab[seen] = val
            else:
                j = rng.integers(0, seen + 1)
                if j < k:
                    res_idx[j] = i
                    res_lab[j] = val
            seen += 1
        i += 1
    if seen < k:
        res_idx = res_idx[:seen]
        res_lab = res_lab[:seen]
    return res_idx, res_lab, seen


def read_rows_at(pipe, idx_set, ncols):
    """从点云字节流中取 idx_set 指定的行，返回 {index: row(list)}。"""
    out = {}
    i = 0
    for raw in pipe:
        if i in idx_set:
            out[i] = np.fromstring(raw, sep=' ', dtype=np.float32)
        i += 1
    return out


def convert_scene(archive, label_archive, label_member, dst_path, n_points, seed, sample_mode):
    rng = np.random.default_rng(seed)
    if sample_mode == 'labeled':
        # 先读标签：只在标注点采样
        l_pipe = stream_7z(label_archive, label_member)
        idx, labels, seen = sample_labeled_indices(l_pipe, n_points, rng)
        l_pipe.close()
        if len(idx) == 0:
            return 0, 0
        idx_set = set(idx.tolist())
        p_pipe = stream_7z(archive)
        row_map = read_rows_at(p_pipe, idx_set, 7)
        p_pipe.close()
        order = np.argsort(idx)
        idx = idx[order]
        labels = labels[order]
        rows = np.array([row_map[int(i)] for i in idx], dtype=np.float32)
        sem = np.array([LABEL2SEM.get(int(l), -1) for l in labels], dtype=np.int16)
    else:  # all
        p_pipe = stream_7z(archive)
        idx, rows, total = reservoir_sample(stream_arrays(p_pipe, 7), n_points, rng)
        p_pipe.close()
        if len(idx) == 0:
            return 0, total
        idx_set = set(idx.tolist())
        l_pipe = stream_7z(label_archive, label_member)
        label_map = read_labels_at(l_pipe, idx_set)
        l_pipe.close()
        order = np.argsort(idx)
        idx = idx[order]
        rows = rows[order]
        labels = [label_map.get(int(i), 0) for i in idx]
        sem = np.array([LABEL2SEM.get(int(l), -1) for l in labels], dtype=np.int16)

    xyz = rows[:, :3].astype(np.float32)
    rgb = (rows[:, 4:7] / 255.0).astype(np.float32)  # 跳过第 4 列 intensity
    np.savez_compressed(dst_path, xyz=xyz, rgb=rgb, sem=sem)
    return len(idx), len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Semantic3D 根目录（含 train/ 与 sem8_labels_training.7z）')
    ap.add_argument('--dst', required=True, help='输出 npz 根目录')
    ap.add_argument('--n_points', type=int, default=16384)
    ap.add_argument('--sample_mode', type=str, default='all', choices=['all', 'labeled'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit_scenes', type=int, default=0, help='调试：只转前 N 个场景')
    args = ap.parse_args()

    label_archive = os.path.join(args.src, 'sem8_labels_training.7z')
    if not os.path.exists(label_archive):
        sys.exit(f'[Semantic3D] 找不到标签 {label_archive}')

    dst_dir = os.path.join(args.dst, 'train')
    os.makedirs(dst_dir, exist_ok=True)

    archives = sorted(glob.glob(os.path.join(args.src, 'train', '*.7z')))
    if args.limit_scenes:
        archives = archives[:args.limit_scenes]

    print(f'[Semantic3D] {len(archives)} 个场景 -> {dst_dir}, N={args.n_points}, '
          f'mode={args.sample_mode}', flush=True)
    n_done = 0
    for ai, arch in enumerate(archives):
        base = os.path.basename(arch).replace('.7z', '')
        # 输出 stem：去掉 _xyz_intensity_rgb 或 _intensity_rgb 后缀
        stem = base.replace('_xyz_intensity_rgb', '').replace('_intensity_rgb', '')
        label_member = base + '.labels'
        dst_path = os.path.join(dst_dir, stem + '.npz')
        t0 = time.time()
        n, total = convert_scene(arch, label_archive, label_member, dst_path,
                                 args.n_points, args.seed, args.sample_mode)
        print(f'  [{ai+1}/{len(archives)}] {stem}: 采样 {n} 点 用时 {time.time()-t0:.0f}s',
              flush=True)
        n_done += 1

    print(f'\n[Semantic3D] 完成: {n_done} 场景转换 (mode={args.sample_mode})', flush=True)
    print(f'  类别重映射: {len(CLASS_NAMES)} 类 {CLASS_NAMES}, 0(未标注)->-1', flush=True)


if __name__ == '__main__':
    main()
