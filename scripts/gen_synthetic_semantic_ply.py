#!/usr/bin/env python3
"""生成合成语义点云（Semantic3D PLY 格式），用于阶段 1 验证语义软代价层分层。

在 room.yaml 地图坐标范围内（origin[-10.4,-6.53]，约 x∈[-10.4,8.4], y∈[-6.53,4.57]）
放置几类语义障碍，semantic_id 即 nuScenes 32 类 id：
  ground(24)=passable→0、person(2)=lethal→254、car(17)=hard→220、
  vegetation(30)=deformable→46、barrier(9)=frangible→57、trafficcone(12)=frangible→35。

用法：
  python3 scripts/gen_synthetic_semantic_ply.py -o /tmp/synthetic_semantic.ply
"""
import argparse
import struct

import numpy as np

# 类别 -> 世界坐标矩形区域 (x0, x1, y0, y1) + 步长
SCENE = [
    (24, -9.0, 7.0, -5.0, 4.0, 0.30),     # 地面铺底（passable）
    (2,  -0.5, 1.5, -0.5, 1.5, 0.10),     # person（lethal）
    (17, -6.0, -4.0, -4.0, -2.0, 0.20),   # car（hard）
    (30, 3.5, 5.5, -4.0, 3.0, 0.20),      # vegetation（deformable，竖直带）
    (9,  -1.0, 1.0, 3.0, 3.5, 0.10),      # barrier（frangible，水平短墙）
    (12, 1.5, 3.0, -4.0, -3.5, 0.10),     # trafficcone（frangible）
]


def gen(cids):
    xs, ys, zs, sems = [], [], [], []
    for cid, x0, x1, y0, y1, step in SCENE:
        if cid not in cids:
            continue
        x = np.arange(x0, x1, step)
        y = np.arange(y0, y1, step)
        gx, gy = np.meshgrid(x, y)
        xs.append(gx.ravel())
        ys.append(gy.ravel())
        zs.append(np.zeros(gx.size))
        sems.append(np.full(gx.size, cid, dtype=np.int32))
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    z = np.concatenate(zs).astype(np.float32)
    sem = np.concatenate(sems)
    return x, y, z, sem


def write_ply(path, x, y, z, sem, rgb=(200, 200, 200)):
    n = x.size
    with open(path, 'wb') as f:
        header = (
            'ply\n'
            'format binary_little_endian 1.0\n'
            f'element vertex {n}\n'
            'property float x\n'
            'property float y\n'
            'property float z\n'
            'property uchar red\n'
            'property uchar green\n'
            'property uchar blue\n'
            'property int semantic_id\n'
            'end_header\n'
        )
        f.write(header.encode('ascii'))
        r, g, b = rgb
        for i in range(n):
            f.write(struct.pack('<fffBBB i', x[i], y[i], z[i], r, g, b, int(sem[i])))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='/tmp/synthetic_semantic.ply')
    ap.add_argument('--cids', default='24,2,17,30,9,12',
                    help='要生成的类别 id 列表（逗号分隔）')
    args = ap.parse_args()
    cids = [int(c) for c in args.cids.split(',')]
    x, y, z, sem = gen(cids)
    n = write_ply(args.out, x, y, z, sem)
    print(f'生成 {n} 点 -> {args.out}')
    print(f'范围 x[{x.min():.1f},{x.max():.1f}] y[{y.min():.1f},{y.max():.1f}]')


if __name__ == '__main__':
    main()
