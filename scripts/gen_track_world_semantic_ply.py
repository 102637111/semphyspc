#!/usr/bin/env python3
"""生成与 semphyspc_sim/worlds/semantic_track.world 对齐的合成语义点云（Semantic3D PLY）。

world 物体（坐标 = map 帧，车 spawn (0,0) yaw=0、静态 TF map→odom 恒等）：
  ground 全场铺底 cid=24（passable→0），走廊 y∈[-2,2] 内自由
  person (3,0) r=0.3        cid=2  （lethal→254，双重致命）
  vegetation_top/bottom     cid=30 （deformable→46，唯一可穿越软通道）
走廊墙 y=±2 与边界墙 y=±6 有碰撞，靠激光标硬障碍，语义不标（避免语义层把墙标 0 覆盖激光）。

用法：
  python3 scripts/gen_track_world_semantic_ply.py -o /tmp/track_semantic.ply --dry-run
"""
import argparse
import struct

import numpy as np

# 类别 -> 世界坐标矩形区域 (x0, x1, y0, y1) + 步长（越靠后叠加优先级越高，逐 cell 取 max）
SCENE = [
    (24, -35.0, 35.0, -7.0, 7.0, 0.10),   # 地面铺底（passable→0，全场 known，含走廊外；步长 0.1 < 0.2 cell 保证覆盖）
    (2,    7.7,  8.3, -0.3, 0.3, 0.05),   # person_1（lethal→254）
    (2,   17.7, 18.3, -0.3, 0.3, 0.05),   # person_2（lethal→254）
    (2,   27.7, 28.3, -0.3, 0.3, 0.05),   # person_3（lethal→254）
    (30,   7.5,  8.5,  0.5, 2.5, 0.08),   # vegetation_1_top（deformable→46，软通道；下沿 0.5 紧贴 person 上沿 0.3，逼路径穿植被）
    (30,   7.5,  8.5, -2.5, -0.5, 0.08),  # vegetation_1_bottom（deformable→46，软通道）
    (30,  17.5, 18.5,  0.5, 2.5, 0.08),   # vegetation_2_top（deformable→46，软通道）
    (30,  17.5, 18.5, -2.5, -0.5, 0.08),  # vegetation_2_bottom（deformable→46，软通道）
    (30,  27.5, 28.5,  0.5, 2.5, 0.08),   # vegetation_3_top（deformable→46，软通道）
    (30,  27.5, 28.5, -2.5, -0.5, 0.08),  # vegetation_3_bottom（deformable→46，软通道）
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
    ap.add_argument('-o', '--out', default='/tmp/track_semantic.ply')
    ap.add_argument('--cids', default='24,2,30',
                    help='要生成的类别 id 列表（逗号分隔）')
    args = ap.parse_args()
    cids = [int(c) for c in args.cids.split(',')]
    x, y, z, sem = gen(cids)
    n = write_ply(args.out, x, y, z, sem)
    print(f'生成 {n} 点 -> {args.out}')
    print(f'范围 x[{x.min():.1f},{x.max():.1f}] y[{y.min():.1f},{y.max():.1f}]')
    for cid in cids:
        m = sem == cid
        print(f'  cid={cid:2d} 点数={int(m.sum()):6d}')


if __name__ == '__main__':
    main()
