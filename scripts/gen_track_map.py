#!/usr/bin/env python3
"""生成 semantic_track world 对齐的静态地图（PGM+YAML），供 map_server 定义 global costmap 尺寸。

world 场地 60×60 中心原点；地图 resolution 0.05、origin[-30,-30]、1200×1200（60m/0.05）。
画 occupied：走廊墙 y=±3（x∈[-28,28]）、边界墙 y=±8（x∈[-28,28]）。
person(10/22/34,0)/vegetation 不画（由激光 ObstacleLayer + 语义软代价层标）。

关键：Humble global costmap 非 rolling 模式无 static map 时 master 固定 5m×5m 不扩展，
必须靠 map_server 的静态地图撑开尺寸，semantic_layer/obstacle_layer 才能正确写入。

用法：
  python3 scripts/gen_track_map.py -o /tmp/track_map
  # 生成 /tmp/track_map.pgm + /tmp/track_map.yaml
"""
import argparse

import numpy as np

RES = 0.05
W = H = 1600        # 80m / 0.05
FREE = 254
OCC = 0
ORIGIN = 40.0       # origin x/y = -40


def fill_wall(img, y_center, half_thick=0.1, x_min=-35.0, x_max=35.0):
    """world 坐标水平墙 → PGM 像素（PGM 第 0 行=top=world y 最大）。"""
    y0, y1 = y_center - half_thick, y_center + half_thick
    r0 = int(round((ORIGIN - y1) / RES))   # y 大 → 行号小（靠近 top）
    r1 = int(round((ORIGIN - y0) / RES))
    c0 = int(round((x_min + ORIGIN) / RES))
    c1 = int(round((x_max + ORIGIN) / RES))
    img[r0:r1 + 1, c0:c1 + 1] = OCC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='/tmp/track_map',
                    help='输出前缀（生成 <out>.pgm + <out>.yaml）')
    args = ap.parse_args()

    img = np.full((H, W), FREE, dtype=np.uint8)
    for yc in (8.0, -8.0, 3.0, -3.0):
        fill_wall(img, yc)

    pgm = args.out + '.pgm'
    yaml = args.out + '.yaml'
    with open(pgm, 'wb') as f:
        f.write(f'P5\n{W} {H}\n255\n'.encode())
        f.write(img.tobytes())

    with open(yaml, 'w') as f:
        f.write(f'image: {pgm}\n')
        f.write(f'resolution: {RES}\n')
        f.write(f'origin: [-{ORIGIN}, -{ORIGIN}, 0.0]\n')
        f.write('negate: 0\n')
        f.write('occupied_thresh: 0.65\n')
        f.write('free_thresh: 0.196\n')

    print(f'生成 {pgm} ({W}x{H}) + {yaml}')
    print(f'occupied 像素 = {int((img == OCC).sum())}，free = {int((img == FREE).sum())}')


if __name__ == '__main__':
    main()
