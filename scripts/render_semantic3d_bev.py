#!/usr/bin/env python3
"""真实点云（Semantic3D）俯视图：读语义 PLY → 裁剪/下采样 → 语义 / 物理安全 / 软代价 三张 BEV 俯视 PNG。

输入是 infer_map.py 导出的 Semantic3D 完整语义 PLY（xyz + rgb + semantic_id，千万点级）。
输出三张俯视图（论文/docx 插图用，深色背景 + 图例）：
  1. *_bev_semantic.png   32 类语义 HSV 配色（识别物体种类）
  2. *_bev_safety.png     5 级碰撞安全配色（lethal 红 / hard 橙 / deformable 绿 / passable 灰）
  3. *_bev_cost.png       语义软代价连续配色（person 254 红 → hard 220 橙 → 植被 46 浅绿 → 地面 0 蓝灰）
第三张即语义软代价层的俯视投影，对应 Nav2 规划里的 /semantic_cost。

用法：
  python3 scripts/render_semantic3d_bev.py \
      -i paper/figures/map/domfountain_station3_map_semantic.ply \
      -o paper/figures/effect --stem domfountain_station3 \
      --max-points 1200000 --xmin -25 --xmax 25 --ymin -25 --ymax 25
"""
import argparse
import colorsys
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.materials import COLLISION_SAFETY, LIDARSEG_32, apply_materials, class_physics  # noqa: E402

_DT = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('semantic_id', '<i4')])


def read_semantic_ply(path):
    with open(path, 'rb') as f:
        n = 0
        while True:
            line = f.readline().decode('ascii', 'ignore').strip()
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            elif line == 'end_header':
                break
        return np.fromfile(f, dtype=_DT, count=n)


def class_to_cost(cid):
    ph = class_physics(int(cid))
    s = ph['safety']
    if s == 'passable':
        return 0
    if s == 'lethal':
        return 254
    L = float(ph['lethality'])
    if s == 'hard':
        return int(150 + (L - 0.50) / 0.40 * 70)
    return int(30 + (L - 0.05) / 0.45 * 50)


_COST_TABLE = np.array([class_to_cost(c) for c in range(32)], dtype=np.int32)

SEM_PALETTE = np.array([colorsys.hsv_to_rgb(i / len(LIDARSEG_32), 0.75, 0.95)
                        for i in range(len(LIDARSEG_32))], dtype=np.float32)
SAFETY_COLORS = {
    'lethal': (1.00, 0.10, 0.10),
    'hard': (1.00, 0.50, 0.05),
    'frangible': (1.00, 0.90, 0.10),
    'deformable': (0.05, 0.75, 0.10),
    'passable': (0.60, 0.60, 0.60),
}
# 软代价连续配色锚点：cost -> RGB（0 蓝灰 / 46 浅绿 / 150 黄 / 220 橙 / 254 红）
_COST_X = np.array([0, 46, 150, 220, 254], dtype=np.float32)
_COST_R = np.array([0.45, 0.30, 0.95, 1.00, 1.00], dtype=np.float32)
_COST_G = np.array([0.55, 0.80, 0.80, 0.50, 0.10], dtype=np.float32)
_COST_B = np.array([0.75, 0.35, 0.20, 0.05, 0.10], dtype=np.float32)


def _short_name(name):
    for pref, abbr in [('human.pedestrian.', 'ped.'), ('movable_object.', 'obj.'),
                       ('static_object.', 'static.'), ('vehicle.', 'veh.')]:
        if name.startswith(pref):
            return abbr + name[len(pref):]
    return name


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def render_bev(xyz, colors, out_path, title, legend_items=None):
    from matplotlib.patches import Patch
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.scatter(xyz[:, 0], xyz[:, 1], c=colors, s=1.2, rasterized=True, linewidths=0)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', color='#c9d1d9')
    ax.set_ylabel('y (m)', color='#c9d1d9')
    ax.set_title(title, fontsize=13, color='#e6edf3')
    ax.tick_params(colors='#8b949e')
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    ax.grid(True, alpha=0.15, color='#30363d')
    if legend_items:
        handles = [Patch(facecolor=c, label=name) for name, c in legend_items]
        leg = ax.legend(handles=handles, loc='upper right', fontsize=7,
                        facecolor='#161b22', edgecolor='#30363d', framealpha=0.9)
        for t in leg.get_texts():
            t.set_color('#c9d1d9')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor(),
                transparent=False)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True, help='Semantic3D 语义 PLY')
    ap.add_argument('-o', '--out-dir', default='paper/figures/effect')
    ap.add_argument('--stem', default=None, help='输出文件名前缀（默认取 PLY basename 去 _map_semantic）')
    ap.add_argument('--max-points', type=int, default=1200000, help='下采样点数（scatter 性能）')
    ap.add_argument('--xmin', type=float, default=None)
    ap.add_argument('--xmax', type=float, default=None)
    ap.add_argument('--ymin', type=float, default=None)
    ap.add_argument('--ymax', type=float, default=None)
    args = ap.parse_args()

    arr = read_semantic_ply(args.input)
    xyz = np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float32)
    rgb = np.column_stack([arr['red'], arr['green'], arr['blue']]).astype(np.float32) / 255.0
    sem = arr['semantic_id'].astype(np.int64)
    del arr
    print(f'读入 {len(sem):,} 点')

    x, y = xyz[:, 0], xyz[:, 1]
    if args.xmin is not None or args.xmax is not None or args.ymin is not None or args.ymax is not None:
        xmin = args.xmin if args.xmin is not None else x.min()
        xmax = args.xmax if args.xmax is not None else x.max()
        ymin = args.ymin if args.ymin is not None else y.min()
        ymax = args.ymax if args.ymax is not None else y.max()
        m = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
        xyz, sem, rgb = xyz[m], sem[m], rgb[m]
        print(f'裁剪 x[{xmin:.0f},{xmax:.0f}) y[{ymin:.0f},{ymax:.0f}) → {len(sem):,} 点')

    if len(sem) > args.max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(sem), args.max_points, replace=False)
        idx.sort()
        xyz, sem, rgb = xyz[idx], sem[idx], rgb[idx]
        print(f'下采样 → {len(sem):,} 点')

    stem = args.stem or os.path.basename(args.input).replace('_map_semantic.ply', '')
    os.makedirs(args.out_dir, exist_ok=True)

    # 0) 原始 RGB 俯视图（无标注，真实点云原貌）
    render_bev(xyz, rgb, os.path.join(args.out_dir, f'{stem}_bev_rgb.png'),
               f'{stem} — real point cloud BEV (RGB)')

    # 1) 语义俯视图
    sem_color = SEM_PALETTE[sem]
    present = np.unique(sem)
    sem_legend = [(_short_name(LIDARSEG_32[c]), SEM_PALETTE[c]) for c in present if c < len(LIDARSEG_32)]
    render_bev(xyz, sem_color, os.path.join(args.out_dir, f'{stem}_bev_semantic.png'),
               f'{stem} — real point cloud BEV (semantic)', legend_items=sem_legend)

    # 2) 物理安全俯视图
    mat = apply_materials(sem)
    safety_color = np.zeros((len(sem), 3), np.float32)
    for si, sname in enumerate(COLLISION_SAFETY):
        safety_color[mat['safety_id'] == si] = SAFETY_COLORS[sname]
    render_bev(xyz, safety_color, os.path.join(args.out_dir, f'{stem}_bev_safety.png'),
               f'{stem} — real point cloud BEV (collision safety)',
               legend_items=[(s, SAFETY_COLORS[s]) for s in COLLISION_SAFETY])

    # 3) 软代价俯视图（连续 0~254）
    cost = _COST_TABLE[sem]
    cost_color = np.column_stack([
        np.interp(cost.astype(np.float32), _COST_X, _COST_R),
        np.interp(cost.astype(np.float32), _COST_X, _COST_G),
        np.interp(cost.astype(np.float32), _COST_X, _COST_B),
    ])
    cost_legend = [('ground 0', (0.45, 0.55, 0.75)), ('vegetation 46', (0.30, 0.80, 0.35)),
                   ('hard 150-220', (1.00, 0.50, 0.05)), ('person 254', (1.00, 0.10, 0.10))]
    render_bev(xyz, cost_color, os.path.join(args.out_dir, f'{stem}_bev_cost.png'),
               f'{stem} — real point cloud BEV (semantic soft-cost)', legend_items=cost_legend)

    print(f'完成 → {args.out_dir}/')
    for f in (f'{stem}_bev_rgb.png', f'{stem}_bev_semantic.png', f'{stem}_bev_safety.png', f'{stem}_bev_cost.png'):
        p = os.path.join(args.out_dir, f)
        print(f'    - {p}  ({os.path.getsize(p)/1024:.0f} KB)')


if __name__ == '__main__':
    main()
