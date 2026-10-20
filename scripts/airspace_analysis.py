#!/usr/bin/env python3
"""室内外模式识别 + 无人机可飞行空域分析（离线，几何法，不依赖语义标签）。

读语义 PLY（xyz + semantic_id），对每个 XY cell 计算：
  1) 地面高度（地面点 z 最小值）——无人机最低飞行下限
  2) 障碍物高度场（第一个非地面物体的相对高度）——无人机安全飞行上限
  3) 低空遮蔽率（地面以上 12m 内被障碍占据的高度层比例）——室内/有顶判定依据

产出：两张子图
  (a) 障碍物高度场（可飞行上限，越高越开阔）
  (b) 室内外分类：有顶/室内（红）、开放室外（蓝绿）、无地面数据（灰）

障碍 = 非地面且非 noise（建筑立面、树冠、路灯、喷泉…）。
室内外判定（可解释的经验阈值）：
  - 有地面 且 头顶 6m 内有障碍物  → 有顶/室内/连廊（天花板/屋顶/树冠低枝）
  - 有地面 且 头顶 6m 内无障碍    → 开放室外（天空开阔，无人机可高飞）
  - 地面点不足                     → 无地面数据（建筑/树冠内部，无人机不可达）

用法（在项目根 /home/saw/ros2_ws/src/semphyspc/ 下）：
  python3 scripts/airspace_analysis.py \
      --semantic paper/figures/map/domfountain_station3_map_semantic.ply \
      --cell_size 1.0 --min_ground_points 5 --max_points 8000000 \
      --output paper/figures/airspace/domfountain_airspace.png
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体：黑体 > 宋体 > 楷体（论文图优先黑体/宋体），避免中文变方框
_zh_avail = {f.name for f in font_manager.fontManager.ttflist
             if ('CJK' in f.name or 'UMing' in f.name or 'UKai' in f.name
                 or 'WenQuanYi' in f.name)}
_zh_pref = [n for n in ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Serif CJK SC',
                        'Noto Serif CJK JP', 'AR PL UMing CN', 'AR PL UKai CN',
                        'WenQuanYi Zen Hei'] if n in _zh_avail]
if _zh_pref:
    plt.rcParams['font.sans-serif'] = [_zh_pref[0], 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# nuScenes 32 类：可通行地面类
GROUND_CLASSES = {24, 25, 26, 27}
# 室内/有顶判定：头顶该高度（米）内存在障碍即判「有顶」
COVERED_CEIL = 6.0
# 低空遮蔽率统计高度（米）
LOW_CEIL = 12.0

_DT = {'float': '<f4', 'uchar': 'u1', 'int': '<i4', 'uint': '<u4',
       'short': '<i2', 'ushort': '<u2'}


def read_ply(path):
    """读二进制 PLY（本项目 write_ply_binary 格式），返回结构化数组。"""
    with open(path, 'rb') as f:
        n_vertex = 0
        props = []
        while True:
            line = f.readline().decode('ascii', 'ignore').strip()
            if line.startswith('element vertex'):
                n_vertex = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                props.append((parts[-1], parts[1]))
            elif line == 'end_header':
                break
        dtype = np.dtype([(name, _DT[t]) for name, t in props])
        return np.fromfile(f, dtype=dtype, count=n_vertex)


def airspace_stats(x, y, z, sem, cell_size, min_ground_points, z_res=1.0,
                   xy_pct=(0.5, 99.5)):
    """逐 cell 聚合地面高度 / 障碍物高度场 / 低空遮蔽率。"""
    is_ground = np.isin(sem, list(GROUND_CLASSES))
    is_obstacle = ~is_ground & (sem != 0)  # 非地面且非 noise = 障碍

    lo, hi = xy_pct
    xmin = float(np.percentile(x, lo))
    xmax = float(np.percentile(x, hi))
    ymin = float(np.percentile(y, lo))
    ymax = float(np.percentile(y, hi))
    gx = np.floor((x - xmin) / cell_size).astype(np.int64)
    gy = np.floor((y - ymin) / cell_size).astype(np.int64)
    nx = int(np.floor((xmax - xmin) / cell_size)) + 1
    ny = int(np.floor((ymax - ymin) / cell_size)) + 1
    n_cells = nx * ny
    inside = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    cid = gx * ny + gy

    # 地面高度：每个 cell 地面点 z 最小值
    gm = inside & is_ground
    gc = cid[gm]
    gz = z[gm]
    g_count = np.bincount(gc, minlength=n_cells)
    g_minz = np.full(n_cells, np.inf)
    np.minimum.at(g_minz, gc, gz)

    # 障碍点相对其 cell 地面的高度（无地面 cell 用全局地面 min 近似）
    om = inside & is_obstacle
    oc = cid[om]
    oz = z[om]
    gz_global = float(gz.min()) if gz.size else float(z.min())
    cell_ground = np.where(np.isfinite(g_minz), g_minz, gz_global)
    h = oz - cell_ground[oc]
    h = np.clip(h, 0.0, None)
    o_count = np.bincount(oc, minlength=n_cells)

    # 障碍物高度场 = 每个 cell 第一个障碍物的相对高度（inf = 无遮挡，可飞最高）
    obstacle_min_h = np.full(n_cells, np.inf)
    np.minimum.at(obstacle_min_h, oc, h)

    # 低空遮蔽率：地面以上 LOW_CEIL 米内被障碍占据的高度层比例
    n_low = int(LOW_CEIL / z_res)
    layer = np.clip((h / z_res).astype(np.int64), 0, n_low - 1)
    occ = np.zeros((n_cells, n_low), dtype=bool)
    occ[oc, layer] = True
    occlusion_low = occ.sum(axis=1) / n_low

    has_ground = g_count >= min_ground_points
    return dict(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                nx=nx, ny=ny, cell_size=cell_size,
                g_count=g_count, o_count=o_count,
                g_minz=g_minz, obstacle_min_h=obstacle_min_h,
                occlusion_low=occlusion_low, has_ground=has_ground)


def classify(stats):
    """室内外分类（基于障碍物高度场，不依赖地面点）：
    0=空白/无点（扫描盲区）, 1=低空遮挡/有顶, 2=开放空域。
    低空遮挡 = 头顶 6m 内有障碍物（天花板/屋顶/连廊顶/树冠低枝/建筑立面），
    对无人机均属「低空受限」区域；开放空域 = 头顶 6m 内无障碍，可自由高飞。
    """
    nx, ny = stats['nx'], stats['ny']
    obs_h = stats['obstacle_min_h'].reshape(nx, ny)
    g = stats['g_count'].reshape(nx, ny)
    o = stats['o_count'].reshape(nx, ny)
    cls = np.zeros((nx, ny), dtype=np.int8)          # 0 = 空白/无点
    has_points = (g + o) > 0
    cls[has_points] = 2                               # 默认开放空域
    cls[has_points & (obs_h < COVERED_CEIL)] = 1      # 头顶 6m 内有障碍 → 低空遮挡/有顶
    return cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--semantic', required=True, help='语义 PLY（含 semantic_id）')
    ap.add_argument('--cell_size', type=float, default=1.0)
    ap.add_argument('--min_ground_points', type=int, default=5)
    ap.add_argument('--max_points', type=int, default=8000000, help='降采样上限')
    ap.add_argument('--xy_percentile', nargs=2, type=float, default=[0.5, 99.5],
                    help='XY 范围分位数裁剪（去离群点），默认 0.5 99.5')
    ap.add_argument('--output', required=True, help='输出 PNG 路径')
    args = ap.parse_args()

    print(f'读取 {args.semantic} ...', flush=True)
    data = read_ply(args.semantic)
    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.float32)
    z = data['z'].astype(np.float32)
    sem = data['semantic_id'].astype(np.int64)
    del data

    n0 = len(x)
    if args.max_points and n0 > args.max_points:
        idx = np.random.default_rng(0).choice(n0, args.max_points, replace=False)
        x, y, z, sem = x[idx], y[idx], z[idx], sem[idx]
    print(f'{n0:,} 点 → 采样 {len(x):,} 点，栅格 {args.cell_size}m ...', flush=True)

    stats = airspace_stats(x, y, z, sem, args.cell_size, args.min_ground_points,
                           xy_pct=tuple(args.xy_percentile))
    nx, ny = stats['nx'], stats['ny']
    cls = classify(stats)
    n_cls0 = int((cls == 0).sum())
    n_cls1 = int((cls == 1).sum())
    n_cls2 = int((cls == 2).sum())
    n_total = nx * ny
    print(f'栅格 {nx}×{ny} cell', flush=True)
    print(f'  空白/无点 {n_cls0:,} ({100.0*n_cls0/n_total:.1f}%)  |  '
          f'低空遮挡/有顶 {n_cls1:,} ({100.0*n_cls1/n_total:.1f}%)  |  '
          f'开放空域 {n_cls2:,} ({100.0*n_cls2/n_total:.1f}%)', flush=True)

    # 障碍物高度场统计（只统计有障碍物的 cell）
    obs_h = stats['obstacle_min_h'].reshape(nx, ny)
    hf = obs_h[np.isfinite(obs_h)]
    if hf.size:
        print(f'障碍物高度场（有遮挡 cell）：中位 {np.median(hf):.1f} m，'
              f'最小 {hf.min():.1f} m，最大 {hf.max():.1f} m', flush=True)

    # ---- 可视化 ----
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    ext = [stats['xmin'], stats['xmax'], stats['ymin'], stats['ymax']]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))

    # (a) 障碍物高度场
    ax = axes[0]
    hh = np.where(np.isfinite(obs_h), obs_h, np.nan)
    hh[(stats['g_count'] + stats['o_count']).reshape(nx, ny) == 0] = np.nan
    im = ax.imshow(np.transpose(hh, (1, 0)), origin='lower', extent=ext,
                   cmap='viridis_r', vmin=0, vmax=30, aspect='equal')
    ax.set_title('(a) 障碍物高度场（无人机可飞行上限）')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('首个障碍物高度 (m)')

    # (b) 室内外分类
    ax = axes[1]
    cmap = matplotlib.colors.ListedColormap(['#999999', '#d1495b', '#2a9d8f'])
    im = ax.imshow(np.transpose(cls, (1, 0)), origin='lower', extent=ext,
                   cmap=cmap, vmin=0, vmax=2, aspect='equal', interpolation='nearest')
    ax.set_title('(b) 室内外模式识别')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['空白/无点', '低空遮挡/有顶', '开放空域'])

    fig.suptitle(f'无人机可飞行空域 + 室内外识别（cell={args.cell_size}m，'
                 f'有顶阈值 {COVERED_CEIL:.0f}m）')
    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print(f'已保存：{args.output}', flush=True)


if __name__ == '__main__':
    main()
