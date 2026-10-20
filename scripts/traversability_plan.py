#!/usr/bin/env python3
"""3D 语义点云 → 可通行性代价图 → 坡度可通过路径规划（离线，路线 B）。

读 infer_map.py 生成的语义 PLY（xyz + semantic_id，nuScenes 32 类编号），
完成：语义地面提取 → XY 栅格化 → 每 cell 坡度估计（PCA 平面拟合）与粗糙度
      → 可通行性代价图（语义代价 + 坡度代价 + 粗糙度代价，障碍/陡坡封死）
      → Dijkstra/A*（8 邻域）从 start 到 goal 的最低代价路线 → 代价图 PNG + 统计。

可通行地面类：24=driveable_surface, 25=flat.other, 26=sidewalk, 27=terrain。

用法（在项目根 /home/saw/ros2_ws/src/semphyspc/ 下）：
  python3 scripts/traversability_plan.py \
      --semantic paper/figures/map/domfountain_station3_map_semantic.ply \
      --cell_size 1.0 --max_slope 25 --max_points 8000000 \
      --start 20 15 --goal 85 75 \
      --output paper/figures/traversability/domfountain_trav.png

不指定 --start/--goal 时，自动在可通行区域两端取起终点。
"""
import os
import sys
import argparse
import heapq

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# nuScenes 32 类：可通行地面类
GROUND_CLASSES = {24, 25, 26, 27}
# 语义代价（路面类型偏好：优先 driveable/sidewalk，越野 terrain 代价更高）
SEM_COST = {24: 0.0, 26: 0.1, 25: 0.3, 27: 0.5}

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


def cell_stats(x, y, z, sem, cell_size, min_ground_points):
    """栅格化并逐 cell 聚合地面/障碍信息，返回 (代价图信息字典)。

    栅格范围由「地面点足迹」决定（可通行区域只关心地面覆盖范围），
    避免把远处建筑/植被拉进栅格导致有效 cell 被稀释。
    """
    is_ground = np.isin(sem, list(GROUND_CLASSES))
    is_obstacle = ~is_ground & (sem != 0)  # 非地面且非 noise = 障碍

    xg0, yg0 = x[is_ground], y[is_ground]
    if xg0.size == 0:
        return None
    xmin, ymin = float(xg0.min()), float(yg0.min())
    xmax, ymax = float(xg0.max()), float(yg0.max())

    gx = np.floor((x - xmin) / cell_size).astype(np.int64)
    gy = np.floor((y - ymin) / cell_size).astype(np.int64)
    nx = int(np.floor((xmax - xmin) / cell_size)) + 1
    ny = int(np.floor((ymax - ymin) / cell_size)) + 1
    n_cells = nx * ny
    inside = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    is_ground &= inside
    is_obstacle &= inside
    cid = gx * ny + gy

    gcid = cid[is_ground]
    ocid = cid[is_obstacle]
    xg, yg, zg = x[is_ground], y[is_ground], z[is_ground]

    # 地面聚合（只统计地面点，无占位污染）
    g_count = np.bincount(gcid, minlength=n_cells)
    g_sumz = np.bincount(gcid, weights=zg, minlength=n_cells)
    g_szz = np.bincount(gcid, weights=zg * zg, minlength=n_cells)
    g_sumz2 = g_szz  # 供 std_z 复用（= Σz²）
    g_sxx = np.bincount(gcid, weights=xg * xg, minlength=n_cells)
    g_syy = np.bincount(gcid, weights=yg * yg, minlength=n_cells)
    g_sxy = np.bincount(gcid, weights=xg * yg, minlength=n_cells)
    g_sxz = np.bincount(gcid, weights=xg * zg, minlength=n_cells)
    g_syz = np.bincount(gcid, weights=yg * zg, minlength=n_cells)
    g_sx = np.bincount(gcid, weights=xg, minlength=n_cells)
    g_sy = np.bincount(gcid, weights=yg, minlength=n_cells)

    # 障碍聚合
    o_count = np.bincount(ocid, minlength=n_cells)

    valid = g_count >= min_ground_points
    n = np.maximum(g_count, 1)
    mean_z = g_sumz / n
    std_z = np.sqrt(np.maximum(g_sumz2 / n - mean_z * mean_z, 0.0))
    # 去中心化协方差
    mx, my = g_sx / n, g_sy / n
    cov = np.zeros((n_cells, 3, 3), dtype=np.float32)
    cov[:, 0, 0] = g_sxx / n - mx * mx
    cov[:, 1, 1] = g_syy / n - my * my
    cov[:, 2, 2] = g_szz / n - mean_z * mean_z
    cov[:, 0, 1] = cov[:, 1, 0] = g_sxy / n - mx * my
    cov[:, 0, 2] = cov[:, 2, 0] = g_sxz / n - mx * mean_z
    cov[:, 1, 2] = cov[:, 2, 1] = g_syz / n - my * mean_z

    # 坡度角 = 法向量（最小特征值方向）与竖直轴夹角
    slope_deg = np.zeros(n_cells, dtype=np.float32)
    for i in np.where(valid)[0]:
        w, v = np.linalg.eigh(cov[i])
        nrm = v[:, 0]  # 最小特征值对应法向量
        nz = abs(nrm[2])
        nz = min(max(nz, 0.0), 1.0)
        slope_deg[i] = np.degrees(np.arccos(nz))

    return dict(xmin=xmin, ymin=ymin, nx=nx, ny=ny,
                g_count=g_count, o_count=o_count,
                mean_z=mean_z, std_z=std_z, slope_deg=slope_deg, valid=valid)


def build_costmap(stats, cell_size, max_slope, max_rough):
    """构建代价图：返回 (cost, passable)。cost 越小越可通过；passable=False 为障碍/未知。

    内部把一维 n_cells 聚合量 reshape 成 (nx, ny)：cell (gx,gy) 对应 cid=gx*ny+gy。
    """
    nx, ny = stats['nx'], stats['ny']

    def r2(a):
        return np.asarray(a).reshape(nx, ny)

    valid = r2(stats['valid'])
    slope = r2(stats['slope_deg'])
    rough = r2(stats['std_z'])
    g = r2(stats['g_count'])
    o = r2(stats['o_count'])

    slope_cost = np.clip(slope / max(1.0, max_slope), 0.0, 1.0)
    rough_cost = np.clip(rough / max(0.01, max_rough), 0.0, 1.0)
    obs_ratio = o / np.maximum(g + o, 1)

    c = slope_cost + 0.5 * rough_cost
    c[obs_ratio > 0.3] = np.inf  # 障碍占主导 → 封死
    c[slope > max_slope] = np.inf  # 超坡度阈值 → 封死
    c[~valid] = np.inf  # 无足够地面点 → 未知

    passable = np.isfinite(c)
    cost = np.where(passable, c, np.inf).astype(np.float32)
    return cost, passable


def astar(cost, start, goal):
    """8 邻域 A*，返回从 start 到 goal 的 cell 坐标列表（含起终点）。"""
    nx, ny = cost.shape
    if cost[start] == np.inf or cost[goal] == np.inf:
        return None
    inf = np.inf
    g = np.full((nx, ny), inf)
    g[start] = 0.0
    came = np.full((nx, ny), -1, dtype=np.int64)
    openq = [(heuristic(start, goal), start[0], start[1])]
    visited = set()

    while openq:
        f, cx, cy = heapq.heappop(openq)
        if (cx, cy) in visited:
            continue
        visited.add((cx, cy))
        if (cx, cy) == goal:
            break
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx2, ny2 = cx + dx, cy + dy
                if not (0 <= nx2 < nx and 0 <= ny2 < ny):
                    continue
                if cost[nx2, ny2] == np.inf:
                    continue
                step = 1.41421356 if dx != 0 and dy != 0 else 1.0
                # 移动代价 = 目标 cell 代价（+基础步长）
                newg = g[cx, cy] + step * (1.0 + 3.0 * cost[nx2, ny2])
                if newg < g[nx2, ny2]:
                    g[nx2, ny2] = newg
                    came[nx2, ny2] = cx * ny + cy
                    heapq.heappush(openq, (newg + heuristic((nx2, ny2), goal), nx2, ny2))
    if g[goal] == np.inf:
        return None
    path = [goal]
    cur = goal
    while cur != start:
        prev = came[cur]
        cur = (prev // ny, prev % ny)
        path.append(cur)
    path.reverse()
    return path


def heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def _largest_component(passable):
    """返回可通行区域最大连通分量（8 邻域）的 cell 坐标列表 [(x,y), ...]。

    np.where 返回 (轴0索引=x, 轴1索引=y)，这里统一存成 (x, y) 顺序，
    与 cost[x, y] / astar 的索引保持一致。
    """
    from collections import deque
    xi, yi = np.where(passable)  # xi = 轴0(x), yi = 轴1(y)
    if len(xi) == 0:
        return []
    cells = set(zip(xi.tolist(), yi.tolist()))
    visited = set()
    best = []
    for s in cells:
        if s in visited:
            continue
        comp = []
        q = deque([s])
        visited.add(s)
        while q:
            c = q.popleft()
            comp.append(c)
            cx, cy = c
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    n = (cx + dx, cy + dy)
                    if n in cells and n not in visited:
                        visited.add(n)
                        q.append(n)
        if len(comp) > len(best):
            best = comp
    return best


def auto_endpoints(passable):
    """在最大连通可通行区域内自动选起终点（返回 (x0,y0),(x1,y1) cell 坐标）。"""
    comp = _largest_component(passable)
    if not comp:
        return None, None
    comp = np.array(comp)
    start = (int(comp[0, 0]), int(comp[0, 1]))
    d = (comp[:, 0] - start[0]) ** 2 + (comp[:, 1] - start[1]) ** 2
    k = int(np.argmax(d))
    goal = (int(comp[k, 0]), int(comp[k, 1]))
    return start, goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--semantic', required=True, help='语义 PLY（含 semantic_id）')
    ap.add_argument('--cell_size', type=float, default=1.0)
    ap.add_argument('--max_slope', type=float, default=25.0, help='最大可通行坡度角（度）')
    ap.add_argument('--max_rough', type=float, default=0.5, help='最大可通行粗糙度（z 标准差，米）')
    ap.add_argument('--min_ground_points', type=int, default=20)
    ap.add_argument('--max_points', type=int, default=8000000, help='降采样上限（大场景）')
    ap.add_argument('--start', nargs=2, type=float, help='起点世界坐标 x y')
    ap.add_argument('--goal', nargs=2, type=float, help='终点世界坐标 x y')
    ap.add_argument('--output', required=True, help='输出代价图 PNG 路径')
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

    stats = cell_stats(x, y, z, sem, args.cell_size, args.min_ground_points)
    if stats is None:
        print('无地面点，无法构建可通行性代价图。', flush=True)
        return
    cost, passable = build_costmap(stats, args.cell_size, args.max_slope, args.max_rough)
    nx, ny = cost.shape
    print(f'代价图 {nx}×{ny} cell，可通行 cell {int(passable.sum()):,} '
          f'({100.0 * passable.sum() / (nx * ny):.1f}%)', flush=True)

    # 起终点（世界坐标 → cell）
    if args.start is not None and args.goal is not None:
        sx, sy = int((args.start[0] - stats['xmin']) / args.cell_size), \
                 int((args.start[1] - stats['ymin']) / args.cell_size)
        gx, gy = int((args.goal[0] - stats['xmin']) / args.cell_size), \
                 int((args.goal[1] - stats['ymin']) / args.cell_size)
        start, goal = (sx, sy), (gx, gy)
    else:
        start, goal = auto_endpoints(passable)
    if start is None or goal is None:
        print('无可通行区域，无法规划。', flush=True)
        return
    print(f'起点 cell {start}，终点 cell {goal}', flush=True)

    path = astar(cost, start, goal)
    if path is None:
        print('未找到路径（起终点不连通）。', flush=True)
    else:
        path = np.array(path)
        n_path = len(path)
        # 路径上的坡度统计（用 stats slope 而非 cost，便于解释）
        path_slopes = stats['slope_deg'].reshape(nx, ny)[path[:, 0], path[:, 1]]
        # 世界坐标路径长度（米）
        wx = stats['xmin'] + (path[:, 0] + 0.5) * args.cell_size
        wy = stats['ymin'] + (path[:, 1] + 0.5) * args.cell_size
        seg = np.hypot(np.diff(wx), np.diff(wy))
        length = float(seg.sum())
        print(f'规划成功：{n_path} 个 cell，长度 {length:.1f} m，'
              f'路径平均坡度 {path_slopes.mean():.1f}°，最大坡度 {path_slopes.max():.1f}°', flush=True)

    # ---- 可视化 ----
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    img = np.full((nx, ny, 3), 0.9, dtype=np.float32)  # 未知 = 浅灰
    # 可通行：绿（低代价）→ 黄（高代价/陡）
    if passable.any():
        c = cost[passable]
        cmap = plt.colormaps['viridis']
        norm = plt.Normalize(0, max(c.max(), 1e-6))
        cols = cmap(norm(c))
        img[passable] = cols[:, :3]
    # 障碍/超坡度：深灰
    slope2d = stats['slope_deg'].reshape(nx, ny)
    valid2d = stats['valid'].reshape(nx, ny)
    img[np.isinf(cost) & valid2d & (slope2d > args.max_slope)] = (0.2, 0.2, 0.2)
    ax.imshow(np.transpose(img, (1, 0, 2)), origin='lower',
              extent=[stats['xmin'], stats['xmin'] + nx * args.cell_size,
                      stats['ymin'], stats['ymin'] + ny * args.cell_size])
    if path is not None:
        ax.plot(wx, wy, 'r-', lw=2.5, label=f'路径 {length:.1f} m')
        ax.plot(wx[0], wy[0], 'o', color='cyan', ms=10, label='start')
        ax.plot(wx[-1], wy[-1], 's', color='magenta', ms=10, label='goal')
        ax.legend(loc='upper right')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(f'可通行性代价图 + 坡度路径（max_slope={args.max_slope}°）')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print(f'已保存代价图：{args.output}', flush=True)


if __name__ == '__main__':
    main()
