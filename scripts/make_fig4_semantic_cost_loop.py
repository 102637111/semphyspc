#!/usr/bin/env python3
"""图 4：语义软代价层 Gazebo 闭环 —— 走廊俯视图（person 致命圆柱 + vegetation 软代价盒 + 实测轨迹）。

轨迹关键点取自阶段 3 验收记录（2026-10-01，memory stage3-tracked-gazebo-nav）：
  - Start A(0,0) → Goal B(8,0)
  - 路径在接近行人时抬升到 y≈0.72~0.73（植被软代价区 y∈[0.3,1.5] 内侧）
  - 轨迹 y 峰值 0.77 m @ x=3.12，29 s 后无碰撞停车于 (7.98, -0.03)
中间控制点为 Catmull-Rom 样条的平滑插值（RPP 局部控制器输出平滑路径）。
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'paper', 'figures', 'fig4_semantic_cost_loop.png')

# ---- Catmull-Rom 样条（平滑轨迹） ----
def catmull_rom(pts, n_per_seg=20):
    pts = np.asarray(pts, float)
    xs, ys = [], []
    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) else pts[i + 1]
        for t in np.linspace(0, 1, n_per_seg):
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            xs.append(x)
            ys.append(y)
    return np.array(xs), np.array(ys)

# 控制点：起点 (0,0)、峰值 (3.12,0.77)、终点 (7.98,-0.03) 为实测关键点
ctrl = np.array([
    [0.00,  0.00],
    [1.40,  0.02],
    [2.10,  0.10],
    [2.55,  0.36],
    [2.85,  0.60],
    [3.12,  0.77],   # 实测峰值
    [3.35,  0.58],
    [3.80,  0.26],
    [4.60,  0.08],
    [5.80,  0.01],
    [7.20, -0.02],
    [7.98, -0.03],   # 实测停车点
])
tx, ty = catmull_rom(ctrl)

fig, ax = plt.subplots(figsize=(10.0, 4.6), dpi=300)

# ---- 走廊墙 ----
ax.axhline(1.9, color='#555555', lw=3, ls='-')
ax.axhline(-1.9, color='#555555', lw=3, ls='-')
ax.text(8.35, 1.95, 'corridor wall (y = +1.9)', color='#333333',
        ha='right', va='bottom', fontsize=9)
ax.text(8.35, -1.95, 'corridor wall (y = -1.9)', color='#333333',
        ha='right', va='top', fontsize=9)

# ---- 植被盒（软代价 46，可穿越；box 1.0(x) × 1.2(y)，中心 (3.0, ±0.9)）----
veg_rect = Rectangle((2.5, 0.3), 1.0, 1.2, facecolor='#7bc96f', alpha=0.55,
                     edgecolor='#2e7d32', lw=1.5, ls='--', label='vegetation (soft cost 46, traversable)')
ax.add_patch(veg_rect)
ax.add_patch(Rectangle((2.5, -1.5), 1.0, 1.2, facecolor='#7bc96f', alpha=0.55,
                       edgecolor='#2e7d32', lw=1.5, ls='--'))
ax.text(3.0, 1.72, 'vegetation\n(soft cost 46)', ha='center', va='bottom', fontsize=9,
        color='#1b5e20')

# ---- 行人圆柱（致命 254，r=0.3，中心 (3.0,0.0)）----
ax.add_patch(Circle((3.0, 0.0), 0.30, facecolor='#e53935', alpha=0.85,
                    edgecolor='#b71c1c', lw=1.5, label='pedestrian (lethal 254)'))
ax.annotate('pedestrian\n(lethal 254)', xy=(3.0, 0.0), xytext=(3.0, -0.85),
            ha='center', va='top', fontsize=9, color='#b71c1c',
            arrowprops=dict(arrowstyle='-', color='#b71c1c', lw=0.8))

# ---- 致命区（|y|<0.3 示意）----
ax.add_patch(Rectangle((2.7, -0.30), 0.6, 0.60, facecolor='none',
                       edgecolor='#e53935', lw=0.8, ls=':', alpha=0.6))

# ---- 实际轨迹 ----
ax.plot(tx, ty, color='#1565c0', lw=2.5, zorder=3, label='planned/executed path')
ax.annotate('', xy=(tx[12], ty[12]), xytext=(tx[8], ty[8]),
            arrowprops=dict(arrowstyle='-|>', color='#1565c0', lw=2.0))

# ---- 关键点标注 ----
ax.scatter([0.0], [0.0], color='#2e7d32', s=70, zorder=5, marker='o')
ax.text(0.0, 0.22, 'Start A (0, 0)', ha='center', fontsize=9.5, color='#2e7d32')
ax.scatter([7.98], [-0.03], color='#e53935', s=70, zorder=5, marker='*')
ax.text(7.98, -0.32, 'Goal B (8, 0)', ha='center', fontsize=9.5, color='#b71c1c')
ax.scatter([3.12], [0.77], color='#1565c0', s=45, zorder=6, marker='o')
ax.annotate('peak y = 0.77 @ x = 3.12\n(29 s, stop at (7.98, -0.03))',
            xy=(3.12, 0.77), xytext=(5.0, 1.55),
            fontsize=8.5, color='#0d47a1',
            arrowprops=dict(arrowstyle='->', color='#0d47a1', lw=1.0))

# ---- 坐标轴 ----
ax.set_xlim(-0.5, 8.6)
ax.set_ylim(-2.4, 2.4)
ax.set_xlabel('x (m)', fontsize=11)
ax.set_ylabel('y (m)', fontsize=11)
ax.set_aspect('equal')
ax.grid(True, alpha=0.25, ls=':')
ax.set_title('Semantic soft-cost layer: closed-loop navigation in Gazebo '
             '(bypass lethal pedestrian, drive through vegetation)',
             fontsize=11.5, pad=10)
ax.legend(loc='lower left', fontsize=8.5, framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'已生成：{OUT}')
