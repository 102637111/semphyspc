#!/usr/bin/env python3
"""图 4（合并后）：真实地图语义软代价导航闭环三子图。

把原「图 4（Gazebo 闭环轨迹）+ 图 5（真实地图路径规划）」合并成一张三子图，
对应 IEEE Sensors Journal 8 页双栏的瘦身要求（图 9 → 6~7 张）。三子图对应
「原始点云 → 软代价 → 路径」的连贯叙事，全部取自同一真实场景
（Semantic3D domfountain_station3）：

  (a) 原始点云俯视          —— domfountain_station3_nav_bev_rgb.png
  (b) 语义软代价层俯视       —— domfountain_station3_nav_bev_cost.png（连续代价 0~254）
  (c) Nav2 规划路径         —— 路径规划地图.png（绕开行人、穿越植被）

Gazebo 闭环的轨迹图不保留（其文字描述已在正文 §4.6 完整覆盖），仿真证据让位于
真实地图这一更强证据。三张子图均保留各自原始渲染（含标题与图例），外部统一加
(a)(b)(c) 子图索引，供正文 caption 引用（与图 2 / 图 3 的 (a)(b)(c) 风格一致）。

用法（项目根 /home/saw/ros2_ws/src/semphyspc/ 下）：
    python3 scripts/make_fig_system_integration.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, 'paper', 'figures')

PATHS = [
    os.path.join(FIG, 'effect', 'domfountain_station3_nav_bev_rgb.png'),      # (a) 原始点云
    os.path.join(FIG, 'effect', 'domfountain_station3_nav_bev_cost.png'),     # (b) 软代价层
    os.path.join(FIG, 'effect', '路径规划地图.png'),                            # (c) Nav2 路径
]
LABELS = ['(a)', '(b)', '(c)']
OUT = os.path.join(FIG, 'fig4_system_integration.png')

imgs = [imread(p) for p in PATHS]
ratios = [im.shape[1] / im.shape[0] for im in imgs]  # 各图宽高比，保证不变形

fig, axes = plt.subplots(
    1, 3,
    figsize=(sum(ratios) * 2.7, 3.0),
    gridspec_kw={'width_ratios': ratios, 'wspace': 0.03},
)
for ax, img, lab in zip(axes, imgs, LABELS):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(lab, fontsize=13, fontweight='bold', pad=4)

fig.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
print(f'已生成三子图：{OUT}')

import PIL.Image
im = PIL.Image.open(OUT)
print(f'  尺寸 {im.size[0]}x{im.size[1]}  宽高比 {im.size[0]/im.size[1]:.2f}  ({os.path.getsize(OUT)/1024:.0f} KB)')
