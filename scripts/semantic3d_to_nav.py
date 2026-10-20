#!/usr/bin/env python3
"""Semantic3D 语义地图 → Nav2 输入：裁剪子区域 + 生成静态地图(PGM/YAML) + 语义软代价。

把 infer_map.py 输出的 Semantic3D 完整语义 PLY（3520 万点级，带 semantic_id）裁成适合 Nav2
的子区域，并产出两样东西：
  1. 静态地图 <out>.pgm + <out>.yaml —— map_server 撑 global costmap 尺寸。
     投影规则（soft_cost，本文）：cell 内有不可穿越点(lethal 行人 / hard 车墙建筑) → occupied；
     可穿越点(植被/护栏/地面) → free；无点 → unknown。植被的可穿越软代价(30~80)由语义软代价层标。
  2. 语义代价 —— 逐 cell 取 max(class_to_cost)，写 <out>_cost.npz（grid/origin/resolution），
     由 semantic_costmap_publisher 的 --cost-npz 模式发布（植被软代价 30~80、不可穿越类 254）。

关键（方案 a 修正后）：物理上「不可穿越」的类（lethal 行人 + hard 车墙建筑，撞上会停）都标
绝对障碍 254；只有「可穿越」的类（frangible 植被护栏 / deformable 可变形物）标连续软代价
30~80（能绕就绕、绕不开才穿）。这正是论文「语义物理可穿越性」的落点——不再把 hard 当软代价
（B③ 权衡曲线证明软代价 <253 会让 Navfn 穿实心车/墙/建筑区）。

用法：
  python3 scripts/semantic3d_to_nav.py \
      -i paper/figures/map/domfountain_station3_map_semantic.ply \
      -o /tmp/s3_nav/dom  --xmin -25 --xmax 25 --ymin -25 --ymax 25 --cell 0.1
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.materials import class_physics, SAFETY_IDS  # noqa: E402

_DT = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('semantic_id', '<i4')])


def class_to_cost(cid, hard_ceil=220, soft_low=30, soft_high=80, lethality_scale=1.0):
    ph = class_physics(int(cid))
    s = ph['safety']
    if s == 'passable':
        return 0
    if s == 'lethal':
        return 254
    if s == 'hard':
        # 方案(a)：车/墙/建筑 物理上不可穿越（撞上会停，非穿过去），标绝对障碍 254。
        # 不再用软代价 185~220——B③ 权衡曲线证明软代价(<253)会让 Navfn 穿实心障碍区（穿 926-cell 连续车/建筑区）。
        return 254
    # frangible(植被/护栏) / deformable(可变形物) → soft_low~soft_high 可穿越软代价（能绕就绕、绕不开才穿）
    # B④ 材料敏感性：soft_low/soft_high 是软代价上下界先验、lethality_scale 是材质库手工 lethality 的缩放
    # （扰动它们量化「材料先验 → 导航代价」的敏感性）。
    L = float(ph['lethality']) * lethality_scale
    v = soft_low + (L - 0.05) / 0.45 * (soft_high - soft_low)
    return int(np.clip(v, 0, 253))


def make_cost_table(hard_ceil=220, soft_low=30, soft_high=80, lethality_scale=1.0):
    return np.array([class_to_cost(c, hard_ceil, soft_low, soft_high, lethality_scale)
                     for c in range(32)], dtype=np.int32)


def eval_cost(cid):
    """评估尺子（独立于 planner 代价）：按安全等级区分 lethal/hard/soft/free，度量路径质量。

    与 class_to_cost 的区别：planner 代价把 hard 与 lethal 都标 254（方案 a，都不可穿越），
    评估时则保留 hard=200 与 lethal=254 的区分，以单独报告「穿行人」vs「穿车墙」。
    """
    ph = class_physics(int(cid))
    s = ph['safety']
    if s == 'passable':
        return 0
    if s == 'lethal':
        return 254
    if s == 'hard':
        return 200
    return 46   # frangible/deformable 软目标


# 碰撞安全等级 id：lethal=0, hard=1, frangible=2, deformable=3, passable=4
SAFETY_TABLE = np.array([SAFETY_IDS[class_physics(c)['safety']] for c in range(32)], dtype=np.int32)


def read_semantic_ply(path):
    with open(path, 'rb') as f:
        n = 0
        while True:
            line = f.readline().decode('ascii', 'ignore').strip()
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            elif line == 'end_header':
                break
        arr = np.fromfile(f, dtype=_DT, count=n)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True, help='Semantic3D 语义 PLY')
    ap.add_argument('-o', '--out', required=True, help='输出前缀（生成 <out>.pgm/.yaml/_cost.npz）')
    ap.add_argument('--xmin', type=float, required=True)
    ap.add_argument('--xmax', type=float, required=True)
    ap.add_argument('--ymin', type=float, required=True)
    ap.add_argument('--ymax', type=float, required=True)
    ap.add_argument('--cell', type=float, default=0.1, help='栅格分辨率（米）')
    ap.add_argument('--dilate', type=int, default=2, help='free 区形态学膨胀 cell 数（填补点云空洞）')
    ap.add_argument('--mode', choices=['soft_cost', 'geometric', 'semantic_ban'], default='soft_cost',
                    help='障碍判定模式：soft_cost=不可穿越类(lethal+hard) occupied + 植被软代价（本文）；'
                         'geometric=非地面 occupied（几何障碍 baseline）；'
                         'semantic_ban=lethal+hard occupied + 植被 free（简单语义禁行 baseline）')
    ap.add_argument('--hard-ceil', type=int, default=220,
                    help='soft_cost 下 hard 类代价上界（185~254，B③ 权衡曲线扫描；220=默认）')
    ap.add_argument('--soft-low', type=int, default=30,
                    help='可穿越软代价下界（frangible/deformable，B④ 材料敏感性扫描；30=默认）')
    ap.add_argument('--soft-high', type=int, default=80,
                    help='可穿越软代价上界（frangible/deformable，B④ 材料敏感性扫描；80=默认）')
    ap.add_argument('--lethality-scale', type=float, default=1.0,
                    help='材质库手工 lethality 的缩放（B④ 材料敏感性：0.5/1.0/1.5；1.0=默认）')
    ap.add_argument('--cost-only', action='store_true',
                    help='只写语义代价 npz，跳过静态地图 pgm/yaml（B③ 复用同一 static map）')
    ap.add_argument('--dry-run', action='store_true', help='只统计不写文件')
    args = ap.parse_args()

    arr = read_semantic_ply(args.input)
    xyz = np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float32)
    sem = arr['semantic_id'].astype(np.int64)
    del arr

    x, y = xyz[:, 0], xyz[:, 1]
    m = (x >= args.xmin) & (x < args.xmax) & (y >= args.ymin) & (y < args.ymax)
    x, y, sem = x[m], y[m], sem[m]
    print(f'子区域 x[{args.xmin},{args.xmax}) y[{args.ymin},{args.ymax})：{len(sem):,} 点')

    # 投影 2D 网格（y-major，row0 = y 最大）
    cell = args.cell
    x0, y0 = args.xmin, args.ymin
    nx = int(np.ceil((args.xmax - x0) / cell))
    ny = int(np.ceil((args.ymax - y0) / cell))
    gx = np.floor((x - x0) / cell).astype(np.int64)
    gy = np.floor((y - y0) / cell).astype(np.int64)
    inside = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    gx, gy, sem = gx[inside], gy[inside], sem[inside]

    pt_cost = make_cost_table(args.hard_ceil, args.soft_low, args.soft_high, args.lethality_scale)[sem]
    safety = SAFETY_TABLE[sem]
    idx = gy * nx + gx
    # 语义代价：逐 cell max（person 254 覆盖植被 46 覆盖地面 0）；-1=无点(unknown)
    cell_cost = np.full(nx * ny, -1, dtype=np.int32)
    np.maximum.at(cell_cost, idx, pt_cost)
    # 障碍判定（按 mode）：lethal=0, hard=1, frangible=2, deformable=3, passable=4
    if args.mode == 'geometric':
        occ_mask = safety != 4          # 非地面即障碍（纯几何，无语义）
    elif args.mode == 'semantic_ban':
        occ_mask = safety <= 1          # lethal + hard 二元禁行，植被/地面 free（无语义代价）
    else:  # soft_cost（本文，方案 a 修正）
        occ_mask = safety <= 1          # 不可穿越类（lethal+hard）标 occupied；可穿越类（植被/地面）靠软代价层标
    cell_hard = np.zeros(nx * ny, dtype=bool)
    np.maximum.at(cell_hard.view(np.int8), idx, occ_mask.astype(np.int8))
    cell_hard = cell_hard.view(np.bool_)
    # 有任意点（非 unknown）
    cell_any = np.zeros(nx * ny, dtype=bool)
    np.maximum.at(cell_any.view(np.int8), idx, np.ones(len(sem), dtype=np.int8))
    cell_any = cell_any.view(np.bool_)

    cc = cell_cost.reshape(ny, nx)
    free = (cc == 0).sum()
    soft = ((cc > 0) & (cc < 150)).sum()      # 可穿越软代价（植被/护栏 30~80）
    obstacle = (cc >= 254).sum()              # 不可穿越（行人 + 车墙建筑，方案 a 都 254）
    unknown = int((~cell_any).sum())
    print(f'栅格 {nx}x{ny} @{cell}m')
    print(f'语义代价分层：free={free} soft={soft} obstacle={obstacle} unknown={unknown}')

    # free 连通性（道路可通行区）
    from scipy import ndimage
    fm = cc == 0
    lbl, nl = ndimage.label(fm)
    sizes = ndimage.sum(fm, lbl, range(1, nl + 1))
    if nl:
        print(f'free 连通块 {nl} 个，最大 {int(sizes.max())} cell（{sizes.max() * cell * cell:.0f}m²）')

    if args.dry_run:
        return

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    # ---- 后处理：可穿越区 = 所有有点但非不可穿越（植被/护栏/地面都 free，
    #      由语义软代价层标植被连续代价 30~80；静态地图把 lethal+hard 标 occupied）----
    _hard = cell_hard.reshape(ny, nx)
    _any = cell_any.reshape(ny, nx)
    _free_final = _any & ~_hard
    _any_final = _any

    # ---- 静态地图 PGM：黑(0)=occupied, 白(254)=free, 灰(205)=unknown ----
    # 注意 map_server 约定：occ=(255-p)/255，p>205→free、p<89→occupied、中间→unknown
    # （与 gen_track_map.py 一致：FREE=254/OCC=0，切勿反置，否则 map_server 全盘翻转）
    # --cost-only：B③ 权衡曲线复用同一 static map，跳过 pgm/yaml。
    if not args.cost_only:
        pgm = np.full((ny, nx), 205, dtype=np.uint8)  # unknown
        pgm[_hard] = 0                                # occupied（黑）
        pgm[_free_final] = 254                        # free（白）
        with open(args.out + '.pgm', 'wb') as f:
            f.write(f'P5\n{nx} {ny}\n255\n'.encode())
            # PGM 图像 row0=top=y最大，而 pgm 数组 row0=gy0=y最小 → 上下翻转
            f.write(pgm[::-1].tobytes())
        with open(args.out + '.yaml', 'w') as f:
            f.write(f'image: {args.out}.pgm\n')
            f.write('mode: trinary\n')
            f.write(f'resolution: {cell}\n')
            f.write(f'origin: [{x0}, {y0}, 0.0]\n')
            f.write('negate: 0\n')
            f.write('occupied_thresh: 0.65\n')
            f.write('free_thresh: 0.196\n')
        print(f'静态地图 -> {args.out}.pgm + {args.out}.yaml'
              f'（occupied={int(_hard.sum())}, free={int(_free_final.sum())}, unknown={int((~_any_final).sum())}）')

    # ---- 语义代价 npz（供 semantic_costmap_publisher --cost-npz 发布）----
    # 仅 soft_cost 模式需要连续语义代价层；geometric/semantic_ban 是纯二元障碍 baseline，
    # 不写 cost.npz（launch 侧对这两种 mode 不挂 semantic_pub）。
    if args.mode == 'soft_cost':
        np.savez_compressed(args.out + '_cost.npz',
                            cost=cc.astype(np.int16), origin=np.array([x0, y0], np.float32),
                            resolution=np.float32(cell))
        print(f'语义代价 -> {args.out}_cost.npz')
        # 评估尺子（独立于 planner 代价，区分 lethal/hard/soft/free），供 plan_and_analyze 度量路径质量
        eval_table = np.array([eval_cost(c) for c in range(32)], dtype=np.int32)
        cell_eval = np.full(nx * ny, -1, dtype=np.int32)
        np.maximum.at(cell_eval, idx, eval_table[sem])
        np.savez_compressed(args.out + '_eval.npz',
                            cost=cell_eval.reshape(ny, nx).astype(np.int16),
                            origin=np.array([x0, y0], np.float32), resolution=np.float32(cell))
        print(f'评估网格 -> {args.out}_eval.npz')


if __name__ == '__main__':
    main()
