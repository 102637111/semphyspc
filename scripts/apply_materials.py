#!/usr/bin/env python3
"""
把分割后的点云按语义类别赋【物理交互属性】（碰撞安全等级 / 致命性 / 可破坏性 / 摩擦），
用于：① 导航失控安全碰撞决策  ② 游戏碰撞/损毁参数  ③ 材质化可视化。

输入：
  A. 带语义标签的数据集样本（.npz，含 sem）            —— 纯属性应用
  B. 原始点云（.ply/.pcd/.npy）+ PointMambaSeg 模型     —— 先推理分割再赋属性

输出（每个样本）：
  <stem>_safety.ply   : 顶点色=碰撞安全等级(红=致命/橙=硬/黄=可破坏/绿=可变形/灰=可通过)
                        附带标量 safety_id / lethality / destructibility / friction / physical_id
  <stem>_semantic.ply : 顶点色=语义配色（检查分割质量）
  输出目录 properties.json : 完整属性库（物理材质 + 视觉材质 + 类别映射 + 安全等级）

用法：
    python scripts/apply_materials.py --input data/nuscenes_seg_full/val/ --output_dir out_physics
    python scripts/apply_materials.py --input cloud.ply \
        --config cfgs/seg_nuscenes_full.yaml --ckpts <ckpt-best.pth> --output_dir out_physics
    # 附带失控软碰撞决策
    python scripts/apply_materials.py --input <npz> --output_dir out --soft_contact --heading 0
"""
import os, sys, argparse, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.materials import (LIDARSEG_32, COLLISION_SAFETY, class_meta,
                             apply_materials, export_library, select_soft_contact)


# ---------------- 语义 / 安全等级配色 ----------------
def _hsv_palette(n):
    import colorsys
    colors = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        h = (i * 0.61803398875) % 1.0
        colors[i] = np.array(colorsys.hsv_to_rgb(h, 0.75, 0.95), dtype=np.float32)
    return colors


SEM_PALETTE = _hsv_palette(len(LIDARSEG_32))

# 碰撞安全等级 → 颜色（红=致命 / 橙=硬 / 黄=可破坏 / 绿=可变形 / 灰=可通过）
SAFETY_COLORS = {
    'lethal':    (1.00, 0.10, 0.10),
    'hard':      (1.00, 0.50, 0.05),
    'frangible': (1.00, 0.90, 0.10),
    'deformable':(0.05, 0.75, 0.10),
    'passable':  (0.60, 0.60, 0.60),
}


# ---------------- 二进制 PLY 写出（多标量字段） ----------------
def write_ply_binary(path, xyz, colors, scalars=None):
    """写二进制 PLY：x y z red green blue + 任意标量字段。

    scalars: OrderedDict(name -> (numpy dtype str, array))
    """
    n = len(xyz)
    xyz = np.asarray(xyz, dtype=np.float32).reshape(n, 3)
    colors = np.asarray(colors, dtype=np.float32).reshape(n, 3)
    colors = np.clip(colors, 0.0, 1.0)
    scalars = scalars or {}

    header = ['ply', 'format binary_little_endian 1.0', f'element vertex {n}',
              'property float x', 'property float y', 'property float z',
              'property uchar red', 'property uchar green', 'property uchar blue']
    for name, (dt, _) in scalars.items():
        header.append(f'property {dt} {name}')
    header.append('end_header\n')
    header = '\n'.join(header).encode('ascii')

    rgb = (colors * 255).astype(np.uint8)

    # 组装结构化数组
    fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
              ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]
    for name, (dt, arr) in scalars.items():
        np_dt = {'int': '<i4', 'float': '<f4', 'uint': '<u4'}[dt]
        fields.append((name, np_dt))
    rec = np.empty(n, dtype=np.dtype(fields))
    rec['x'], rec['y'], rec['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec['r'], rec['g'], rec['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    for name, (_, arr) in scalars.items():
        rec[name] = arr.reshape(n)

    with open(path, 'wb') as f:
        f.write(header)
        rec.tofile(f)


# ---------------- 数据加载 / 推理 ----------------
def load_npz_sem(path):
    d = np.load(path)
    return d['xyz'].astype(np.float32), d['sem'].astype(np.int64).reshape(-1)


def load_point_cloud(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        a = np.load(path).astype(np.float32)
        return a[:, :3], (a[:, 3:6] if a.shape[1] >= 6 else None)
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors, dtype=np.float32)
    rgb = cols if (len(cols) == len(xyz) and cols.max() > 0) else None
    return xyz, rgb


def run_seg_inference(xyz, rgb, config_path, ckpt_path, npoints):
    import torch
    from tools import builder
    from utils.config import get_config

    class _A: pass
    args = _A()
    args.config = config_path
    args.local_rank = 0
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.resume = False
    args.experiment_path = '/tmp/apply_mat_exp'
    os.makedirs(args.experiment_path, exist_ok=True)
    args.log_name = os.path.basename(config_path).replace('.yaml', '')
    os.environ['LOCAL_RANK'] = '0'
    from utils.logger import get_root_logger
    logger = get_root_logger(name='apply_mat')
    config = get_config(args, logger=logger)

    in_channels = getattr(config.model, 'in_channels', 6)
    pts = np.concatenate([xyz, rgb], axis=1) if (in_channels >= 6 and rgb is not None) else xyz
    if pts.shape[1] < in_channels:
        pts = np.concatenate([pts, np.zeros((len(pts), in_channels - pts.shape[1]), dtype=np.float32)], axis=1)

    from pointnet2_ops import pointnet2_utils
    if len(pts) > npoints:
        t = torch.from_numpy(pts[:, :3].astype(np.float32)).cuda().unsqueeze(0)
        idx = pointnet2_utils.furthest_point_sample(t, npoints).squeeze(0).cpu().numpy()
        pts = pts[idx]
    elif len(pts) < npoints:
        pts = np.concatenate([pts, np.zeros((npoints - len(pts), pts.shape[1]), dtype=np.float32)], axis=0)

    model = builder.model_builder(config.model)
    builder.load_model(model, ckpt_path, logger=logger)
    model = torch.nn.DataParallel(model).cuda().eval()
    batch = torch.from_numpy(pts).float().unsqueeze(0).cuda()
    with torch.no_grad():
        logits = model(batch)
    sem = logits.argmax(-1).squeeze(0).cpu().numpy().astype(np.int64)
    return pts[:, :3], sem


# ---------------- 主流程 ----------------
def process_one(src, out_dir, args):
    stem = os.path.splitext(os.path.basename(src))[0]
    ext = os.path.splitext(src)[1].lower()

    if ext == '.npz':
        xyz, sem = load_npz_sem(src)
    elif ext in ('.ply', '.pcd', '.npy') and args.config:
        xyz, rgb = load_point_cloud(src)
        xyz, sem = run_seg_inference(xyz, rgb, args.config, args.ckpts, args.npoints)
    else:
        raise ValueError(f'无法处理 {src}：npz 需含 sem；ply/pcd/npy 需同时给 --config 与 --ckpts')

    mat = apply_materials(sem)

    # 安全等级配色（红/橙/黄/绿/灰）
    safety_color = np.zeros((len(xyz), 3), dtype=np.float32)
    for si, sname in enumerate(COLLISION_SAFETY):
        safety_color[mat['safety_id'] == si] = SAFETY_COLORS[sname]

    # 语义配色
    sem_color = np.zeros((len(xyz), 3), dtype=np.float32)
    for c in range(len(LIDARSEG_32)):
        sem_color[sem == c] = SEM_PALETTE[c]

    write_ply_binary(os.path.join(out_dir, f'{stem}_safety.ply'), xyz, safety_color,
                     scalars={'safety_id': ('int', mat['safety_id']),
                              'physical_id': ('int', mat['physical_id']),
                              'lethality': ('float', mat['lethality']),
                              'destructibility': ('float', mat['destructibility']),
                              'friction': ('float', mat['friction'])})
    write_ply_binary(os.path.join(out_dir, f'{stem}_semantic.ply'), xyz, sem_color, None)

    # 软碰撞决策
    contact = None
    if args.soft_contact:
        contact = select_soft_contact(sem, xyz, heading=args.heading,
                                      max_range=args.max_range, half_fov_deg=args.half_fov)
    return dict(stem=stem, n_points=len(xyz), contact=contact)


def _print_contact(stem, contact):
    print(f'  [{stem}] 失控软碰撞候选（前向视锥，heading={contact["ego_heading"]:.2f}rad）：')
    if not contact['candidates']:
        print('    前向视锥内无可安全撞击目标（只有致命/硬刚性物体）')
        return
    for c in contact['candidates']:
        print(f'    - {c["class_name"]:<36} 致命性 {c["lethality"]:.2f}  点 {c["count"]:>6}  '
              f'均值距离 {c["mean_dist"]:.1f}m')
    r = contact['recommended']
    if r:
        print(f'    ★ 推荐撞击：{r["class_name"]}（安全等级 {r["safety"]}，'
              f'致命性 {r["lethality"]:.2f}，距离 {r["mean_dist"]:.1f}m）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='npz / ply / pcd / npy，或目录')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--config', default=None, help='分割模型 config（对原始点云推理时必填）')
    ap.add_argument('--ckpts', default=None, help='分割模型 ckpt（对原始点云推理时必填）')
    ap.add_argument('--npoints', type=int, default=16384, help='推理时的采样点数')
    ap.add_argument('--soft_contact', action='store_true', help='输出失控软碰撞决策')
    ap.add_argument('--heading', type=float, default=0.0, help='ego 航向（弧度，软碰撞用）')
    ap.add_argument('--max_range', type=float, default=50.0, help='前向搜索距离（m）')
    ap.add_argument('--half_fov', type=float, default=40.0, help='前向视锥半角（度）')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'properties.json'), 'w') as f:
        json.dump(export_library(), f, indent=2, ensure_ascii=False)

    inputs = []
    if os.path.isdir(args.input):
        inputs = sorted([os.path.join(args.input, f) for f in os.listdir(args.input)
                         if f.lower().endswith(('.npz', '.ply', '.pcd', '.npy'))])
    elif os.path.isfile(args.input):
        inputs = [args.input]
    else:
        raise FileNotFoundError(args.input)

    print(f'处理 {len(inputs)} 个样本 → {args.output_dir}')
    for i, src in enumerate(inputs):
        try:
            r = process_one(src, args.output_dir, args)
            print(f'  [{i+1}/{len(inputs)}] {r["stem"]} ({r["n_points"]:,} points)')
            if args.soft_contact:
                _print_contact(r['stem'], r['contact'])
        except Exception as e:
            print(f'  [ERROR] {src}: {e}')
    print(f'\n完成。属性库 → {os.path.join(args.output_dir, "properties.json")}')


if __name__ == '__main__':
    main()
