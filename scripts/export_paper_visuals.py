#!/usr/bin/env python3
"""导出论文效果图 + 可 3D 软件查看的彩色点云。

三种着色，各出 PLY（CloudCompare / MeshLab / 3D 查看器打开即彩色）+ PNG（论文/docx 插图）：
  1. 真实 RGB   —— 点云原始相机颜色（真实还原，解决 Unity 单色问题）
  2. 语义识别   —— 32 类 HSV 配色 + semantic_id 标量（"识别物体类型种类"）
  3. 物理安全   —— 5 级碰撞安全配色 + safety_id/lethality 等标量（"标记地图"）

语义来源：默认用 npz 自带 sem（真值，展示标注质量）；
         给 --config + --ckpts 时用 PointMambaSeg 推理（展示"识别"能力）。

用法：
  # 真值语义（最快，不需要 GPU）
  python scripts/export_paper_visuals.py --input <npz> --output_dir paper/figures/effect
  # 模型预测语义
  python scripts/export_paper_visuals.py --input <npz> --output_dir paper/figures/effect \
      --config experiments/.../config.yaml --ckpts experiments/.../ckpt-best.pth
"""
import os
import sys
import argparse
import colorsys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.materials import LIDARSEG_32, COLLISION_SAFETY, apply_materials


# ---------------- 配色 ----------------
def _hsv_palette(n):
    colors = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        colors[i] = np.array(colorsys.hsv_to_rgb(i / max(n, 1), 0.75, 0.95), dtype=np.float32)
    return colors


SEM_PALETTE = _hsv_palette(len(LIDARSEG_32))
SAFETY_COLORS = {
    'lethal': (1.00, 0.10, 0.10),
    'hard': (1.00, 0.50, 0.05),
    'frangible': (1.00, 0.90, 0.10),
    'deformable': (0.05, 0.75, 0.10),
    'passable': (0.60, 0.60, 0.60),
}


# ---------------- 二进制 PLY 写出（RGB 顶点色 + 多标量字段）----------------
def write_ply_binary(path, xyz, colors, scalars=None):
    n = len(xyz)
    xyz = np.asarray(xyz, dtype=np.float32).reshape(n, 3)
    colors = np.clip(np.asarray(colors, dtype=np.float32).reshape(n, 3), 0.0, 1.0)
    scalars = scalars or {}

    header = ['ply', 'format binary_little_endian 1.0', f'element vertex {n}',
              'property float x', 'property float y', 'property float z',
              'property uchar red', 'property uchar green', 'property uchar blue']
    for name, (dt, _) in scalars.items():
        header.append(f'property {dt} {name}')
    header.append('end_header\n')

    rgb = (colors * 255).astype(np.uint8)
    fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
              ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]
    np_dt = {'int': '<i4', 'float': '<f4', 'uint': '<u4'}
    for name, (dt, arr) in scalars.items():
        fields.append((name, np_dt[dt]))
    rec = np.empty(n, dtype=np.dtype(fields))
    rec['x'], rec['y'], rec['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec['r'], rec['g'], rec['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    for name, (_, arr) in scalars.items():
        rec[name] = arr.reshape(n)

    with open(path, 'wb') as f:
        f.write('\n'.join(header).encode('ascii'))
        rec.tofile(f)


# ---------------- 分割推理（修正 forward 返回 tuple 的解包）----------------
def run_seg_inference(xyz, rgb, config_path, ckpt_path, npoints):
    import torch
    # 编译的 CUDA 扩展（chamfer/emd 的 .so）位于 extensions/*/ 目录，需加入 sys.path
    # 否则 models/__init__.py 链式 import PointMambaScan 时 `import chamfer` 失败
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _ext in ('chamfer_dist', 'emd'):
        _p = os.path.join(_root, 'extensions', _ext)
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from tools import builder
    from utils.config import get_config

    class _A:
        pass
    args = _A()
    args.config = config_path
    args.local_rank = 0
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.resume = False
    args.experiment_path = '/tmp/export_vis_exp'
    os.makedirs(args.experiment_path, exist_ok=True)
    args.log_name = os.path.basename(config_path).replace('.yaml', '')
    os.environ.setdefault('LOCAL_RANK', '0')

    from utils.logger import get_root_logger
    logger = get_root_logger(name='export_vis')
    config = get_config(args, logger=logger)

    in_channels = getattr(config.model, 'in_channels', 6)
    pts = np.concatenate([xyz, rgb], axis=1) if (in_channels >= 6 and rgb is not None) else xyz
    if pts.shape[1] < in_channels:
        pts = np.concatenate([pts, np.zeros((len(pts), in_channels - pts.shape[1]), np.float32)], axis=1)

    from pointnet2_ops import pointnet2_utils
    if len(pts) > npoints:
        t = torch.from_numpy(pts[:, :3].astype(np.float32)).cuda().unsqueeze(0)
        idx = pointnet2_utils.furthest_point_sample(t, npoints).squeeze(0).cpu().numpy()
        pts = pts[idx]
    elif len(pts) < npoints:
        pts = np.concatenate([pts, np.zeros((npoints - len(pts), pts.shape[1]), np.float32)], axis=0)

    model = builder.model_builder(config.model)
    builder.load_model(model, ckpt_path, logger=logger)
    model = torch.nn.DataParallel(model).cuda().eval()
    batch = torch.from_numpy(pts).float().unsqueeze(0).cuda()
    with torch.no_grad():
        ret = model(batch)
    # PointMambaSeg.forward 返回 (logits, mat_logits[, inst])，取 logits
    logits = ret[0] if isinstance(ret, (tuple, list)) else ret
    sem = logits.argmax(-1).squeeze(0).cpu().numpy().astype(np.int64)
    return pts[:, :3], pts[:, 3:6], sem


# ---------------- 渲染（俯视 BEV + 图例）----------------
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


def render_bev(xyz, colors, out_path, title, legend_items=None, mask=None):
    plt = _plt()
    from matplotlib.patches import Patch
    if mask is not None:
        xyz = xyz[mask]
        colors = colors[mask]
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.scatter(xyz[:, 0], xyz[:, 1], c=colors, s=1.5, rasterized=True, linewidths=0)
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
    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def render_legend(items, out_path, title, ncol=2):
    plt = _plt()
    from matplotlib.patches import Patch
    n = len(items)
    fig, ax = plt.subplots(figsize=(9, max(3, n / ncol * 0.5)))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    handles = [Patch(facecolor=c, label=name) for name, c in items]
    leg = ax.legend(handles=handles, loc='center', fontsize=8, ncol=ncol, frameon=False)
    for t in leg.get_texts():
        t.set_color('#c9d1d9')
    ax.set_title(title, fontsize=12, color='#e6edf3')
    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--ckpts', default=None)
    ap.add_argument('--npoints', type=int, default=16384)
    ap.add_argument('--no_rgb_render', action='store_true', help='跳过真实 RGB 渲染（无 rgb 时）')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    d = np.load(args.input)
    xyz = d['xyz'].astype(np.float32)
    rgb = d.get('rgb', None)
    if rgb is not None:
        rgb = rgb.astype(np.float32).reshape(len(xyz), 3)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
    gt_sem = d.get('sem', None)
    if gt_sem is not None:
        gt_sem = gt_sem.astype(np.int64).reshape(-1)

    stem = os.path.splitext(os.path.basename(args.input))[0]

    # 语义来源：推理 > 真值
    sem_src = 'gt'
    if args.config and args.ckpts:
        if rgb is None:
            raise ValueError('模型推理需要 rgb（6 通道输入）')
        xyz, rgb, sem = run_seg_inference(xyz, rgb, args.config, args.ckpts, args.npoints)
        sem_src = 'pred'
    elif gt_sem is not None:
        sem = gt_sem
    else:
        raise ValueError('npz 需含 sem，或提供 --config + --ckpts 做推理')

    mat = apply_materials(sem)

    # 1) 真实 RGB PLY + PNG
    if rgb is not None and not args.no_rgb_render:
        write_ply_binary(os.path.join(args.output_dir, f'{stem}_rgb.ply'), xyz, rgb)
        mask = (rgb.sum(axis=1) > 0.05)
        render_bev(xyz, rgb, os.path.join(args.output_dir, f'{stem}_rgb.png'),
                   f'{stem} — real RGB point cloud', mask=mask)

    # 2) 语义 PLY + PNG（带 semantic_id 标量）
    sem_color = np.zeros((len(xyz), 3), np.float32)
    for c in range(len(LIDARSEG_32)):
        sem_color[sem == c] = SEM_PALETTE[c]
    write_ply_binary(os.path.join(args.output_dir, f'{stem}_semantic.ply'), xyz, sem_color,
                     scalars={'semantic_id': ('int', sem.astype(np.int32))})
    present = np.unique(sem)
    sem_legend = [(_short_name(LIDARSEG_32[c]), SEM_PALETTE[c]) for c in present if c < len(LIDARSEG_32)]
    render_bev(xyz, sem_color, os.path.join(args.output_dir, f'{stem}_semantic.png'),
               f'{stem} — semantic segmentation ({"pred" if sem_src == "pred" else "GT"})',
               legend_items=sem_legend)

    # 3) 安全 PLY + PNG（带物理标量）
    safety_color = np.zeros((len(xyz), 3), np.float32)
    for si, sname in enumerate(COLLISION_SAFETY):
        safety_color[mat['safety_id'] == si] = SAFETY_COLORS[sname]
    write_ply_binary(os.path.join(args.output_dir, f'{stem}_safety.ply'), xyz, safety_color,
                     scalars={'safety_id': ('int', mat['safety_id']),
                              'physical_id': ('int', mat['physical_id']),
                              'lethality': ('float', mat['lethality']),
                              'destructibility': ('float', mat['destructibility']),
                              'friction': ('float', mat['friction'])})
    safety_legend = [(s, SAFETY_COLORS[s]) for s in COLLISION_SAFETY]
    render_bev(xyz, safety_color, os.path.join(args.output_dir, f'{stem}_safety.png'),
               f'{stem} — collision safety level', legend_items=safety_legend)

    # 全局图例
    render_legend([(_short_name(n), SEM_PALETTE[i]) for i, n in enumerate(LIDARSEG_32)],
                  os.path.join(args.output_dir, 'legend_semantic.png'), '32-class semantic palette', ncol=3)
    render_legend(safety_legend, os.path.join(args.output_dir, 'legend_safety.png'),
                  '5-level collision safety palette', ncol=1)

    print(f'[{stem}] semantic={sem_src}  points={len(xyz):,}  classes={len(present)}')
    print(f'  classes: ' + ', '.join(_short_name(LIDARSEG_32[c]) for c in present))
    print(f'  output: {args.output_dir}/')
    for f in sorted(os.listdir(args.output_dir)):
        if f.startswith(stem):
            print(f'    - {f}')


if __name__ == '__main__':
    main()
