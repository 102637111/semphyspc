#!/usr/bin/env python3
"""
分割点云 → 实例聚类 → 游戏物理/破坏参数导出（含位置相关破坏演示）。

流程：
  1. 逐点语义标签（npz 或 PointMambaSeg 推理）
  2. 实例聚类（utils/instances.py，DBSCAN 按类）
  3. 每实例构建破坏模型（utils/damage.py，部位分解 + 强度 + 状态机）
  4. 导出 game_scene.json：每物体 collider / rigidbody / damage 参数
  5. 演示：同一能量撞同一物体的不同部位 → 不同破坏结果（位置相关）

用法：
    python scripts/export_game_scene.py --input data/nuscenes_seg_full/val/<x>.npz \
        --output_dir out_game --impact_energy 300
    # 原始点云推理
    python scripts/export_game_scene.py --input cloud.ply \
        --config cfgs/seg_nuscenes_full.yaml --ckpts <ckpt-best.pth> --output_dir out_game
"""
import os, sys, argparse, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.materials import LIDARSEG_32
from utils.instances import cluster_instances
from utils.damage import DamageModel


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
    return xyz, (cols if (len(cols) == len(xyz) and cols.max() > 0) else None)


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
    args.experiment_path = '/tmp/export_game_exp'
    os.makedirs(args.experiment_path, exist_ok=True)
    args.log_name = os.path.basename(config_path).replace('.yaml', '')
    os.environ['LOCAL_RANK'] = '0'
    from utils.logger import get_root_logger
    logger = get_root_logger(name='export_game')
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
    with torch.no_grad():
        logits = model(torch.from_numpy(pts).float().unsqueeze(0).cuda())
    sem = logits.argmax(-1).squeeze(0).cpu().numpy().astype(np.int64)
    return pts[:, :3], sem


# ---------------- 导出 ----------------
def _f(x):
    return float(x) if np.isfinite(x) else None


def build_objects(instance_id, instances, xyz):
    """每个实例 → DamageModel → 导出字典。"""
    objs = []
    for iid, info in instances.items():
        m = instance_id == iid
        pts = xyz[m]
        dm = DamageModel(pts, info['class_id'], info['physical'],
                         info['damage_response'], info['density'])
        bb_min, bb_max = dm.bbox()
        objs.append(dict(
            id=iid,
            class_id=info['class_id'],
            class_name=info['class_name'],
            physical=info['physical'],
            safety=info['safety'],
            lethality=info['lethality'],
            n_points=info['n_points'],
            collider=dict(type='box',
                          center=[_f(c) for c in info['center']],
                          bbox_min=[_f(c) for c in bb_min],
                          bbox_max=[_f(c) for c in bb_max]),
            rigidbody=dict(is_static=bool(dm.is_static), mass=_f(dm.mass),
                           friction=info['friction'],
                           restitution=info['restitution']),
            damage=dm.summary(),
        ))
    return objs


def demo_position_dependent(instance_id, instances, xyz, energy, topk=3):
    """演示：同一能量撞同一物体不同部位 → 不同结果（位置相关破坏）。

    选取「部位强度差异最大」的实例，用介于最弱/最强部位 hp 之间的能量，
    让「撞弱部位→碎、撞强部位→不碎」的差异显式可见。
    """
    print(f'\n{"="*72}\n位置相关破坏演示\n{"="*72}')

    # 选强度差异（max/min ratio）最大的实例
    cands = []
    for iid, info in instances.items():
        if info['damage_response'] == 'none':
            continue  # 刚性不可破坏的跳过
        pts = xyz[instance_id == iid]
        dm = DamageModel(pts, info['class_id'], info['physical'],
                         info['damage_response'], info['density'])
        if len(dm.parts) < 2:
            continue
        strengths = [p['strength'] for p in dm.parts]
        ratio = max(strengths) / max(min(strengths), 1e-6)
        cands.append((ratio, iid, info, pts, dm))
    cands.sort(key=lambda c: -c[0])

    if not cands:
        print('无可演示的破坏实例（本帧只有刚性物体或过小实例）')
        return

    shown = 0
    for ratio, iid, info, pts, dm in cands:
        if shown >= topk:
            break
        parts_sorted = sorted(dm.parts, key=lambda p: -p['strength'])
        strongest = parts_sorted[0]
        weakest = parts_sorted[-1]
        if ratio < 1.15:  # 部位强度差异太小，演示无意义
            continue
        # 能量取最弱部位 hp 的 1.2 倍：弱部位必碎，强部位（若 ratio>1.2）不碎
        e = weakest['hp'] * 1.2
        print(f'\n[{iid}] {info["class_name"]} ({info["physical"]}, '
              f'{info["damage_response"]}, {len(pts)} pts, {len(dm.parts)} parts, '
              f'强度比={ratio:.2f})')
        for label, part in (('最弱', weakest), ('最强', strongest)):
            dm2 = DamageModel(pts, info['class_id'], info['physical'],
                              info['damage_response'], info['density'])
            ev = dm2.apply_impact(part['center'], e)
            print(f'  撞{label}部位 @ {np.round(part["center"],1)} '
                  f'(strength={part["strength"]:.3f}, hp={part["hp"]:.0f}J, '
                  f'能量={e:.0f}J) → {ev["state"]:>9} | {ev["note"]}')
        shown += 1


def write_csv(objs, out_dir):
    """导出 CSV（Unreal DataTable 可直接导入）。"""
    import csv
    obj_rows = [['id', 'class_id', 'class_name', 'physical', 'safety', 'lethality',
                 'is_static', 'mass', 'friction', 'restitution', 'damage_response',
                 'cx', 'cy', 'cz', 'sx', 'sy', 'sz']]
    part_rows = [['object_id', 'part_idx', 'px', 'py', 'pz', 'strength', 'hp', 'damage', 'state']]
    for o in objs:
        bbmin, bbmax = o['collider']['bbox_min'], o['collider']['bbox_max']
        cx, cy, cz = o['collider']['center']
        sx, sy, sz = [bbmax[i] - bbmin[i] for i in range(3)]
        obj_rows.append([o['id'], o['class_id'], o['class_name'], o['physical'], o['safety'],
                         o['lethality'], int(o['rigidbody']['is_static']), o['rigidbody']['mass'],
                         o['rigidbody']['friction'], o['rigidbody']['restitution'],
                         o['damage']['damage_response'], cx, cy, cz, sx, sy, sz])
        for p in o['damage']['parts']:
            part_rows.append([o['id'], p['idx'], p['center'][0], p['center'][1], p['center'][2],
                              p['strength'], p['hp'], p['damage'], p['state']])
    with open(os.path.join(out_dir, 'objects.csv'), 'w', newline='') as f:
        csv.writer(f).writerows(obj_rows)
    with open(os.path.join(out_dir, 'parts.csv'), 'w', newline='') as f:
        csv.writer(f).writerows(part_rows)
    print(f'CSV 已导出 → objects.csv / parts.csv（{len(obj_rows) - 1} 物体，{len(part_rows) - 1} 部位）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--ckpts', default=None)
    ap.add_argument('--npoints', type=int, default=16384)
    ap.add_argument('--min_points', type=int, default=8)
    ap.add_argument('--impact_energy', type=float, default=300.0)
    ap.add_argument('--demo_topk', type=int, default=3)
    ap.add_argument('--emit_csv', action='store_true', help='额外导出 objects.csv / parts.csv（Unreal DataTable 用）')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ext = os.path.splitext(args.input)[1].lower()
    if ext == '.npz':
        xyz, sem = load_npz_sem(args.input)
    elif ext in ('.ply', '.pcd', '.npy') and args.config:
        xyz, rgb = load_point_cloud(args.input)
        xyz, sem = run_seg_inference(xyz, rgb, args.config, args.ckpts, args.npoints)
    else:
        raise ValueError('npz 需含 sem；ply/pcd/npy 需同时给 --config 与 --ckpts')

    print(f'点数 {len(xyz):,}，聚类中...')
    instance_id, instances = cluster_instances(sem, xyz, min_points=args.min_points)
    print(f'得到 {len(instances)} 个实例')

    # 类别统计
    from collections import Counter
    cls_cnt = Counter(info['class_name'] for info in instances.values())
    print('实例类别分布（前10）:')
    for c, n in cls_cnt.most_common(10):
        print(f'  {c:<36} {n:>4}')

    objs = build_objects(instance_id, instances, xyz)

    # 导出 JSON
    out_json = os.path.join(args.output_dir, 'game_scene.json')
    payload = dict(
        num_objects=len(objs),
        impact_energy_used=args.impact_energy,
        objects=objs,
    )
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'\n场景已导出 → {out_json}')

    if args.emit_csv:
        write_csv(objs, args.output_dir)

    # 位置相关破坏演示
    demo_position_dependent(instance_id, instances, xyz, args.impact_energy, args.demo_topk)


if __name__ == '__main__':
    main()
