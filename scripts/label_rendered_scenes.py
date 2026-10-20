#!/usr/bin/env python3
"""双模型标注：对光栅化重渲染场景做 语义(32) + 材质(14) 逐点识别。

用途：全量光照生成完成后，用训练好的两个单任务模型（语义 / 材质）对渲染场景
进行逐点标注，输出并排对比图 + 逐点标签 npz + 类别分布汇总，量化「两个都跑对比」。

流程：
  1. 加载 GLB 城市场景，对内容 mesh 做稠密表面采样（世界坐标）。
  2. 读 manifest，逐个相机构建外参，把表面点变换到相机系
     （x 右 / y 下 / z 前，与 UR-MAT backproject 一致），投影到图像平面。
  3. 在可见点里裁剪出相机附近一个 ~100m 直径的局部 patch（匹配训练分布尺度），
     并把 RGB 从渲染图对应像素采样 -> XYZ+RGB 点云。
  4. 降采样到 npoints，分别喂给语义模型（nuScenes 32 类）与材质模型（UR-MAT 14 类）。
  5. 输出：
       {out}/{scene}/cam{c}.png         并排 [原始RGB | 语义标注 | 材质标注]
       {out}/{scene}/cam{c}.npz         xyz/rgb/sem_pred/mat_pred
       {out}/{scene}/legend_sem.txt     语义 32 类名
       {out}/{scene}/legend_mat.txt     材质 14 类名
       {out}/{scene}/summary.json       逐相机类别分布 + 平均置信度

注意：
  - 材质模型输入用相机系（匹配 UR-MAT 训练）；语义模型输入转 z-up（匹配 nuScenes）。
  - 当前用「表面采样 + 投影取色」，存在遮挡点取色偏差（被挡面会拿到前景颜色），
    作为演示足够；若要无偏，需后续用深度反投影（见 render 脚本的 depth 方案）。
  - 局部裁剪会丢失相机远处的大范围，但把尺度拉回训练分布（避免 Hilbert depth>16 越界）。

用法：
    python scripts/label_rendered_scenes.py \
        --glb /mnt/data/datasets/UR-MAT/extracted/glb/Siemenstaddtlod2ue.glb \
        --rendered /mnt/data/datasets/UR-MAT/lighting_aug/Siemenstaddtlod2ue \
        --sem_cfg experiments/seg_nuscenes_full_16k/cfgs/singletask_sem/20260821-221520/config.yaml \
        --sem_ckpt experiments/seg_nuscenes_full_16k/cfgs/singletask_sem/20260821-221520/ckpt-best.pth \
        --mat_cfg experiments/seg_urmat_16k/cfgs/singletask_mat/20260822-063146/config.yaml \
        --mat_ckpt experiments/seg_urmat_16k/cfgs/singletask_mat/20260822-063146/ckpt-best.pth \
        --out /mnt/data/datasets/UR-MAT/labeled_scenes
"""

import argparse
import json
import os
import sys
import time

# 允许从 scripts/ 直接运行也能 import 到仓库根目录的 models / utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from easydict import EasyDict
from PIL import Image

W, H = 640, 480
FX, FY, CX, CY = 320.0, 320.0, 320.0, 240.0
BIG_EXTENT = 5000.0          # 超过该 extent 的 mesh 视为天空球/离群，剔除
NEAR = 0.5                   # 相机近裁剪
LOCAL_R = 50.0               # 局部 patch 半径(m)，直径 ~100m 匹配训练分布
NPOINTS = 16384

# nuScenes-lidarseg 32 类（与 utils/materials.py LIDARSEG_32 一致）
LIDARSEG_32 = [
    'noise', 'animal',
    'human.pedestrian.adult', 'human.pedestrian.child',
    'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer', 'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.barrier', 'movable_object.debris',
    'movable_object.pushable_pullable', 'movable_object.trafficcone',
    'static_object.bicycle_rack', 'vehicle.bicycle',
    'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car',
    'vehicle.construction', 'vehicle.emergency.ambulance',
    'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer',
    'vehicle.truck', 'flat.driveable_surface', 'flat.other',
    'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other',
    'static.vegetation', 'vehicle.ego',
]

# UR-MAT 14 类材质（与 scripts/build_urmat_dataset.py MATERIAL_NAMES 一致）
MATERIAL_NAMES = [
    'Limestone', 'Slate', 'Glass', 'Steel', 'Brick', 'Cobblestone',
    'Plaster', 'Concrete', 'Gravel', 'Tiles', 'Wood', 'Asphalt',
    'Plastic', 'Unknown',
]


def hsv_to_rgb(h, s, v):
    h = h % 1.0
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        return np.array([v, t, p])
    if i == 1:
        return np.array([q, v, p])
    if i == 2:
        return np.array([p, v, t])
    if i == 3:
        return np.array([p, q, v])
    if i == 4:
        return np.array([t, p, v])
    return np.array([v, p, q])


def colormap(n):
    """n 个彼此区分度好的 RGB(0-1) 颜色。"""
    return np.stack([hsv_to_rgb(i * 0.61803398875, 0.72, 0.92) for i in range(n)])


def compute_content_bounds(model):
    mins, maxs = [], []
    for m in model.meshes:
        name = (m.mesh_name or "").lower()
        if "sky" in name or "sphere" in name:
            continue
        v = np.asarray(m.mesh.vertices)
        if v.size == 0:
            continue
        lo, hi = v.min(0), v.max(0)
        if (hi - lo).max() > BIG_EXTENT:
            continue
        mins.append(lo)
        maxs.append(hi)
    if not mins:
        raise RuntimeError("没有有效内容 mesh")
    return np.array(mins).min(0), np.array(maxs).max(0)


def sample_surface(model, total=1_500_000):
    """对内容 mesh 做稠密表面采样，返回 (M,3) float32 世界坐标。"""
    meshes = []
    vcounts = []
    for m in model.meshes:
        name = (m.mesh_name or "").lower()
        if "sky" in name or "sphere" in name:
            continue
        v = np.asarray(m.mesh.vertices)
        if v.size == 0 or (v.max(0) - v.min(0)).max() > BIG_EXTENT:
            continue
        meshes.append(m.mesh)
        vcounts.append(len(v))
    if not meshes:
        raise RuntimeError("没有可采样 mesh")

    total_v = sum(vcounts)
    pts = []
    for mesh, vc in zip(meshes, vcounts):
        n = max(300, int(total * vc / total_v))
        try:
            pcd = mesh.sample_points_uniformly(number_of_points=n)
            p = np.asarray(pcd.points).astype(np.float32)
            if p.size:
                pts.append(p)
        except Exception:
            continue
    if not pts:
        raise RuntimeError("表面采样失败")
    return np.concatenate(pts, axis=0)


def cam_basis(eye, center):
    """由 eye/center/up=[0,1,0] 构造相机基（世界系），与 Open3D look_at 一致。"""
    eye = np.asarray(eye, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fwd = center - eye
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    upc = np.cross(right, fwd)
    return right, upc, fwd


def world2cam(p, eye, right, upc, fwd):
    """世界 -> 相机系 (x 右 / y 下 / z 前)，返回 (M,3)。"""
    d = p - eye
    x = d @ right
    y = -(d @ upc)          # 下
    z = d @ fwd             # 前
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def build_model(cfg_path, ckpt_path):
    """从 yaml 提取 model 段 -> PointMambaSeg -> 载入 ckpt -> eval。"""
    from models.point_mamba_seg import PointMambaSeg
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = PointMambaSeg(EasyDict(cfg['model']))
    sd = torch.load(ckpt_path, map_location='cpu')
    base = sd.get('base_model') or sd.get('model')
    base = {k.replace('module.', ''): v for k, v in base.items()}
    model.load_state_dict(base, strict=False)
    model.cuda().eval()
    return model


@torch.no_grad()
def infer(model, pts):
    """pts: (1, N, 6) -> (N,) 标签, (N,) top-1 置信度。"""
    logits, _ = model(pts)
    prob = torch.softmax(logits, dim=-1)  # (1,N,C)
    conf, pred = prob.max(-1)
    return pred[0].cpu().numpy().astype(np.int64), conf[0].cpu().numpy().astype(np.float32)


def draw_labels(u, v, pred, cmap):
    """把 (u,v,pred) 画成 (H,W,3) uint8 标注图（3x3 块，更醒目）。"""
    colors = (cmap * 255.0).astype(np.uint8)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    c = colors[pred]
    for dr in (-1, 0, 1):
        vv = np.clip(v + dr, 0, H - 1)
        for dc in (-1, 0, 1):
            uu = np.clip(u + dc, 0, W - 1)
            img[vv, uu] = c
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glb', required=True)
    ap.add_argument('--rendered', required=True, help='含 manifest.json + png 的目录')
    ap.add_argument('--sem_cfg', required=True)
    ap.add_argument('--sem_ckpt', required=True)
    ap.add_argument('--mat_cfg', required=True)
    ap.add_argument('--mat_ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--npoints', type=int, default=NPOINTS)
    ap.add_argument('--local_r', type=float, default=LOCAL_R)
    ap.add_argument('--lighting_indices', default='0', help='每相机取哪些光照(逗号分隔, 按 manifest 内序号)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scene_name = os.path.splitext(os.path.basename(args.glb))[0]
    out_dir = os.path.join(args.out, scene_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.rendered, 'manifest.json')) as f:
        manifest = json.load(f)

    # 图片按相机分组，并映射到 manifest 中的光照序号
    cam_images = {}
    for li, im in enumerate(manifest['images']):
        cam_images.setdefault(im['camera'], []).append((li, im))

    light_idx = [int(x) for x in args.lighting_indices.split(',') if x != '']

    print(f"[load model]", flush=True)
    sem_model = build_model(args.sem_cfg, args.sem_ckpt)
    mat_model = build_model(args.mat_cfg, args.mat_ckpt)
    sem_cmap = colormap(len(LIDARSEG_32))
    mat_cmap = colormap(len(MATERIAL_NAMES))

    print(f"[load glb] {args.glb}", flush=True)
    import open3d as o3d
    t0 = time.time()
    model = o3d.io.read_triangle_model(args.glb)
    print(f"       {len(model.meshes)} meshes, {time.time()-t0:.1f}s", flush=True)

    cmin, cmax = compute_content_bounds(model)
    center = (cmin + cmax) / 2.0
    print(f"[bounds] center={center}", flush=True)

    t0 = time.time()
    surf = sample_surface(model)
    print(f"[surface] {surf.shape[0]} points, {time.time()-t0:.1f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    summary = {'scene': scene_name, 'cameras': []}

    for cam in manifest['cameras']:
        cam_id = cam['id']
        eye = np.asarray(cam['eye'], dtype=np.float32)
        right, upc, fwd = cam_basis(eye, center)

        # 表面点 -> 相机系 -> 投影
        pc_cam = world2cam(surf, eye, right, upc, fwd)   # (M,3) x右 y下 z前
        z = pc_cam[:, 2]
        u = pc_cam[:, 0] / z * FX + CX
        v = pc_cam[:, 1] / z * FY + CY
        vis = (z > NEAR) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if vis.sum() < 2048:
            print(f"[skip] cam{cam_id}: 可见点不足 {vis.sum()}", flush=True)
            continue

        # 取本相机的一个光照图像
        if cam_id not in cam_images:
            print(f"[skip] cam{cam_id}: manifest 无对应图像", flush=True)
            continue
        li, im = cam_images[cam_id][light_idx[0] % len(cam_images[cam_id])]
        img = np.asarray(Image.open(os.path.join(args.rendered, im['file'])).convert('RGB'),
                         dtype=np.float32) / 255.0

        # 局部 patch：以可见点质心为中心、半径 local_r 的球
        vis_pts = pc_cam[vis]
        centroid = vis_pts.mean(0)
        d = np.linalg.norm(pc_cam - centroid, axis=1)
        keep = vis & (d < args.local_r)
        if keep.sum() < 2048:
            print(f"[skip] cam{cam_id}: 局部 patch 点不足 {keep.sum()}", flush=True)
            continue

        # 局部化（中心移到原点），保持相机系轴向
        xyz_local = pc_cam[keep] - centroid
        uu = u[keep].astype(np.int32)
        vv = v[keep].astype(np.int32)
        uu = np.clip(uu, 0, W - 1)
        vv = np.clip(vv, 0, H - 1)
        rgb = img[vv, uu].astype(np.float32)   # (K,3)

        # 降采样到 npoints
        K = xyz_local.shape[0]
        if K > args.npoints:
            sel = rng.choice(K, args.npoints, replace=False)
        else:
            sel = np.arange(K)
        xyz = xyz_local[sel]
        rgb = rgb[sel]
        uu_s, vv_s = uu[sel], vv[sel]

        # 语义模型输入：z-up (x右 / z前 / -y下 -> x右 / y前 / z上)
        xyz_lidar = np.stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]], axis=-1).astype(np.float32)
        pts_sem = torch.from_numpy(np.concatenate([xyz_lidar, rgb], axis=1)).float().unsqueeze(0).cuda()
        # 材质模型输入：相机系 (x右 / y下 / z前)
        pts_mat = torch.from_numpy(np.concatenate([xyz, rgb], axis=1)).float().unsqueeze(0).cuda()

        sem_pred, sem_conf = infer(sem_model, pts_sem)
        mat_pred, mat_conf = infer(mat_model, pts_mat)

        # 可视化
        sem_img = draw_labels(uu_s, vv_s, sem_pred, sem_cmap)
        mat_img = draw_labels(uu_s, vv_s, mat_pred, mat_cmap)
        rgb_img = (img * 255.0).astype(np.uint8)
        combined = np.concatenate([rgb_img, sem_img, mat_img], axis=1)
        Image.fromarray(combined).save(os.path.join(out_dir, f'cam{cam_id}.png'))

        np.savez_compressed(
            os.path.join(out_dir, f'cam{cam_id}.npz'),
            xyz_cam=xyz.astype(np.float32), xyz_lidar=xyz_lidar,
            rgb=rgb, sem_pred=sem_pred, mat_pred=mat_pred,
        )

        entry = {
            'camera': cam_id,
            'lighting': im['lighting'],
            'n_points': int(xyz.shape[0]),
            'sem_top1_conf': float(sem_conf.mean()),
            'mat_top1_conf': float(mat_conf.mean()),
            'sem_hist': np.bincount(sem_pred, minlength=len(LIDARSEG_32)).tolist(),
            'mat_hist': np.bincount(mat_pred, minlength=len(MATERIAL_NAMES)).tolist(),
        }
        summary['cameras'].append(entry)
        print(f"[done] cam{cam_id}: {entry['n_points']} pts | "
              f"sem_conf={entry['sem_top1_conf']:.3f} mat_conf={entry['mat_top1_conf']:.3f}",
              flush=True)

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, 'legend_sem.txt'), 'w') as f:
        f.write('\n'.join(f'{i:2d} {n}' for i, n in enumerate(LIDARSEG_32)))
    with open(os.path.join(out_dir, 'legend_mat.txt'), 'w') as f:
        f.write('\n'.join(f'{i:2d} {n}' for i, n in enumerate(MATERIAL_NAMES)))

    print(f"[done] {scene_name} -> {out_dir}", flush=True)


if __name__ == '__main__':
    main()
