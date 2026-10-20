#!/usr/bin/env python3
"""街道级相机 + 深度反投影：生成与 UR-MAT 训练分布一致的 XYZ+RGB 点云。

背景：
  前一阶段 render_lighting_aug.py 用「斜俯视环绕」相机（~600m 高），渲染图像虽好看，
  但反投影点云的尺度/视角与 UR-MAT 训练分布（街道级、z<100m、相机系 x右/y下/z前）
  不一致，直接喂模型会越界 Hilbert 序列化(depth>16) 且视角错位。

本脚本用 RaycastingScene 做「射线反投影」，等效于 UR-MAT 的 depth 反投影但更精确：
  1. 地面估计用表面 y 的 5% 分位数（而非 min，避免离群点把相机压到地下）。
  2. 在地面+2m 网格上采样候选相机位，向上射线 ≥5m 无命中 = 开敞（非建筑内/屋檐下）；
     再做粗射线计数，只保留「附近有足够几何」的位置（避免浮在场景边缘空处）。
  3. 每个相机 cast 全像素射线（RaycastingScene）拿到精确命中距离 -> 相机系 XYZ，
     并用 Filament 渲染的多光照 RGB 上色（XYZ 与光照无关，仅上色变化）。
  4. 裁剪 z<100m，输出 {xyz (N,3) [m], rgb (N,3) [0,1]} 的 npz（与 seg_16k 训练格式一致）。

用法：
    python scripts/render_street_pc.py \
        --glb /mnt/data/datasets/UR-MAT/extracted/glb/Siemenstaddtlod2ue.glb \
        --out /mnt/data/datasets/UR-MAT/street_pc \
        --num_cam 8 --num_lighting 12 --seed 0
"""

import argparse
import json
import os
import time

import numpy as np

W, H = 640, 480
FX, FY, CX, CY = 320.0, 320.0, 320.0, 240.0
NEAR = 0.1
DEPTH_MAX = 100.0    # 与 UR-MAT DEPTH_SENTINEL 一致
BIG_EXTENT = 5000.0
CAM_HEIGHT = 2.0
CLEARANCE = 5.0      # 相机上方 ≥5m 无遮挡 = 开敞


def color_temperature_rgb(kelvin):
    if kelvin <= 3200:
        return np.array([1.00, 0.72, 0.45], dtype=np.float32)
    if kelvin <= 4500:
        return np.array([1.00, 0.90, 0.76], dtype=np.float32)
    if kelvin <= 5800:
        return np.array([1.00, 0.98, 0.92], dtype=np.float32)
    if kelvin <= 7000:
        return np.array([0.95, 0.95, 1.00], dtype=np.float32)
    return np.array([0.85, 0.88, 1.00], dtype=np.float32)


def v3(x):
    return np.asarray(x, dtype=np.float32).reshape(3, 1)


def content_meshes(model):
    """返回内容 mesh 列表（剔除天空球/离群），供 bounds/地面/raycast 共用。"""
    out = []
    for m in model.meshes:
        name = (m.mesh_name or "").lower()
        if "sky" in name or "sphere" in name:
            continue
        v = np.asarray(m.mesh.vertices)
        if v.size == 0 or (v.max(0) - v.min(0)).max() > BIG_EXTENT:
            continue
        out.append(m)
    if not out:
        raise RuntimeError("没有有效内容 mesh")
    return out


def estimate_ground(meshes, p=5.0):
    """地面高度：所有内容 mesh 顶点 y 的 p% 分位数（鲁棒于离群低点）。"""
    ys = [np.asarray(m.mesh.vertices)[:, 1] for m in meshes]
    return float(np.percentile(np.concatenate(ys), p))


def build_raycasting_scene(meshes):
    import open3d as o3d
    rc = o3d.t.geometry.RaycastingScene()
    for m in meshes:
        rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m.mesh))
    return rc


def cam_basis(eye, center):
    eye = np.asarray(eye, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fwd = center - eye
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    upc = np.cross(right, fwd)
    return right, upc, fwd


def pixel_ray_dirs(right, upc, fwd):
    """全像素射线方向 (H,W,3) 单位向量。"""
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = (u - CX) / FX          # 右
    dy = (CY - v) / FY          # 上
    dirs = dx[..., None] * right + dy[..., None] * upc + fwd
    dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8)
    return dirs


def cast_rays(rc, eye, right, upc, fwd):
    """全像素射线求交，返回 (t_hit, xyz_cam, valid)。
    xyz_cam: (H,W,3) 相机系 (x右 y下 z前)，与 UR-MAT backproject 一致。
    valid: (H,W) 命中且 0.1<z<100。
    """
    import open3d as o3d
    dirs = pixel_ray_dirs(right, upc, fwd)                 # (H,W,3)
    dirs_flat = dirs.reshape(-1, 3).astype(np.float32)
    origins = np.tile(eye.astype(np.float32), (W * H, 1))
    rays = np.concatenate([origins, dirs_flat], axis=1)
    res = rc.cast_rays(o3d.core.Tensor(rays))
    t = res['t_hit'].numpy().astype(np.float32).reshape(H, W)

    # 相机系坐标：z = t * dot(dir, fwd)（垂直深度），x/y 按内参投影
    dot_fwd = (dirs * fwd).sum(-1)
    dot_right = (dirs * right).sum(-1)
    dot_upc = (dirs * upc).sum(-1)
    with np.errstate(invalid='ignore'):     # t=inf(天空) 时乘出 nan，会被 valid 掩掉
        z = t * dot_fwd
        x = t * dot_right
        y = -t * dot_upc
    valid = np.isfinite(t) & (z > NEAR) & (z < DEPTH_MAX)
    return t, np.stack([x, y, z], axis=-1), valid


def sample_lighting(rng):
    return {
        "azimuth": rng.uniform(0.0, 360.0),
        "elevation": rng.uniform(15.0, 60.0),
        "kelvin": float(rng.choice([2700, 4000, 5500, 6500, 8000])),
        "sun_intensity": float(rng.uniform(3000.0, 120000.0)),
        "fill_intensity": float(rng.uniform(0.0, 60000.0)),
        "ambient": float(rng.uniform(6000.0, 16000.0)),   # 天空 ambient 强度(阴晴变化, IBL 量纲~1e4)
    }


def sun_direction(azimuth_deg, elevation_deg):
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    sun_pos = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
    return -sun_pos


def apply_lighting(scene, light_cfg):
    d = sun_direction(light_cfg["azimuth"], light_cfg["elevation"])
    color = color_temperature_rgb(light_cfg["kelvin"])
    scene.scene.set_sun_light(v3(d), v3(color), light_cfg["sun_intensity"])
    scene.scene.enable_sun_light(True)
    scene.scene.update_light_intensity("fill", light_cfg["fill_intensity"])
    # 天空 ambient（IBL）：强度随光照配置变化，阴天强/晴天弱，避免背光立面全黑
    scene.scene.set_indirect_light_intensity(light_cfg["ambient"])


def azimuth_view_score(rc, pos, az_deg):
    """评估某水平朝向的「街道感」：命中射线中位深度越大 = 越像望向街道纵深，
    覆盖率过低（看向场景外空旷）则判为无效。返回 (score, center)。"""
    import open3d as o3d
    a = np.deg2rad(az_deg)
    d = np.array([np.cos(a), 0.0, np.sin(a)], dtype=np.float32)
    eye = np.asarray(pos, dtype=np.float32)
    center = eye + d * 30.0
    center[1] = eye[1]          # 保持水平朝向（与 UR-MAT 街道相机一致）
    right, upc, fwd = cam_basis(eye, center)
    dirs = pixel_ray_dirs(right, upc, fwd)[::8, ::10].reshape(-1, 3).astype(np.float32)
    origins = np.tile(eye, (dirs.shape[0], 1))
    rays = np.concatenate([origins, dirs], axis=1)
    t = rc.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    valid = np.isfinite(t) & (t > NEAR) & (t < DEPTH_MAX)
    frac = float(valid.mean())
    if frac < 0.30:             # 大面积天空/出界 -> 无效视角
        return -np.inf, center
    return float(np.median(t[valid])), center


def sample_cameras(rc, cmin, cmax, ground, num_cam, rng):
    """采样开敞街道级相机位，并为每个位置挑选「望向街道纵深」的水平朝向，
    返回 [(eye, center)]。"""
    import open3d as o3d
    span_x = cmax[0] - cmin[0]
    span_z = cmax[2] - cmin[2]
    step = max(10.0, min(span_x, span_z) / 15.0)
    xs = np.arange(cmin[0] + 5, cmax[0] - 5, step)
    zs = np.arange(cmin[2] + 5, cmax[2] - 5, step)

    # 1) 开敞检查（向上射线 5m 无命中 = 非建筑内/屋檐下）
    open_pos = []
    for x in xs:
        for z in zs:
            pos = np.array([x, ground + CAM_HEIGHT, z], dtype=np.float32)
            rays = np.array([[pos[0], pos[1], pos[2], 0.0, 1.0, 0.0]], dtype=np.float32)
            t = rc.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()[0]
            if t > CLEARANCE:
                open_pos.append(pos)

    # 2) 每个位置在 8 个水平方位里挑中位深度最大者（望向街道而非近墙）
    scored = []
    for pos in open_pos:
        best_score, best_center = -np.inf, None
        for az in range(0, 360, 45):
            s, c = azimuth_view_score(rc, pos, float(az))
            if s > best_score:
                best_score, best_center = s, c
        if best_center is not None and best_score > 8.0:   # 中位深度 > 8m 才算街道视角
            scored.append((best_score, pos, best_center))

    scored.sort(key=lambda s: -s[0])
    eyes = [p for _, p, _ in scored[:num_cam]]
    centers = [c for _, _, c in scored[:num_cam]]
    return eyes, centers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glb', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--num_cam', type=int, default=8)
    ap.add_argument('--num_lighting', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bg', nargs=3, type=float, default=[0.53, 0.81, 0.92])
    ap.add_argument('--min_points', type=int, default=4096)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    import open3d as o3d
    from open3d.visualization import rendering

    scene_name = os.path.splitext(os.path.basename(args.glb))[0]
    out_dir = os.path.join(args.out, scene_name)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print(f"[load] {args.glb}", flush=True)
    model = o3d.io.read_triangle_model(args.glb)
    meshes = content_meshes(model)
    print(f"       {len(meshes)} meshes, {time.time()-t0:.1f}s", flush=True)

    cmin = np.min([np.asarray(m.mesh.vertices).min(0) for m in meshes], axis=0)
    cmax = np.max([np.asarray(m.mesh.vertices).max(0) for m in meshes], axis=0)
    ground = estimate_ground(meshes)
    print(f"[bounds] cmin={cmin.round(1)} cmax={cmax.round(1)} ground={ground:.2f}", flush=True)

    t0 = time.time()
    rc = build_raycasting_scene(meshes)
    print(f"[raycast] scene built {time.time()-t0:.1f}s", flush=True)

    # Filament 渲染器（仅用于 RGB 上色）
    renderer = rendering.OffscreenRenderer(W, H)
    scene = renderer.scene
    for i, m in enumerate(meshes):
        scene.add_geometry(f"m{i}", m.mesh, model.materials[m.material_idx])
    scene.set_background(np.array([*args.bg, 1.0], dtype=np.float32).reshape(4, 1))
    scene.scene.enable_indirect_light(True)   # 保留默认天空 IBL 作 ambient，避免背光立面全黑
    scene.scene.add_directional_light("fill", v3([1.0, 1.0, 1.0]),
                                      v3([0.0, -1.0, 0.0]), 0.0, False)
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

    print(f"[camera] 采样开敞街道相机位...", flush=True)
    eyes, centers = sample_cameras(rc, cmin, cmax, ground, args.num_cam, rng)
    print(f"        得到 {len(eyes)} 个相机位", flush=True)

    manifest = {"scene": scene_name, "glb": args.glb,
                "cameras": [{"id": i, "eye": e.tolist(), "center": c.tolist()}
                            for i, (e, c) in enumerate(zip(eyes, centers))],
                "samples": []}
    n = 0
    for ci, (eye, tgt) in enumerate(zip(eyes, centers)):
        right, upc, fwd = cam_basis(eye, tgt)
        # 每相机射线求交一次（与光照无关）
        _, xyz_cam, valid = cast_rays(rc, eye, right, upc, fwd)
        n_valid = int(valid.sum())
        if n_valid < args.min_points:
            print(f"[skip] cam{ci}: 有效点不足 {n_valid}", flush=True)
            continue
        # 命中点的像素索引
        vv, uu = np.mgrid[0:H, 0:W]
        u_valid = uu[valid].astype(np.int32)
        v_valid = vv[valid].astype(np.int32)
        xyz_v = xyz_cam[valid].astype(np.float32)

        for li in range(args.num_lighting):
            light_cfg = sample_lighting(rng)
            try:
                apply_lighting(scene, light_cfg)
                scene.camera.look_at(v3(tgt), v3(eye), v3([0.0, 1.0, 0.0]))
                scene.camera.set_projection(K, NEAR, 120.0, W, H)
                rgb = np.asarray(renderer.render_to_image()).astype(np.float32) / 255.0
            except Exception as e:
                print(f"[skip] cam{ci} light{li}: {e}", flush=True)
                continue
            rgb_v = rgb[v_valid, u_valid].astype(np.float32)

            fname = f"{scene_name}_cam{ci}_light{li}.npz"
            np.savez_compressed(os.path.join(out_dir, fname), xyz=xyz_v, rgb=rgb_v)
            manifest["samples"].append({
                "file": fname, "camera": ci, "eye": eye.tolist(),
                "lighting": light_cfg, "n_points": n_valid,
            })
            n += 1

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] {n} 个点云 -> {out_dir}  total {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
