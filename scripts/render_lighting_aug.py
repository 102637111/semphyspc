#!/usr/bin/env python3
"""光栅化光照模拟：对 UR-MAT GLB 城市场景在多变光照下重渲染 RGB 图像。

原理：
  用 Open3D 的 Filament 光栅化渲染器（PBR），加载 UR-MAT 的 GLB 城市场景，
  在（近似的）相同相机位姿下采样多种光照环境，生成大量光照多样的 RGB 图像。
  这些图像用于颜色注入（XYZ+RGB 早期融合）的光照鲁棒性增强 —— 让模型见过
  不同太阳方向/色温/强度、不同环境光下的同一几何，从而对光照变化更鲁棒。

光照参数（可独立采样）：
  - 太阳光方向：方位角 azimuth [0,360°)、仰角 elevation [10,70°]
  - 太阳光色温：2700K(暖橙) ~ 8000K(冷蓝)
  - 太阳光强度：20000 ~ 120000 lux
  - 阴天漫射补光强度：0 ~ 60000（顶部向下定向灯，模拟多云 vs 晴天）

相机：
  - 内参复用 UR-MAT/UnrealCV 默认：FX=FY=320, CX=320, CY=240, 640x480, FOV≈90°
  - 位姿：围绕内容包围盒中心做斜俯视环绕（俯角约 30°），保证城市稳定入画

用法：
    python scripts/render_lighting_aug.py \
        --glb /mnt/data/datasets/UR-MAT/extracted/glb/Siemenstaddtlod2ue.glb \
        --out /mnt/data/datasets/UR-MAT/lighting_aug \
        --num_cam 4 --num_lighting 12 --seed 0

输出：
    {out}/{scene_name}/{scene}_cam{c}_light{l}.png + manifest.json
"""

import argparse
import json
import os
import time
import numpy as np

W, H = 640, 480
FX, FY, CX, CY = 320.0, 320.0, 320.0, 240.0
NEAR, FAR_SCALE = 0.1, 20.0  # 近平面 0.1m；远平面 = extent * FAR_SCALE
BIG_EXTENT = 5000.0          # 超过该 extent 的 mesh 视为天空球/离群，剔除


def color_temperature_rgb(kelvin):
    """色温(K) -> 白点 RGB（近似，归一化到 [0,1] 附近）。"""
    if kelvin <= 3200:
        return np.array([1.00, 0.72, 0.45], dtype=np.float32)  # 2700-3200K 暖橙
    if kelvin <= 4500:
        return np.array([1.00, 0.90, 0.76], dtype=np.float32)  # 4000K 暖白
    if kelvin <= 5800:
        return np.array([1.00, 0.98, 0.92], dtype=np.float32)  # 5500K 日光
    if kelvin <= 7000:
        return np.array([0.95, 0.95, 1.00], dtype=np.float32)  # 6500K 冷白
    return np.array([0.85, 0.88, 1.00], dtype=np.float32)      # 8000K 冷蓝


def v3(x):
    return np.asarray(x, dtype=np.float32).reshape(3, 1)


def compute_content_bounds(model):
    """计算内容包围盒，排除天空球/超大离群 mesh。"""
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
        raise RuntimeError("没有有效内容 mesh（可能全部被过滤）")
    return np.array(mins).min(0), np.array(maxs).max(0)


def add_model_no_sky(scene, model):
    """逐 mesh 加入场景，剔除天空球/离群 mesh，返回加入数量。"""
    added = 0
    for i, m in enumerate(model.meshes):
        name = (m.mesh_name or "").lower()
        if "sky" in name or "sphere" in name:
            continue
        v = np.asarray(m.mesh.vertices)
        if v.size == 0 or (v.max(0) - v.min(0)).max() > BIG_EXTENT:
            continue
        scene.add_geometry(f"m{i}", m.mesh, model.materials[m.material_idx])
        added += 1
    return added


def sample_camera_eyes(cmin, cmax, num_cam, rng):
    """斜俯视环绕采样：围绕内容中心，俯角约 30°，保证城市稳定入画。"""
    center = (cmin + cmax) / 2.0
    span = cmax - cmin
    diag = float(np.linalg.norm(span))
    radius = diag * 0.55
    height = float(cmin[1] + span[1] + diag * 0.30)
    eyes = []
    for i in range(num_cam):
        az = 2.0 * np.pi * i / num_cam + rng.uniform(0.0, 0.3)
        ex = center[0] + radius * np.cos(az)
        ez = center[2] + radius * np.sin(az)
        eyes.append(np.array([ex, height, ez], dtype=np.float32))
    return eyes, center


def sample_lighting(rng):
    """采样一组光照参数。"""
    return {
        "azimuth": rng.uniform(0.0, 360.0),
        "elevation": rng.uniform(5.0, 65.0),     # 低仰角 -> 更长阴影
        "kelvin": float(rng.choice([2700, 4000, 5500, 6500, 8000])),
        "sun_intensity": float(rng.uniform(3000.0, 120000.0)),  # 黄昏弱光 -> 正午强光
        "fill_intensity": float(rng.uniform(0.0, 60000.0)),   # 阴天漫射补光(定向灯)
    }


def sun_direction(azimuth_deg, elevation_deg):
    """由方位角/仰角计算太阳光方向（光传播方向，指向下方）。"""
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    sun_pos = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
    return -sun_pos


def render_one(renderer, scene, eye, center, light_cfg, far):
    """应用光照 + 相机，渲染一帧。"""
    d = sun_direction(light_cfg["azimuth"], light_cfg["elevation"])
    color = color_temperature_rgb(light_cfg["kelvin"])
    scene.scene.set_sun_light(v3(d), v3(color), light_cfg["sun_intensity"])
    scene.scene.enable_sun_light(True)
    # 阴天漫射补光：顶部向下的定向灯，强度随样本变化（模拟多云 vs 晴天）
    scene.scene.update_light_intensity("fill", light_cfg["fill_intensity"])

    scene.camera.look_at(v3(center), v3(eye), v3([0.0, 1.0, 0.0]))
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
    scene.camera.set_projection(K, NEAR, far, W, H)

    return np.asarray(renderer.render_to_image())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_cam", type=int, default=4)
    ap.add_argument("--num_lighting", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bg", nargs=3, type=float, default=[0.53, 0.81, 0.92],
                    help="背景天空色 RGB(0-1)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    import open3d as o3d
    import open3d.visualization.rendering as rendering

    scene_name = os.path.splitext(os.path.basename(args.glb))[0]
    out_dir = os.path.join(args.out, scene_name)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print(f"[load] {args.glb}", flush=True)
    model = o3d.io.read_triangle_model(args.glb)
    print(f"       {len(model.meshes)} meshes, {time.time()-t0:.1f}s", flush=True)

    cmin, cmax = compute_content_bounds(model)
    print(f"[bounds] min={cmin} max={cmax} extent={(cmax-cmin)}", flush=True)

    renderer = rendering.OffscreenRenderer(W, H)
    scene = renderer.scene
    added = add_model_no_sky(scene, model)
    print(f"[scene] added {added} meshes (sky/outlier removed)", flush=True)
    scene.set_background(np.array([*args.bg, 1.0], dtype=np.float32).reshape(4, 1))

    # 关闭默认间接光(带绿色底)，让太阳 + 补光成为主光源，光照变化更显著
    scene.scene.enable_indirect_light(False)
    # 顶部向下的阴天漫射补光灯（中性白，初始 0）
    scene.scene.add_directional_light("fill", v3([1.0, 1.0, 1.0]),
                                      v3([0.0, -1.0, 0.0]), 0.0, False)

    eyes, center = sample_camera_eyes(cmin, cmax, args.num_cam, rng)
    far = max(1000.0, float(np.linalg.norm(cmax - cmin)) * FAR_SCALE)

    manifest = {"scene": scene_name, "glb": args.glb,
                "cameras": [{"id": i, "eye": e.tolist()} for i, e in enumerate(eyes)],
                "images": []}
    n = 0
    for ci, eye in enumerate(eyes):
        for li in range(args.num_lighting):
            light_cfg = sample_lighting(rng)
            try:
                arr = render_one(renderer, scene, eye, center, light_cfg, far)
            except Exception as e:
                print(f"[skip] cam{ci} light{li}: {e}", flush=True)
                continue
            fname = f"{scene_name}_cam{ci}_light{li}.png"
            o3d.io.write_image(os.path.join(out_dir, fname), o3d.geometry.Image(arr))
            manifest["images"].append({
                "file": fname, "camera": ci, "eye": eye.tolist(),
                "lighting": light_cfg, "mean": float(arr.mean()),
            })
            n += 1

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[done] {n} images -> {out_dir}  total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
