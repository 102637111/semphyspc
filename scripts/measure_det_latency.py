#!/usr/bin/env python3
"""实测 6 相机检测的单帧延迟，并折算成「这段时间车能跑多远」。

为什么必须实测而不是估：安全决策层的时间预算是唯一可能让它变成净负收益的因素。
50 km/h = 13.9 m/s，30 m 预警只有 2.2 s；多一级检测+门控+决策都在吃这个预算。
不测出来就没法说这一层是否安全。

分三段计时：预处理（processor）/ 前向 / 后处理，各报均值和 p95。用 CUDA Event 计时，
每次迭代前 synchronize，避免把异步 kernel 的排队时间算进去。

用法：
    python3 scripts/measure_det_latency.py --model experiments/s3_rtdetr_finetune/hf_best \
        --out results/det_latency_s3.json
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_2d_detector as r2d          # 复用它的相机顺序，避免自己再推一遍搞错

from nuscenes.nuscenes import NuScenes
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


def timed(fn, iters, device):
    """跑 iters 次，返回每次的毫秒数列表。"""
    ts = []
    for _ in range(iters):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='PekingU/rtdetr_r50vd')
    ap.add_argument('--dataroot', default='/mnt/data/nuscenes')
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--proj', default='data/nuscenes_projection_index.npz')
    ap.add_argument('--n_frames', type=int, default=20, help='取几帧真实数据')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--score_thr', type=float, default=0.35)
    ap.add_argument('--out', default='results/det_latency.json')
    args = ap.parse_args()

    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[lat] 设备 {device}，模型 {args.model}', flush=True)

    proc = RTDetrImageProcessor.from_pretrained(args.model)
    model = RTDetrForObjectDetection.from_pretrained(args.model).eval().to(device)

    proj = np.load(args.proj, allow_pickle=True)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    tokens = [str(t) for t in proj['tokens']][:args.n_frames]
    tok2imgs = r2d.build_token_to_images(nusc, tokens)

    # 一帧 = 6 张相机图，这是最小的时间单元
    frames = []
    for t in tokens:
        paths = tok2imgs.get(t)
        if not paths or any(p is None or not os.path.exists(p) for p in paths):
            continue
        imgs = [Image.open(p).convert('RGB') for p in paths]
        if len(imgs) != 6:
            continue
        frames.append(imgs)
    if not frames:
        raise SystemExit('没取到可用帧')
    print(f'[lat] 取到 {len(frames)} 帧 × 6 相机，图尺寸 {frames[0][0].size}', flush=True)

    print(f'[lat] 预热 {args.warmup} 次 ...', flush=True)
    for _ in range(args.warmup):
        o = proc(images=frames[0], return_tensors='pt')
        o = {k: v.to(device) for k, v in o.items()}
        with torch.no_grad():
            model(**o)

    def one(fr):
        o = proc(images=fr, return_tensors='pt')
        o = {k: v.to(device) for k, v in o.items()}
        with torch.no_grad():
            outs = model(**o)
            proc.post_process_object_detection(
                outs, target_sizes=[(r2d.IMG_H, r2d.IMG_W)] * len(fr),
                threshold=args.score_thr)
        return outs

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # 三段分别计时：先只做预处理，再只做前向，最后整体
    fr = frames[0]

    def pre_only():
        return proc(images=fr, return_tensors='pt')

    pre = timed(pre_only, args.iters, device)

    o = proc(images=fr, return_tensors='pt')
    o = {k: v.to(device) for k, v in o.items()}

    def fwd_only():
        with torch.no_grad():
            return model(**o)

    fwd = timed(fwd_only, args.iters, device)
    outs = model(**o)

    def post_only():
        proc.post_process_object_detection(
            outs, target_sizes=[(r2d.IMG_H, r2d.IMG_W)] * 6, threshold=args.score_thr)

    post = timed(post_only, args.iters, device)
    e2e = timed(lambda: one(fr), max(10, args.iters // 2), device)

    peak = (torch.cuda.max_memory_allocated() / 2**30) if device == 'cuda' else 0.0

    def stat(v):
        return {'mean_ms': round(statistics.mean(v), 2),
                'p50_ms': round(statistics.median(v), 2),
                'p95_ms': round(sorted(v)[int(0.95 * (len(v) - 1))], 2)}

    res = {'model': args.model, 'device': device, 'n_images': 6,
           'image_size': list(frames[0][0].size),
           'preprocess': stat(pre), 'forward': stat(fwd),
           'postprocess': stat(post), 'end_to_end': stat(e2e),
           'peak_gpu_gib': round(peak, 2)}

    print('\n' + '=' * 62)
    print(f"{'阶段':<16}{'均值 ms':>10}{'p50 ms':>10}{'p95 ms':>10}")
    print('-' * 62)
    for k, nm in (('preprocess', '预处理'), ('forward', '前向'), ('postprocess', '后处理'),
                  ('end_to_end', '端到端(单帧6图)')):
        s = res[k]
        print(f'{nm:<16}{s["mean_ms"]:>10.2f}{s["p50_ms"]:>10.2f}{s["p95_ms"]:>10.2f}')
    print('-' * 62)
    print(f'显存峰值 {peak:.2f} GiB')
    print('=' * 62)

    print(f"\n{'车速':<12}{'每秒位移':>10}{'端到端耗时内位移':>18}{'20ms 预算内位移':>16}")
    print('-' * 58)
    travel = {}
    for kmh in (30, 50, 70):
        v = kmh / 3.6
        d = v * res['end_to_end']['mean_ms'] / 1000.0
        travel[kmh] = round(d, 3)
        print(f'{kmh} km/h{"":<4}{v:>9.1f} m{d:>17.2f} m{v * 0.02:>15.2f} m')
    res['travel_m_per_inference'] = travel
    print('=' * 58)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f'\n已写出：{args.out}')


if __name__ == '__main__':
    main()
