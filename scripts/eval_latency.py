#!/usr/bin/env python3
"""D⑧ 时延分项：3D 分割模型的单帧/批前向时延（论文 §4.16 只测了 2D 检测器 67.66ms，缺 3D 分割时延）。

测量模型一次前向的纯 GPU 计算时延（含 host→device 传输 + 前向 + 取 logits），用 torch.cuda.Event
精确计时（GPU 侧），预热后取 N 次统计（mean / median / p95 / max）。分别测 B=1（单帧，导航实时口径）
与 B=4（批，离线建图口径）。

用法:
    python3 scripts/eval_latency.py \
        --config cfgs/seg_16k_ng6144_g050_merged.yaml \
        --ckpts experiments/seg_16k_ng6144_g050_merged/cfgs/merged_pretrain_ft/20260927-070312/ckpt-best.pth \
        --n 100 --out-json results/latency_merged_pretrain_ft.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import builder
from utils.config import get_config
from utils.logger import get_root_logger, print_log


def time_forward(model, x, n_warmup, n_iter):
    """返回 (list[ms]) 每次前向的 GPU 侧耗时。"""
    for _ in range(n_warmup):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    ts = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(n_iter):
        start.record()
        with torch.no_grad():
            model(x)
        end.record()
        torch.cuda.synchronize()
        ts.append(start.elapsed_time(end))
    return ts


def stats(ts):
    a = np.array(ts)
    return {
        'mean_ms': round(float(a.mean()), 3),
        'median_ms': round(float(np.median(a)), 3),
        'p95_ms': round(float(np.percentile(a, 95)), 3),
        'max_ms': round(float(a.max()), 3),
        'n': int(len(a)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpts', required=True)
    ap.add_argument('--n', type=int, default=100, help='计时迭代次数（预热 10 次不计）')
    ap.add_argument('--npoints', type=int, default=16384)
    ap.add_argument('--in-channels', type=int, default=6)
    ap.add_argument('--tag', default='d8_latency', help='结果标签')
    ap.add_argument('--out-json', default=None)
    args = ap.parse_args()

    args.local_rank = 0
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.resume = False
    args.experiment_path = '/tmp/eval_latency_exp'
    os.makedirs(args.experiment_path, exist_ok=True)
    args.log_name = os.path.basename(args.config).replace('.yaml', '')
    os.environ['LOCAL_RANK'] = '0'
    logger = get_root_logger(name='eval_latency')

    config = get_config(args, logger=logger)
    print_log(f'加载 checkpoint: {args.ckpts}', logger=logger)
    model = builder.model_builder(config.model)
    builder.load_model(model, args.ckpts, logger=logger)
    model = nn.DataParallel(model).cuda()
    model.eval()

    rec = {'tag': args.tag, 'config': args.config, 'ckpts': args.ckpts,
           'npoints': args.npoints, 'in_channels': args.in_channels,
           'device': torch.cuda.get_device_name(0), 'bs': {}}

    for bs in (1, 4):
        x = torch.randn(bs, args.npoints, args.in_channels).cuda()
        ts = time_forward(model, x, n_warmup=10, n_iter=args.n)
        s = stats(ts)
        rec['bs'][str(bs)] = s
        print_log(f'B={bs}: mean={s["mean_ms"]}ms median={s["median_ms"]}ms '
                  f'p95={s["p95_ms"]}ms max={s["max_ms"]}ms', logger=logger)
        # 单帧吞吐（FPS）
        print_log(f'  B={bs} 吞吐 ≈ {1000 / s["median_ms"]:.1f} FPS（按中位数）', logger=logger)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        print_log(f'结果已写入 {args.out_json}', logger=logger)


if __name__ == '__main__':
    main()
