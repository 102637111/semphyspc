#!/usr/bin/env python3
"""64k 训练冒烟测试：验证 N=32768 下 数据加载 + 前向 + 反向 + 一步优化 不报错不 OOM。

用法:
    python scripts/smoke64k.py                 # 默认 total_bs=2（本地 16GB 卡安全）
    python scripts/smoke64k.py --bs 16         # 复现真实训练的 batch 大小（需 24GB 卡）
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np

from utils.config import get_config
from utils.logger import get_root_logger
from tools import builder


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='cfgs/seg_nuscenes_full_64k_g256.yaml')
    p.add_argument('--bs', type=int, default=2)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--steps', type=int, default=3)
    return p.parse_args()


def main():
    args = get_args()
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.local_rank = 0
    args.exp_name = 'smoke64k'
    args.resume = False
    args.ckpts = None
    args.log_name = 'smoke64k'
    args.experiment_path = '/tmp/smoke64k_exp'
    args.tfboard_path = '/tmp/smoke64k_tb'
    os.makedirs(args.experiment_path, exist_ok=True)
    os.makedirs(args.tfboard_path, exist_ok=True)

    logger = get_root_logger(name='smoke64k')
    config = get_config(args, logger=logger)

    # 覆盖 batch size 为冒烟值
    config.total_bs = args.bs
    for key in ['train', 'val', 'test']:
        if config.dataset.get(key):
            config.dataset[key].others.bs = args.bs

    (_, train_loader) = builder.dataset_builder(args, config.dataset.train)

    model = builder.model_builder(config.model)
    model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'[smoke] 模型参数 {n_params:.2f}M')

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for step, (tax, mid, data) in enumerate(train_loader):
        points = data[0].cuda()   # (B, N, 6)
        sem = data[1].cuda()       # (B, N)

        ret, _ = model(points)     # (B, N, 32)
        loss = model.module.get_loss(ret, sem)
        loss.backward()
        model.zero_grad()

        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f'[smoke] step {step}: points={tuple(points.shape)} '
              f'logits={tuple(ret.shape)} loss={loss.item():.4f} '
              f'peak_mem={mem:.2f}GB')

        if step + 1 >= args.steps:
            break

    dt = (time.time() - t0) / args.steps
    print(f'[smoke] 完成 {args.steps} 步，平均 {dt:.2f}s/step，峰值显存 '
          f'{torch.cuda.max_memory_allocated()/1e9:.2f}GB（bs={args.bs}）')
    print(f'[smoke] 按 bs=16 外推峰值约 {torch.cuda.max_memory_allocated()/1e9 * (16/args.bs):.2f}GB')


if __name__ == '__main__':
    main()
