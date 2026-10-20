#!/usr/bin/env python3
"""训练冒烟测试：跑 N 个 batch 的完整训练步（数据加载→前向→loss→反向→optimizer.step），
验证 config 集成与训练循环无错误、无 OOM，不跑完整 epoch。

用法:
    python scripts/smoke_train.py --config cfgs/seg_nuscenes_grid_g1024.yaml --num_batches 8
"""
import os, time, types, argparse
import torch
import torch.nn as nn

from utils.config import get_config
from tools import builder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--num_batches', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=0, help='0=用 config 的 total_bs')
    ap.add_argument('--num_workers', type=int, default=4)
    args_cli = ap.parse_args()

    exp = '/tmp/smoke_train'
    os.makedirs(exp, exist_ok=True)

    args = types.SimpleNamespace(
        config=args_cli.config, exp_name='smoke', local_rank=0, resume=False,
        test=False, ckpts=None, num_workers=args_cli.num_workers, seed=0,
        distributed=False, use_gpu=torch.cuda.is_available(),
        log_name=os.path.splitext(os.path.basename(args_cli.config))[0],
        experiment_path=exp, tfboard_path=exp + '/tb',
    )

    config = get_config(args)
    if args_cli.batch_size > 0:
        config.total_bs = args_cli.batch_size
    config.dataset.train.others.bs = config.total_bs
    config.dataset.val.others.bs = config.total_bs

    print(f'[smoke] config={args_cli.config}  total_bs={config.total_bs}  '
          f'num_group={config.model.num_group}  use_grid_group={config.model.use_grid_group}')

    # ---- dataloader ----
    (_, train_loader) = builder.dataset_builder(args, config.dataset.train)
    print(f'[smoke] train batches = {len(train_loader)}')

    # ---- model ----
    base_model = builder.model_builder(config.model)
    base_model = nn.DataParallel(base_model).cuda()
    optimizer, _ = builder.build_opti_sche(base_model, config)

    base_model.train()
    base_model.zero_grad()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for idx, (taxonomy_ids, model_ids, data) in enumerate(train_loader):
        if idx >= args_cli.num_batches:
            break
        points = data[0].cuda()
        sem = data[1].cuda()

        ret, _ = base_model(points)
        loss = base_model.module.get_loss(ret, sem)
        loss.backward()
        if config.get('grad_norm_clip') is not None:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
        optimizer.step()
        base_model.zero_grad()

        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f'[smoke] batch {idx}: loss={loss.item():.4f}  '
              f'{dt/(idx+1):.2f}s/batch  peak_mem={mem:.2f} GiB')

    print(f'[smoke] 完成：{args_cli.num_batches} 个 batch 训练步无错误。')


if __name__ == '__main__':
    main()
