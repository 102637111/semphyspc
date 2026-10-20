#!/usr/bin/env python3
"""实验C3：解冻微调 backbone —— 联合语义 + offset 回归，做级联第②环的最优状态。

背景：
  C2 冻结 backbone 只训 offset 头，实例 PQ 停在 0.50（backbone 特征缺乏实例判别力）。
  本脚本解冻全部参数，从 ckpt-offset.pth（语义 backbone + seg_head + 训好的 offset 头）
  出发，联合「语义 CE + offset L1」微调，让 backbone 学到实例级特征，把 PQ 往上推。

损失：
  loss = sem_weight * CE(logits, sem) + SmoothL1(offset, gt_offset)  (thing 点)
  sem_weight 默认 0.1，语义 CE 作轻正则，防止 backbone 只学 offset 而崩掉。

用法:
    python scripts/train_instance_finetune.py \
        --train data/nuscenes_seg_full_16k/train \
        --val data/nuscenes_seg_full_16k/val \
        --pretrained experiments/cascade_models/ckpt-offset.pth \
        --epochs 30 --bs 2 --lr 0.0001 --n_train 10000 --n_val 300 \
        --sem_weight 0.1 --save_dir experiments/cascade_models --seed 0
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.point_mamba_seg import PointMambaSeg
from scripts.train_instance_offset import (
    OffsetDataset, cluster_dbscan, match_instances, SIZE_BINS,
)


def build_model(pretrained):
    cfg = EasyDict({
        'NAME': 'PointMambaSeg', 'trans_dim': 384, 'depth': 12,
        'group_size': 32, 'num_group': 64, 'encoder_dims': 384,
        'in_channels': 6, 'num_classes': 32, 'upsample_k': 3,
        'rms_norm': False, 'drop_path': 0.1, 'drop_out': 0.0,
        'num_materials': 0, 'num_instances': 3,
    })
    model = PointMambaSeg(cfg)
    if pretrained is not None:
        sd = torch.load(pretrained, map_location='cpu')
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'[C3] 载入 ckpt-offset, missing {len(missing)} unexpected {len(unexpected)}',
              flush=True)
    return model


@torch.no_grad()
def evaluate(model, files, eps, min_points):
    """与 C2 一致：thing 点平移 offset 后 DBSCAN，与真值实例匹配，报 PQ。"""
    sum_iou = tp = fp = fn = 0.0
    bin_tp = {b: 0 for b in SIZE_BINS}
    bin_gt = {b: 0 for b in SIZE_BINS}
    for f in files:
        d = np.load(f)
        xyz = d['xyz'].astype(np.float32)
        rgb = d['rgb'].astype(np.float32)
        inst = d['inst'].astype(np.int64)
        pts = torch.from_numpy(np.concatenate([xyz, rgb], 1)).float().unsqueeze(0).cuda()
        _, _, off = model(pts)
        off = off[0].cpu().numpy()

        thing = inst > 0
        if thing.sum() < min_points:
            continue
        xyz_t = xyz[thing]
        gt = inst[thing]
        shifted = xyz_t + off[thing]
        pred = cluster_dbscan(shifted, eps, min_points)
        matched_ious, t, f_p, f_n, matched_gt = match_instances(gt, pred)
        sum_iou += float(np.sum(matched_ious))
        tp += t
        fp += f_p
        fn += f_n
        for g in np.unique(gt):
            sz = int((gt == g).sum())
            b = next(b for b in SIZE_BINS if b[0] <= sz < b[1])
            bin_gt[b] += 1
            if g in matched_gt:
                bin_tp[b] += 1
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    pq = sum_iou / max(tp + 0.5 * fp + 0.5 * fn, 1)
    return rec, prec, pq, bin_tp, bin_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--pretrained', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=0.0001)
    ap.add_argument('--n_train', type=int, default=10000)
    ap.add_argument('--n_val', type=int, default=300)
    ap.add_argument('--sem_weight', type=float, default=0.1)
    ap.add_argument('--eps', type=float, default=0.5)
    ap.add_argument('--min_points', type=int, default=3)
    ap.add_argument('--save_dir', type=str, default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_files = sorted(glob.glob(os.path.join(args.train, '*.npz')))
    val_files = sorted(glob.glob(os.path.join(args.val, '*.npz')))
    if args.n_train < len(train_files):
        train_files = rng.choice(train_files, args.n_train, replace=False).tolist()
    if args.n_val < len(val_files):
        val_files = rng.choice(val_files, args.n_val, replace=False).tolist()

    print(f'[C3] train {len(train_files)} / val {len(val_files)}, '
          f'sem_weight={args.sem_weight}', flush=True)
    model = build_model(args.pretrained).cuda()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[C3] 解冻全部参数 {trainable/1e6:.2f}M', flush=True)

    ds = OffsetDataset(train_files)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=True,
                                     num_workers=4, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl))
    l1 = nn.SmoothL1Loss(reduction='mean')

    best_pq = 0.0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        run_sem = run_off = 0.0
        nb = 0
        for pts, sem, inst, offset in dl:
            pts = pts.cuda()
            sem_g = sem.cuda()
            inst_g = inst.cuda()
            off_g = offset.cuda()

            logits, _, off = model(pts)  # (B,N,32), (B,N,3)

            # 语义 CE（轻正则）
            valid = sem_g >= 0
            sem_loss = F.cross_entropy(
                logits.reshape(-1, 32)[valid.reshape(-1)],
                sem_g.reshape(-1)[valid.reshape(-1)])

            # offset L1（thing 点）
            mask = inst_g > 0
            if mask.sum() == 0:
                off_loss = torch.tensor(0.0, device=pts.device)
            else:
                off_loss = l1(off[mask], off_g[mask])

            loss = args.sem_weight * sem_loss + off_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            run_sem += sem_loss.item()
            run_off += off_loss.item()
            nb += 1

        model.eval()
        rec, prec, pq, bin_tp, bin_gt = evaluate(model, val_files, args.eps, args.min_points)
        small = bin_tp[SIZE_BINS[0]] / max(bin_gt[SIZE_BINS[0]], 1)
        tag = ''
        if pq > best_pq:
            best_pq = pq
            tag = ' *best'
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(args.save_dir, 'ckpt-offset-finetuned.pth'))
        print(f'[epoch {epoch+1:3d}/{args.epochs}] sem={run_sem/max(nb,1):.4f} '
              f'off={run_off/max(nb,1):.4f} rec={rec:.4f} prec={prec:.4f} PQ={pq:.4f} '
              f'small={small:.4f} bestPQ={best_pq:.4f} 用时{time.time()-t0:.0f}s{tag}',
              flush=True)

    print(f'\n[C3 done] best PQ = {best_pq:.4f} '
          f'-> {args.save_dir}/ckpt-offset-finetuned.pth', flush=True)


if __name__ == '__main__':
    main()
