#!/usr/bin/env python3
"""跨数据集迁移学习：冻结 nuScenes 预训练 backbone + 微调 8 类分割头 on Semantic3D。

背景：
  PointMambaSeg 的 backbone（Group/Encoder/Mamba blocks）在 nuScenes(32 类) 上预训练后，
  冻结其全部权重，只把 seg_head 换成 8 类并微调，在 Semantic3D(8 类) 上评测。
  对照「从零训练整模型」基线，验证 RGB+XYZ 早期融合 backbone 的特征是否跨数据集迁移。

方法：
  - 15 个 Semantic3D labeled 场景随机 split 12 train / 3 val（seed 固定）。
  - 迁移组：载入 32 类预训练权重(strict=False 跳过末层卷积形状不匹配)，
    冻结除 seg_head 外全部参数，只训 seg_head。
  - 从零组：随机初始化，训练整模型（不冻结）。
  - 均训 ~50 epoch，逐场景 CE loss，报告 val 集 mIoU / OA。

用法:
    python scripts/transfer_semantic3d.py \
        --config cfgs/seg_nuscenes_full_16k.yaml \
        --ckpt experiments/seg_nuscenes_full_16k/cfgs/singletask_sem/20260821-221520/ckpt-best.pth \
        --s3d_dir /mnt/data/datasets/Semantic3D/seg_npz_labeled/train \
        --epochs 50 --lr 0.001 --seed 0
"""

import os, sys, glob, argparse, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import builder
from utils.config import get_config

S3D_NAMES = ['man-made terrain', 'natural terrain', 'high vegetation', 'low vegetation',
             'buildings', 'hard scape', 'scanning artefacts', 'cars']


def load_scenes(s3d_dir, npoints):
    """加载全部 Semantic3D labeled 场景为 (pts, sem) 列表。"""
    files = sorted(glob.glob(os.path.join(s3d_dir, '*.npz')))
    scenes = []
    for f in files:
        d = np.load(f)
        xyz = d['xyz'].astype(np.float32)
        rgb = d['rgb'].astype(np.float32)
        sem = d['sem'].astype(np.int64)
        if xyz.shape[0] > npoints:
            idx = np.random.choice(xyz.shape[0], npoints, replace=False)
            xyz, rgb, sem = xyz[idx], rgb[idx], sem[idx]
        scenes.append((np.concatenate([xyz, rgb], 1).astype(np.float32), sem))
    return scenes


def build_8cls_model(config, ckpt):
    config.model.num_classes = 8
    config.model.num_instances = 0
    config.model.num_materials = 0
    model = builder.model_builder(config.model)
    if ckpt:
        # strict=False 仍会对末层卷积(32 vs 8)报 shape mismatch，需手动过滤
        sd = torch.load(ckpt, map_location='cpu')
        base = sd.get('base_model') or sd.get('model')
        base = {k.replace('module.', ''): v for k, v in base.items()}
        own = model.state_dict()
        filtered = {}
        n_skip = 0
        for k, v in base.items():
            if k in own and own[k].shape == v.shape:
                filtered[k] = v
            else:
                n_skip += 1
        model.load_state_dict(filtered, strict=False)
        print(f'[transfer] 载入预训练 {len(filtered)}/{len(base)} 个参数 (跳过 {n_skip} 个形状不匹配)')
    return model


def freeze_backbone(model):
    """冻结除 seg_head 外全部参数。"""
    for name, p in model.named_parameters():
        if 'seg_head' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'[transfer] 可训练参数 {n_train:,} / {n_total:,} (仅 seg_head)')
    return model


def eval_miou(model, scenes, npoints):
    model.eval()
    conf = np.zeros((8, 8), dtype=np.int64)
    with torch.no_grad():
        for pts, sem in scenes:
            p = torch.from_numpy(pts).float().unsqueeze(0).cuda()
            logits = model(p)[0]
            pred = logits.argmax(-1)[0].cpu().numpy().astype(np.int64)
            valid = sem >= 0
            pred_v, sem_v = pred[valid], sem[valid]
            for c in range(8):
                m = sem_v == c
                if m.sum():
                    conf[c] += np.bincount(pred_v[m], minlength=8)
    ious = []
    for c in range(8):
        gt = conf[c].sum(); tp = conf[c, c]; pred_sum = conf[:, c].sum()
        union = gt + pred_sum - tp
        ious.append(tp / union if union else float('nan'))
    miou = float(np.mean([i for i in ious if not np.isnan(i)]))
    oa = np.trace(conf) / max(conf.sum(), 1)
    return miou, oa, ious


def train_transfer(model, train_scenes, val_scenes, epochs, lr, npoints, seed):
    model = model.cuda()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.05)
    crit = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    best = 0.0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train_scenes))
        tot_loss = 0.0
        for i in order:
            pts, sem = train_scenes[i]
            # 训练增强：绕 Z 轴随机旋转
            ang = rng.uniform(0, 2 * np.pi)
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            xyz_r = pts[:, :3] @ R.T
            pts_r = np.concatenate([xyz_r, pts[:, 3:]], 1)
            p = torch.from_numpy(pts_r).float().unsqueeze(0).cuda()
            s_t = torch.from_numpy(sem).long().unsqueeze(0).cuda()
            logits = model(p)[0]
            loss = crit(logits.reshape(-1, 8), s_t.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item()
        miou, oa, _ = eval_miou(model, val_scenes, npoints)
        tag = ' *best' if miou > best else ''
        if miou > best:
            best = miou
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'[transfer] epoch {ep+1:3d}/{epochs} loss={tot_loss/len(train_scenes):.4f} '
                  f'val mIoU={miou:.4f} OA={oa:.4f} 用时{time.time()-t0:.0f}s{tag}', flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--s3d_dir', required=True)
    ap.add_argument('--npoints', type=int, default=16384)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    args.local_rank = 0; args.use_gpu = torch.cuda.is_available()
    args.distributed = False; args.resume = False
    args.experiment_path = '/tmp/transfer_s3d'
    os.makedirs(args.experiment_path, exist_ok=True)
    os.environ['LOCAL_RANK'] = '0'
    config = get_config(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scenes = load_scenes(args.s3d_dir, args.npoints)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(scenes))
    n_val = 3
    val_idx = sorted(idx[:n_val]); train_idx = sorted(idx[n_val:])
    train_scenes = [scenes[i] for i in train_idx]
    val_scenes = [scenes[i] for i in val_idx]
    print(f'[transfer] Semantic3D {len(scenes)} 场景 -> train {len(train_scenes)} / val {len(val_scenes)}',
          flush=True)

    # 迁移组：冻结 backbone + 训头
    print('\n=== 迁移组（冻结 nuScenes backbone + 微调 8 类头）===', flush=True)
    m1 = freeze_backbone(build_8cls_model(config, args.ckpt))
    b1 = train_transfer(m1, train_scenes, val_scenes, args.epochs, args.lr, args.npoints, args.seed)

    # 从零组：随机初始化，训练整模型
    print('\n=== 从零组（随机初始化，训练整模型）===', flush=True)
    m2 = build_8cls_model(config, None).cuda()
    for p in m2.parameters():
        p.requires_grad = True
    b2 = train_transfer(m2, train_scenes, val_scenes, args.epochs, args.lr, args.npoints, args.seed)

    print('\n' + '=' * 56, flush=True)
    print(f'迁移组（冻结 backbone+训头） best mIoU = {b1:.4f}', flush=True)
    print(f'从零组（训练整模型）         best mIoU = {b2:.4f}', flush=True)
    print('=' * 56, flush=True)


if __name__ == '__main__':
    main()
