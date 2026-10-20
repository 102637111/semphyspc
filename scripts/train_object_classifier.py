#!/usr/bin/env python3
"""实验B2：训练物体级分类器（PointMambaScan，11 类 nuScenes thing 物体）。

输入：scripts/extract_nuscenes_objects.py 产出的 objects.npz。
流程：
  1. 载入 objects.npz，按类分层 80/20 划分 train/val。
  2. 每个物体：居中(centroid) + 单位球归一化 + 随机 z 旋转(仅训练) + 采样到 1024 点。
  3. PointMambaScan(cls_dim=11, in_channels=6)，可选载入 ModelNet40-RGB 预训练 backbone。
  4. 类别均衡采样(每类等量)训练，逐类/宏准确率评估。

用法:
    python scripts/train_object_classifier.py \
        --data /mnt/data/datasets/UR-MAT/nuscenes_objects/objects.npz \
        --pretrained experiments/finetune_modelnet_rgb/cfgs/color_mamba_v1/20260723-154515/ckpt-best.pth \
        --epochs 80 --bs 32 --lr 0.001 --seed 0
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.point_mamba_scan import PointMambaScan


def build_model(num_classes, pretrained):
    cfg = EasyDict({
        'NAME': 'PointMambaScan', 'trans_dim': 384, 'depth': 12,
        'cls_dim': num_classes, 'group_size': 32, 'num_group': 64,
        'encoder_dims': 384, 'in_channels': 6, 'rms_norm': False,
        'drop_path': 0.1, 'drop_out': 0.0,
        'use_cls_token': False, 'max_head': False, 'avg_head': True,
    })
    model = PointMambaScan(cfg)
    if pretrained is not None:
        ckpt = torch.load(pretrained, map_location='cpu')
        base = ckpt.get('base_model') or ckpt.get('model')
        base = {k.replace('module.', ''): v for k, v in base.items()}
        # 按形状过滤：只迁移 shape 匹配的层（backbone + 分类头共享层），
        # 跳过 shape 不匹配的分类头末层(40->11 类)。
        model_state = model.state_dict()
        filtered = {k: v for k, v in base.items()
                    if k in model_state and model_state[k].shape == v.shape}
        n_skip = len(base) - len(filtered)
        model.load_state_dict(filtered, strict=False)
        print(f'[B2] 迁移预训练 {len(filtered)} 层, 跳过 {n_skip} 层(形状不匹配)')
    else:
        model.load_model_from_ckpt(None)  # 随机初始化
    return model


class ObjectDataset(torch.utils.data.Dataset):
    def __init__(self, points, offsets, labels, obj_indices, train, npoints=1024, seed=0):
        self.points = points
        self.offsets = offsets
        self.labels = labels
        self.obj_indices = np.asarray(obj_indices, dtype=np.int64)
        self.train = train
        self.npoints = npoints
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.obj_indices)

    def _prep(self, oi):
        s, e = self.offsets[oi], self.offsets[oi + 1]
        pts = self.points[s:e].astype(np.float32)  # (Ni, 6) [xyz, rgb]
        xyz = pts[:, :3]
        rgb = pts[:, 3:6]

        # 居中 + 单位球归一化（去除平移/尺度，绕开"小物体淹没在场景"问题）
        xyz = xyz - xyz.mean(0)
        norm = np.linalg.norm(xyz, axis=1).max()
        if norm > 1e-6:
            xyz = xyz / norm

        if self.train:
            # z 轴随机旋转 + 轻微缩放抖动
            ang = self.rng.uniform(0, 2 * np.pi)
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            xyz = xyz @ R.T
            xyz = xyz * self.rng.uniform(0.9, 1.1)

        Ni = xyz.shape[0]
        if Ni > self.npoints:
            idx = self.rng.choice(Ni, self.npoints, replace=False)
        else:
            idx = self.rng.choice(Ni, self.npoints, replace=True)
        xyz, rgb = xyz[idx], rgb[idx]
        return np.concatenate([xyz, rgb], axis=1).astype(np.float32)

    def __getitem__(self, i):
        oi = int(self.obj_indices[i])
        pts = self._prep(oi)
        return torch.from_numpy(pts).float(), self.labels[oi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='objects.npz 路径')
    ap.add_argument('--pretrained', type=str, default=None)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--target_per_class', type=int, default=1500)
    ap.add_argument('--save_dir', type=str, default=None, help='保存分类器完整模型')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    d = np.load(args.data, allow_pickle=True)
    points = d['points']
    offsets = d['offsets']
    labels = d['labels'].astype(np.int64)
    class_names = list(d['class_names'])
    num_classes = len(class_names)
    print(f"[B2] 载入 {len(labels)} 个物体, {num_classes} 类: {class_names}", flush=True)

    # 分层 80/20 划分
    train_idx, val_idx = [], []
    for c in range(num_classes):
        cidx = np.where(labels == c)[0]
        rng.shuffle(cidx)
        n_val = max(1, int(len(cidx) * 0.2))
        val_idx.extend(cidx[:n_val].tolist())
        train_idx.extend(cidx[n_val:].tolist())
    print(f"[B2] train={len(train_idx)} val={len(val_idx)}", flush=True)

    train_ds = ObjectDataset(points, offsets, labels, train_idx, train=True, seed=args.seed)
    val_ds = ObjectDataset(points, offsets, labels, val_idx, train=False, seed=args.seed)

    # 类别均衡采样（每 epoch 每类采样 target_per_class 个）
    def balanced_batch_indices(ds_indices):
        per_class = [[] for _ in range(num_classes)]
        for oi in ds_indices:
            per_class[labels[oi]].append(oi)
        out = []
        for c in range(num_classes):
            pool = per_class[c]
            out.extend(rng.choice(pool, args.target_per_class, replace=True).tolist())
        rng.shuffle(out)
        return out

    model = build_model(num_classes, args.pretrained).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total_steps = args.epochs * (num_classes * args.target_per_class // args.bs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    loss_ce = torch.nn.CrossEntropyLoss()

    print(f"[B2] 开始训练, 每 epoch ~{num_classes*args.target_per_class//args.bs} batch", flush=True)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        indices = balanced_batch_indices(train_idx)
        order = np.arange(len(indices))
        rng.shuffle(order)
        n_batch = 0
        run_loss = 0.0
        for b in range(0, len(order), args.bs):
            bidx = [indices[o] for o in order[b:b + args.bs]]
            pts = torch.stack([torch.from_numpy(train_ds._prep(oi)) for oi in bidx]).cuda()
            lab = torch.tensor([labels[oi] for oi in bidx], dtype=torch.long).cuda()
            logits = model(pts)
            loss = loss_ce(logits, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            run_loss += loss.item()
            n_batch += 1

        # 评估
        model.eval()
        correct = np.zeros(num_classes, dtype=np.int64)
        total = np.zeros(num_classes, dtype=np.int64)
        with torch.no_grad():
            for b in range(0, len(val_idx), args.bs):
                bidx = val_idx[b:b + args.bs]
                pts = torch.stack([torch.from_numpy(val_ds._prep(oi)) for oi in bidx]).cuda()
                lab = torch.tensor([labels[oi] for oi in bidx], dtype=torch.long).cuda()
                pred = model(pts).argmax(-1).cpu().numpy()
                for j, oi in enumerate(bidx):
                    c = labels[oi]
                    total[c] += 1
                    correct[c] += (pred[j] == c)

        per_class = correct / np.maximum(total, 1)
        macro = per_class.mean()
        overall = correct.sum() / max(total.sum(), 1)
        print(f"[epoch {epoch+1:3d}/{args.epochs}] loss={run_loss/max(n_batch,1):.4f} "
              f"val_acc={overall:.4f} macro_acc={macro:.4f} 用时{time.time()-t0:.0f}s",
              flush=True)

    print("\n================ 实验B2 结果（物体级分类器，逐类准确率）================", flush=True)
    for c in range(num_classes):
        print(f"  {c:2d} {class_names[c]:<16s} : {per_class[c]:.4f}  (n={total[c]})", flush=True)
    print(f"  宏平均准确率(macro) : {macro:.4f}", flush=True)
    print(f"  整体准确率(overall) : {overall:.4f}", flush=True)
    print("======================================================================", flush=True)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        out = os.path.join(args.save_dir, 'ckpt-cls.pth')
        torch.save(model.state_dict(), out)
        print(f'[B2] 已保存分类器完整模型 -> {out}', flush=True)


if __name__ == '__main__':
    main()
