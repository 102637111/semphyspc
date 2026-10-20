#!/usr/bin/env python3
"""实验C2：学习式实例分割 —— offset 回归(投票到质心) + 聚类，闭合级联第②环。

背景：
  几何基线(DBSCAN, min_pts=3, eps=0.5)在 16k 点云上：召回46%/精度45%/PQ0.38，
  小物体(<50点, 占84%实例)召回仅 41% —— 稀疏小物体是根本瓶颈。
  本脚本用 PointMambaSeg 的 instance_head 训逐点 offset(dx,dy,dz)=质心-点坐标，
  推理时把 thing 点平移到预测质心再 DBSCAN 聚类，靠学习到的上下文
  把稀疏小物体的点"拉"到一起，试图超越几何基线。

流程：
  1. 载入语义预训练 backbone，冻结，只训 instance_head(3 通道 offset)。
  2. 损失：smooth L1(预测 offset, 真值 offset) 于 thing 点(inst>0)。
  3. 评估：平移点 → DBSCAN → 与真值实例贪心匹配，报召回/精度/PQ(按大小分桶)。

用法:
    python scripts/train_instance_offset.py \
        --train data/nuscenes_seg_full_16k/train \
        --val data/nuscenes_seg_full_16k/val \
        --pretrained experiments/seg_nuscenes_full_16k/cfgs/singletask_sem/20260821-221520/ckpt-best.pth \
        --epochs 30 --bs 1 --lr 0.001 --n_train 10000 --n_val 300 --seed 0
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.point_mamba_seg import PointMambaSeg

THING_CLASSES = set(range(1, 24))
IOU_THRESH = 0.5
SIZE_BINS = [(0, 50), (50, 200), (200, 1000), (1000, 10 ** 9)]


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
        ck = torch.load(pretrained, map_location='cpu')
        base = ck.get('base_model') or ck.get('model')
        base = {k.replace('module.', ''): v for k, v in base.items()}
        ms = model.state_dict()
        # 按形状过滤：instance_head 是新的(3 通道 offset)，语义 ckpt 里没有 -> 跳过
        filt = {k: v for k, v in base.items() if k in ms and ms[k].shape == v.shape}
        model.load_state_dict(filt, strict=False)
        print(f'[C2] 迁移预训练 {len(filt)}/{len(base)} 层, '
              f'instance_head 从头初始化', flush=True)
    return model


def freeze_backbone(model):
    """冻结除 instance_head 外所有参数，只训 offset 头。"""
    for name, p in model.named_parameters():
        if 'instance_head' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False


class OffsetDataset(torch.utils.data.Dataset):
    """加载 npz，返回 (pts, sem, inst, offset)。offset 为 thing 点指向实例质心的 3D 向量。"""

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        xyz = d['xyz'].astype(np.float32)
        rgb = d['rgb'].astype(np.float32)
        sem = d['sem'].astype(np.int64)
        inst = d['inst'].astype(np.int64)

        offset = np.zeros_like(xyz)
        thing = inst > 0
        for oid in np.unique(inst[thing]):
            m = inst == oid
            offset[m] = xyz[m].mean(0) - xyz[m]

        pts = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
        return (torch.from_numpy(pts).float(),
                torch.from_numpy(sem).long(),
                torch.from_numpy(inst).long(),
                torch.from_numpy(offset).float())


def cluster_dbscan(xyz, eps, min_points):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points,
                                         print_progress=False))


def match_instances(gt_labels, pred_labels):
    """贪心匹配，返回 (matched_ious, tp, fp, fn, matched_gt)."""
    gt_ids = np.unique(gt_labels)
    pred_ids = np.unique(pred_labels[pred_labels >= 0])
    if len(pred_ids) == 0:
        return [], 0, 0, len(gt_ids), set()
    pred_sets = {p: set(np.where(pred_labels == p)[0]) for p in pred_ids}
    gt_sets = {g: set(np.where(gt_labels == g)[0]) for g in gt_ids}
    matched_ious, matched_gt, matched_pred = [], set(), set()
    for p in sorted(pred_sets, key=lambda x: -len(pred_sets[x])):
        best_g, best_iou = None, 0.0
        for g in gt_sets:
            if g in matched_gt:
                continue
            inter = len(pred_sets[p] & gt_sets[g])
            union = len(pred_sets[p] | gt_sets[g])
            iou = inter / max(union, 1)
            if iou > best_iou:
                best_iou, best_g = iou, g
        if best_iou >= IOU_THRESH:
            matched_ious.append(best_iou)
            matched_gt.add(best_g)
            matched_pred.add(p)
    tp = len(matched_ious)
    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - len(matched_gt)
    return matched_ious, tp, fp, fn, matched_gt


@torch.no_grad()
def evaluate(model, files, eps, min_points):
    """对 val 子集：thing 点平移预测 offset 后 DBSCAN，与真值实例匹配。"""
    sum_iou = tp = fp = fn = 0.0
    bin_tp = {b: 0 for b in SIZE_BINS}
    bin_gt = {b: 0 for b in SIZE_BINS}
    t0 = time.time()
    for fi, f in enumerate(files):
        d = np.load(f)
        xyz = d['xyz'].astype(np.float32)
        rgb = d['rgb'].astype(np.float32)
        inst = d['inst'].astype(np.int64)
        pts = torch.from_numpy(np.concatenate([xyz, rgb], 1)).float().unsqueeze(0).cuda()
        _, _, off = model(pts)
        off = off[0].cpu().numpy()  # (N,3)

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
        if (fi + 1) % 200 == 0:
            print(f"    [eval {fi+1}/{len(files)}] 用时 {time.time()-t0:.0f}s", flush=True)

    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    pq = sum_iou / max(tp + 0.5 * fp + 0.5 * fn, 1)
    return rec, prec, pq, bin_tp, bin_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--pretrained', type=str, default=None)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--n_train', type=int, default=10000)
    ap.add_argument('--n_val', type=int, default=300)
    ap.add_argument('--eps', type=float, default=0.5)
    ap.add_argument('--min_points', type=int, default=3)
    ap.add_argument('--save_dir', type=str, default=None, help='保存 offset 头完整模型')
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

    print(f'[C2] train {len(train_files)} / val {len(val_files)}', flush=True)

    model = build_model(args.pretrained).cuda()
    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[C2] 可训练参数 {trainable/1e3:.1f}K (仅 instance_head)', flush=True)

    ds = OffsetDataset(train_files)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=True,
                                     num_workers=4, drop_last=False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl))
    loss_fn = nn.SmoothL1Loss(reduction='mean')

    print(f'[C2] 训练 {args.epochs} epochs, 每 epoch {len(dl)} batch', flush=True)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        run_loss, nb = 0.0, 0
        for pts, sem, inst, offset in dl:
            pts = pts.cuda()
            inst_g = inst.cuda()
            offset_g = offset.cuda()
            _, _, off = model(pts)  # (B,N,3)
            mask = (inst_g > 0)
            if mask.sum() == 0:
                continue
            loss = loss_fn(off[mask], offset_g[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item()
            nb += 1

        model.eval()
        rec, prec, pq, bin_tp, bin_gt = evaluate(model, val_files, args.eps, args.min_points)
        s = (f'[epoch {epoch+1:3d}/{args.epochs}] loss={run_loss/max(nb,1):.5f} '
             f'rec={rec:.4f} prec={prec:.4f} PQ={pq:.4f} '
             f'小物体<50点召回={bin_tp[SIZE_BINS[0]]/max(bin_gt[SIZE_BINS[0]],1):.4f} '
             f'用时{time.time()-t0:.0f}s')
        print(s, flush=True)

    print('\n================ 实验C2 结果（offset 回归实例分割）================', flush=True)
    rec, prec, pq, bin_tp, bin_gt = evaluate(model, val_files, args.eps, args.min_points)
    print(f'  最终  召回={rec:.4f} 精度={prec:.4f} PQ={pq:.4f}', flush=True)
    for b in SIZE_BINS:
        r = bin_tp[b] / max(bin_gt[b], 1)
        print(f'    {b[0]:>5}-{b[1]:<6} 点 : {r:.4f}  (GT实例 {bin_gt[b]})', flush=True)
    print('  几何基线对照(DBSCAN eps0.5 min_pts3): 召回0.4605 精度0.4546 PQ0.3812 小物体0.4087',
          flush=True)
    print('=====================================================================', flush=True)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        out = os.path.join(args.save_dir, 'ckpt-offset.pth')
        torch.save(model.state_dict(), out)
        print(f'[C2] 已保存 offset 头完整模型 -> {out}', flush=True)


if __name__ == '__main__':
    main()
