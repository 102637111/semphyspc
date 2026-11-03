#!/usr/bin/env python3
"""训练并保存射线序列 Mamba（MambaRay 单向）作为盲区特征提取器。

复用 eval_ray_blindspot 的真实多帧盲区 GT（rasterize/build_split/训练循环）。
锁数结论：MambaRay 的 flag recall/F1 最高（0.654/0.656），作「盲区邻接特征」的
特征源比 BiMambaRay 更合适（后者拿 recall 换 depth）。故这里只训单向。

输出 experiments/ray_blindspot_feat/raymamba_ckpt.pth：
  {model_state, config: {d_model, n_layers}}
"""
import os, sys, argparse, pickle
import numpy as np
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ray_sequence import MambaRay, N_RINGS, W_AZIM, R_THRESH, R_SCALE
from eval_ray_blindspot import (build_split, BlindSpotDataset, run_epoch, evaluate)

DATA_ROOT = '/mnt/data/nuscenes'
LIDAR_DIR = os.path.join(DATA_ROOT, 'samples', 'LIDAR_TOP')
META_PATH = 'experiments/ray_blindspot/blindspot_meta.pkl'
OUT_DIR = 'experiments/ray_blindspot_feat'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta_path', default=META_PATH)
    ap.add_argument('--out_dir', default=OUT_DIR)
    ap.add_argument('--n_train', type=int, default=1000)
    ap.add_argument('--n_val', type=int, default=200)
    ap.add_argument('--future_k', type=int, default=1)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--n_layers', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('[加载] 元数据缓存 ...')
    with open(args.meta_path, 'rb') as f:
        meta = pickle.load(f)
    keyframes, ego_trans, ego_rot = meta['keyframes'], meta['ego_trans'], meta['ego_rot']
    t_cs, q_cs = meta['lidar_extrinsic']

    # 连续帧对（同场景，间隔 future_k 个关键帧）
    pairs = []
    for i in range(len(keyframes) - args.future_k):
        if keyframes[i]['scene'] == keyframes[i + args.future_k]['scene']:
            pairs.append((i, i + args.future_k))
    print(f'[帧对] 共 {len(pairs)} 对（future_k={args.future_k}）')

    # 场景隔离划分（与 eval_ray_blindspot.main 一致）
    by_scene = OrderedDict()
    for j, (ia, _) in enumerate(pairs):
        by_scene.setdefault(keyframes[ia]['scene'], []).append(j)
    tr_idx, va_idx = [], []
    for sc, idxs in by_scene.items():
        if len(tr_idx) < args.n_train:
            tr_idx.extend(idxs)
        elif len(va_idx) < args.n_val:
            va_idx.extend(idxs)
    tr_idx, va_idx = tr_idx[:args.n_train], va_idx[:args.n_val]
    if len(tr_idx) < args.n_train or len(va_idx) < args.n_val:
        raise SystemExit(f'数据不足：train {len(tr_idx)} / val {len(va_idx)}')

    print(f'[构建] 栅格化 train {len(tr_idx)} / val {len(va_idx)} ...')
    Xtr, Ftr, Dtr, Etr, Rtr = build_split(tr_idx, pairs, keyframes, ego_trans, ego_rot, t_cs, q_cs)
    Xva, Fva, Dva, Eva, Rva = build_split(va_idx, pairs, keyframes, ego_trans, ego_rot, t_cs, q_cs)

    tr = DataLoader(BlindSpotDataset(Xtr, Ftr, Dtr, Etr, Rtr),
                    batch_size=args.bs, shuffle=True, num_workers=0)
    va = DataLoader(BlindSpotDataset(Xva, Fva, Dva, Eva, Rva),
                    batch_size=args.bs, shuffle=False, num_workers=0)

    model = MambaRay(args.d_model, args.n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    best_f1 = -1.0
    for ep in range(args.epochs):
        tl, tfl, trl = run_epoch(model, tr, opt, device, train=True)
        vl, vfl, vrl = run_epoch(model, va, opt, device, train=False)
        met = evaluate(model, va, device)
        print(f'  ep{ep}: train_loss={tl:.4f} val_loss={vl:.4f}  '
              f'flag_f1={met["f1"]:.4f} depth_mae={met["depth_mae_m"]:.3f}m')
        best_f1 = max(best_f1, met['f1'])
    print(f'[完成] best flag F1={best_f1:.4f}')

    ckpt_path = os.path.join(args.out_dir, 'raymamba_ckpt.pth')
    torch.save({'model_state': model.state_dict(),
                'config': {'d_model': args.d_model, 'n_layers': args.n_layers}},
               ckpt_path)
    print(f'[保存] 特征提取器 → {ckpt_path}')


if __name__ == '__main__':
    main()
