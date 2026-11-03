#!/usr/bin/env python3
"""给 16k 逐点分割数据集生成 per-point「射线盲区邻接」特征（learned）。

对每个 npz（token = LIDAR_TOP 关键帧 sample_data token）：
  1. 由 sample_data.json 找到原始 pcd 路径（samples/LIDAR_TOP/<filename>）
  2. rasterize 成 32×1084 range image（每 (ring,bin) 取最近表面）
  3. 跑 MambaRay → flag_prob = sigmoid(flag_head)，只在空射线上有意义（隐藏近距障碍概率）
  4. blind_adj[ring,bin] = ±window 方位角内空射线 flag_prob 的极大值（盲区邻接度）
  5. 每个原始点按 (ring,bin) 取 blind_adj；cKDTree 把原始点回射到 16k npz 点（FPS 子集，NN 距离≈0）
  6. 存 {token}_ray.npz  {blind_adj:(N,) float32}，顺序与 npz 的 xyz 一致

用法：
  python3 scripts/build_ray_feature.py --ckpt experiments/ray_blindspot_feat/raymamba_ckpt.pth --limit 3   # 冒烟
  python3 scripts/build_ray_feature.py --ckpt experiments/ray_blindspot_feat/raymamba_ckpt.pth              # 全量
"""
import os, sys, json, argparse
import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import maximum_filter1d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ray_sequence import MambaRay, N_RINGS, W_AZIM, R_SCALE
from eval_ray_blindspot import rasterize, load_pcd

DATA_ROOT = '/mnt/data/nuscenes'
LIDAR_DIR = os.path.join(DATA_ROOT, 'samples', 'LIDAR_TOP')
META_DIR = os.path.join(DATA_ROOT, 'v1.0-trainval')


def build_token2pcd():
    """sample_data.json → {token: filename}（仅 LIDAR_TOP 关键帧）。"""
    sd = json.load(open(os.path.join(META_DIR, 'sample_data.json')))
    m = {}
    for r in sd:
        if 'LIDAR_TOP' in r['filename'] and r['fileformat'] == 'pcd' and r['is_key_frame']:
            m[r['token']] = r['filename']
    return m


def compute_blind_adj(R, flag_prob, empty, window):
    """空射线上 flag_prob 的方位角邻域极大值 → per-ray 盲区邻接度 (32,W)。"""
    fp = flag_prob * empty.astype(np.float32)  # 只取空射线的隐藏障碍概率
    adj = maximum_filter1d(fp, size=2 * window + 1, axis=1, mode='nearest')
    return adj.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='data/nuscenes_seg_full_16k')
    ap.add_argument('--splits', default='train,val')
    ap.add_argument('--window', type=int, default=8, help='方位角邻域半宽（bin，≈0.33°/bin）')
    ap.add_argument('--limit', type=int, default=0, help='0=全量；>0 只处理每 split 前 N 帧（冒烟）')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[加载] MambaRay 特征提取器 ...')
    ckpt = torch.load(args.ckpt, map_location=device)
    model = MambaRay(ckpt['config']['d_model'], ckpt['config']['n_layers']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    token2pcd = build_token2pcd()
    print(f'[索引] LIDAR_TOP 关键帧 {len(token2pcd)}')

    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    for split in splits:
        d = Path(args.data) / split
        files = sorted(d.glob('*.npz'))
        # sidecar 独立目录，避免污染数据目录（数据集的 glob('*.npz') 会把 *_ray.npz 也当样本）
        out_dir = Path(args.data) / 'ray_feat' / split
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.limit > 0:
            files = files[:args.limit]
        n_miss = n_far = 0
        adj_sum = adj_n = 0.0
        for i, fp in enumerate(files):
            token = fp.stem
            if token not in token2pcd:
                n_miss += 1
                continue
            pcd_path = os.path.join(LIDAR_DIR, os.path.basename(token2pcd[token]))
            if not os.path.exists(pcd_path):
                n_miss += 1
                continue

            pts = load_pcd(pcd_path)                     # (N,5) x,y,z,intensity,ring
            R, I = rasterize(pts)                        # (32,W) range, intensity
            empty = (R == 0)
            x = np.stack([R / R_SCALE * (1 - empty), I / 255.0 * (1 - empty), empty], axis=0)
            xb = torch.from_numpy(x).float().to(device).unsqueeze(0)   # (1,3,32,W)
            with torch.no_grad():
                flag_logit, _ = model(xb)
            flag_prob = torch.sigmoid(flag_logit)[0].cpu().numpy()      # (32,W)

            adj = compute_blind_adj(R, flag_prob, empty, args.window)   # (32,W)

            # 每个原始点 → (ring, bin) → blind_adj
            ring = pts[:, 4].astype(np.int64)
            az = np.arctan2(pts[:, 1], pts[:, 0])
            b = ((az + np.pi) / (2 * np.pi) * W_AZIM).astype(np.int64) % W_AZIM
            valid = (ring >= 0) & (ring < N_RINGS)
            feat_raw = np.zeros(len(pts), np.float32)
            feat_raw[valid] = adj[ring[valid], b[valid]]

            # cKDTree：16k npz 点（FPS 子集）→ 原始点
            z = np.load(str(fp))
            xyz = z['xyz'].astype(np.float32)
            tree = cKDTree(pts[:, :3].astype(np.float64))
            dd, ii = tree.query(xyz.astype(np.float64), k=1)
            n_far += int((dd > 1e-3).sum())
            feat = feat_raw[ii]

            np.savez_compressed(str(out_dir / f'{token}.npz'), blind_adj=feat.astype(np.float32))
            adj_sum += float(feat.sum()); adj_n += len(feat)

            if (i + 1) % 500 == 0:
                print(f'  [{split}] {i + 1}/{len(files)}  mean_adj={adj_sum / max(adj_n, 1):.4f}  '
                      f'far_NN(>1e-3m)={n_far}')

        if len(files):
            print(f'[{split}] 完成 {len(files) - n_miss}/{len(files)}，缺 pcd={n_miss}，'
                  f'远邻点(>1e-3m)={n_far}，盲区邻接均值={adj_sum / max(adj_n, 1):.4f}')
    print('[全部完成]')


if __name__ == '__main__':
    main()
