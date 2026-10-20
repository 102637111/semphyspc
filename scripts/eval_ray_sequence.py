#!/usr/bin/env python3
"""方案 A 最小可行性实验：扫描线射线序列 Mamba vs 逐射线独立 MLP。

只验证一个假设：Mamba 隐状态沿扫描线（ring 顺序）能学到比逐射线独立 MLP 更强的
遮挡连续性——即给定被「丢失/遮挡」的射线，能否从相邻射线推断其结构。

任务（自监督，无需语义/多帧）：
  把一次 LiDAR 扫描构建成 range image（32 ring × W 方位角，每格 = 一条射线）。
  随机丢弃 p 比例射线（模拟稀疏/遮挡），模型要在观测到的剩余射线上，预测被丢
  射线的 (1) 近距障碍标志（0<range<R_thresh）与 (2) 归一化深度。

对照：
  - 基线 MLPRay：每条射线独立映射 [range, intensity, drop_flag, ring]，无跨射线上下文。
  - 提出 MambaRay：按 ring 顺序序列化 → SSM 状态传递 → 逐射线预测头。

评估（不用 PSNR/LPIPS）：被丢射线的障碍 recall/precision/F1 + 深度 MAE + 端到端 P95 延迟。

用法：
  python3 scripts/eval_ray_sequence.py --data_root /mnt/data/nuscenes/samples/LIDAR_TOP \
      --out_dir experiments/ray_sequence --n_train 200 --n_val 40
"""
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_RINGS = 32
W_AZIM = 1084          # nuScenes LIDAR_TOP 每环固定 1084 个方位角（已实测）
R_THRESH = 30.0        # 近距障碍阈值（米）
R_SCALE = 100.0        # 深度归一化上限


def load_sweep_range_image(path, w=W_AZIM):
    """读 .pcd.bin，按 (ring, 方位角) 排序成 range image (32, w) 的 (range, intensity)。"""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    xyz, intensity, ring = pts[:, :3], pts[:, 3], pts[:, 4].astype(np.int64)
    rng = np.linalg.norm(xyz, axis=1)
    az = np.arctan2(xyz[:, 1], xyz[:, 0])          # -pi..pi
    R = np.zeros((N_RINGS, w), dtype=np.float32)
    I = np.zeros((N_RINGS, w), dtype=np.float32)
    for r in range(N_RINGS):
        m = ring == r
        if not m.any():
            continue
        rr, ii, aa = rng[m], intensity[m], az[m]
        order = np.argsort(aa)                       # 按方位角升序 = 扫描线顺序
        n = order.shape[0]
        nw = min(n, w)
        R[r, :nw] = rr[order][:nw]
        I[r, :nw] = ii[order][:nw]
    return R, I


class RayDataset(Dataset):
    """在内存中缓存 N 个 sweep 的 (range, intensity)；drop 在 __getitem__ 在线做。"""

    def __init__(self, R, I, drop_p, r_thresh, r_scale, seed, fixed=False):
        self.R = R                                    # (N,32,W) 米
        self.I = I
        self.drop_p = drop_p
        self.r_thresh = r_thresh
        self.r_scale = r_scale
        self.rng = np.random.default_rng(seed)
        # fixed=True（验证集）：预计算并固定 drop 掩码，保证不同模型在【同一组】被丢射线上评测。
        if fixed:
            g = np.random.default_rng(seed)
            self.fixed_masks = (g.random(R.shape) < drop_p).astype(np.float32)

    def __len__(self):
        return self.R.shape[0]

    def __getitem__(self, idx):
        rn = self.R[idx] / self.r_scale               # (32,W) [0,1]
        in_ = self.I[idx] / 255.0
        if getattr(self, 'fixed_masks', None) is not None:
            mask = self.fixed_masks[idx]
        else:
            mask = (self.rng.random(rn.shape) < self.drop_p).astype(np.float32)
        rn_obs = rn * (1 - mask)
        in_obs = in_ * (1 - mask)
        y_flag = ((self.R[idx] > 0.1) & (self.R[idx] < self.r_thresh)).astype(np.float32)
        # 输入通道 [range_obs, intensity_obs, drop_flag] 拼成 (3,32,W)
        x = np.stack([rn_obs, in_obs, mask], axis=0).astype(np.float32)
        return (torch.from_numpy(x),                # (3,32,W)
                torch.from_numpy(y_flag),           # (32,W)
                torch.from_numpy(rn),               # (32,W) 深度目标
                torch.from_numpy(mask))             # (32,W) drop 位置


class MLPRay(nn.Module):
    """逐射线独立 MLP：每条射线 [range, intensity, drop_flag, ring_norm] → flag logit + range。"""

    def __init__(self, hid=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )
        self.flag_head = nn.Linear(hid, 1)
        self.range_head = nn.Linear(hid, 1)

    def forward(self, x):
        # x: (B,3,32,W) -> 每条射线 4 维（+ring_norm）
        B = x.shape[0]
        ring = torch.linspace(0, 1, N_RINGS, device=x.device)[None, :, None].expand(B, N_RINGS, W_AZIM)
        feat = torch.stack([x[:, 0], x[:, 1], x[:, 2], ring], dim=1)  # (B,4,32,W)
        feat = feat.permute(0, 2, 3, 1).reshape(-1, 4)               # (B*32*W, 4)
        h = self.mlp(feat)
        flag = self.flag_head(h).reshape(B, N_RINGS, W_AZIM)
        rng = torch.sigmoid(self.range_head(h)).reshape(B, N_RINGS, W_AZIM)
        return flag, rng


class MambaRay(nn.Module):
    """扫描线射线序列 Mamba：ring-major 序列化 -> SSM -> 逐射线预测头。"""

    def __init__(self, d_model=128, n_layers=2, d_state=16):
        super().__init__()
        from mamba_ssm import Mamba
        self.d_model = d_model
        self.embed = nn.Linear(3, d_model)
        self.ring_emb = nn.Embedding(N_RINGS, d_model)
        self.layers = nn.ModuleList(
            [Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
             for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.flag_head = nn.Linear(d_model, 1)
        self.range_head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B,3,32,W) -> (B, L=32*W, d_model)
        B = x.shape[0]
        seq = x.reshape(B, 3, -1).transpose(1, 2)          # (B, L, 3) ring-major
        rid = torch.arange(N_RINGS, device=x.device).repeat_interleave(W_AZIM)[None].expand(B, -1)
        h = self.embed(seq) + self.ring_emb(rid)            # (B, L, d_model)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        flag = self.flag_head(h).reshape(B, N_RINGS, W_AZIM)
        rng = torch.sigmoid(self.range_head(h)).reshape(B, N_RINGS, W_AZIM)
        return flag, rng


def run_epoch(model, loader, opt, device, train=True):
    model.train(train)
    tot = tot_flag = tot_rng = 0.0
    n = 0
    bce = nn.BCEWithLogitsLoss()
    for x, y_flag, rn, mask in loader:
        x, y_flag, rn, mask = (t.to(device) for t in (x, y_flag, rn, mask))
        flag_logit, rng_pred = model(x)
        flag_loss = bce(flag_logit, y_flag)
        range_loss = (mask * (rng_pred - rn).abs()).sum() / mask.sum().clamp(min=1)
        loss = flag_loss + 0.5 * range_loss
        if train:
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        tot += loss.item(); tot_flag += flag_loss.item(); tot_rng += range_loss.item()
        n += 1
    return tot / n, tot_flag / n, tot_rng / n


@torch.no_grad()
def evaluate(model, loader, device, r_scale=R_SCALE):
    """在被丢射线上算障碍 recall/precision/F1 + 深度 MAE（米）。"""
    model.eval()
    tp = fp = fn = 0
    range_err = 0.0
    range_n = 0
    for x, y_flag, rn, mask in loader:
        x, y_flag, rn, mask = (t.to(device) for t in (x, y_flag, rn, mask))
        flag_logit, rng_pred = model(x)
        pred_flag = (torch.sigmoid(flag_logit) > 0.5).float()
        d = mask.bool()
        tp += ((pred_flag[d] == 1) & (y_flag[d] == 1)).sum().item()
        fp += ((pred_flag[d] == 1) & (y_flag[d] == 0)).sum().item()
        fn += ((pred_flag[d] == 0) & (y_flag[d] == 1)).sum().item()
        range_err += (mask * (rng_pred - rn).abs() * r_scale).sum().item()
        range_n += mask.sum().item()
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * rec * prec / max(rec + prec, 1e-9)
    mae = range_err / max(range_n, 1)
    return {'recall': rec, 'precision': prec, 'f1': f1, 'range_mae_m': mae}


def measure_latency(model, device, iters=50, warmup=10):
    """单 sweep 端到端前向延迟 P95（毫秒）。"""
    model.eval()
    x = torch.zeros(1, 3, N_RINGS, W_AZIM, device=device)
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(ts, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='/mnt/data/nuscenes/samples/LIDAR_TOP')
    ap.add_argument('--out_dir', default='experiments/ray_sequence')
    ap.add_argument('--n_train', type=int, default=200)
    ap.add_argument('--n_val', type=int, default=40)
    ap.add_argument('--drop_p', type=float, default=0.3)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--n_layers', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    files = sorted(os.path.join(args.data_root, f) for f in os.listdir(args.data_root)
                   if f.endswith('.pcd.bin'))
    # 场景隔离划分：按场景名（文件名首段）分组，train/val 不共享场景，避免同场景邻帧泄漏。
    from collections import OrderedDict
    by_scene = OrderedDict()
    for fp in files:
        by_scene.setdefault(os.path.basename(fp).split('__')[0], []).append(fp)
    scenes = list(by_scene.keys())
    train_files, val_files = [], []
    for sc in scenes:
        if len(train_files) < args.n_train:
            train_files.extend(by_scene[sc])
        elif len(val_files) < args.n_val:
            val_files.extend(by_scene[sc])
    train_files = train_files[:args.n_train]
    val_files = val_files[:args.n_val]
    if len(train_files) < args.n_train or len(val_files) < args.n_val:
        raise SystemExit(f'数据不足：train {len(train_files)} / val {len(val_files)}')

    print(f'[数据] 构建 range image（场景隔离）：train {len(train_files)} / val {len(val_files)}')
    def build(flist):
        Rs, Is = [], []
        for fp in flist:
            R, I = load_sweep_range_image(fp)
            Rs.append(R); Is.append(I)
        return np.stack(Rs), np.stack(Is)
    Rtr, Itr = build(train_files)
    Rva, Iva = build(val_files)

    tr = DataLoader(RayDataset(Rtr, Itr, args.drop_p, R_THRESH, R_SCALE, args.seed),
                    batch_size=args.bs, shuffle=True, num_workers=0)
    # 验证集 fixed=True：drop 掩码预计算并固定，MLP 与 Mamba 在【同一组】被丢射线上对比。
    va = DataLoader(RayDataset(Rva, Iva, args.drop_p, R_THRESH, R_SCALE, args.seed + 1, fixed=True),
                    batch_size=args.bs, shuffle=False, num_workers=0)

    summary = {'args': vars(args)}
    for name, model in [('MLPRay', MLPRay()), ('MambaRay', MambaRay(args.d_model, args.n_layers))]:
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        print(f'\n===== 训练 {name} =====')
        for ep in range(args.epochs):
            tl, tfl, trl = run_epoch(model, tr, opt, device, train=True)
            vl, vfl, vrl = run_epoch(model, va, opt, device, train=False)
            print(f'  ep{ep}: train_loss={tl:.4f} val_loss={vl:.4f} (flag {vfl:.4f} range {vrl:.4f})')
        met = evaluate(model, va, device)
        met['latency_p95_ms'] = measure_latency(model, device)
        summary[name] = met
        print(f'  [结果] {name}: 障碍recall={met["recall"]:.4f} precision={met["precision"]:.4f} '
              f'f1={met["f1"]:.4f} 深度MAE={met["range_mae_m"]:.3f}m P95={met["latency_p95_ms"]:.1f}ms')

    out_json = os.path.join(args.out_dir, 'ray_sequence_result.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[完成] 结果已存档 → {out_json}')


if __name__ == '__main__':
    main()
