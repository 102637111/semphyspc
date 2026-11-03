#!/usr/bin/env python3
"""方案 A 真实多帧盲区 GT 实验：射线序列 Mamba 预测真实盲区占用 + 深度。

与 eval_ray_sequence.py（随机 dropout 自监督）的关键区别：
  - 空射线 = 当前帧无回波；用未来关键帧 + ego-pose 对齐，把「ego 移动后显露」的表面
    变换回当前帧坐标系，落在空射线上的点即【真实盲区表面】→ 盲区占用(flag) + 深度(depth) 监督。
  - 模型输入 [观测 range, 观测 intensity, 空射线掩码]；仅在空射线上评测 flag/depth。

对照：MLPRay(逐射线独立) / MambaRay(单向) / BiMambaRay(双向)。

用法：
  python3 scripts/eval_ray_blindspot.py --n_train 200 --n_val 40 --future_k 1
  python3 scripts/eval_ray_blindspot.py --n_train 200 --n_val 40 --future_k 1 --probe   # 只看 GT 统计不训练
"""
import os, sys, time, json, pickle, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ray_sequence import (MLPRay, MambaRay, BiMambaRay,
                               N_RINGS, W_AZIM, R_THRESH, R_SCALE)

DATA_ROOT = '/mnt/data/nuscenes'
LIDAR_DIR = os.path.join(DATA_ROOT, 'samples', 'LIDAR_TOP')
META_PATH = 'experiments/ray_blindspot/blindspot_meta.pkl'


def quat_rotate(q, v):
    w, x, y, z = q
    qv = np.stack([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], np.float64)


def lidar_to_lidar(pts, t_cs, q_cs, t_ego_f, q_ego_f, t_ego_a, q_ego_a):
    """未来帧 LIDAR 系 → 当前帧 LIDAR 系。"""
    p = quat_rotate(q_cs, pts) + t_cs                    # LIDAR_f -> ego_f
    p = quat_rotate(q_ego_f, p) + t_ego_f                # ego_f -> world
    p = quat_rotate(quat_conj(q_ego_a), p - t_ego_a)     # world -> ego_a
    p = quat_rotate(quat_conj(q_cs), p - t_cs)           # ego_a -> LIDAR_a
    return p


def rasterize(pts, w=W_AZIM):
    """几何正确栅格化：bin=round((az+π)/2π·w)，每 (ring,bin) 取最近表面（min range）。
    返回 (range, intensity)。注意 numpy 2.4 的 np.minimum.at(tuple 索引) 已坏，用 lexsort 取每组 min。"""
    xyz = pts[:, :3]
    rng = np.linalg.norm(xyz, axis=1)
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    ring = pts[:, 4].astype(np.int64)
    intensity = pts[:, 3]
    valid = (ring >= 0) & (ring < N_RINGS) & (rng > 0)
    ring, rng, az, intensity = ring[valid], rng[valid], az[valid], intensity[valid]
    b = ((az + np.pi) / (2 * np.pi) * w).astype(np.int64) % w
    idx = ring * w + b
    order = np.lexsort((rng, idx))                 # 主键 idx、次键 rng → 同格内 range 升序
    idx_s = idx[order]; rng_s = rng[order]; int_s = intensity[order]
    first = np.concatenate(([True], idx_s[1:] != idx_s[:-1]))
    R = np.zeros(N_RINGS * w, np.float32)
    I = np.zeros(N_RINGS * w, np.float32)
    R[idx_s[first]] = rng_s[first]
    I[idx_s[first]] = int_s[first]
    return R.reshape(N_RINGS, w), I.reshape(N_RINGS, w)


def load_pcd(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)


def build_split(pair_idxs, pairs, keyframes, ego_trans, ego_rot, t_cs, q_cs):
    """把若干 (anchor, future) 帧对栅格化成训练数组。"""
    Xs, Fs, Ds, Es, Rs = [], [], [], [], []
    for j in pair_idxs:
        ia, ifu = pairs[j]
        Ra, Ia = rasterize(load_pcd(os.path.join(LIDAR_DIR, keyframes[ia]['basename'])))
        empty = (Ra == 0).astype(np.float32)
        x = np.stack([Ra / R_SCALE * (1 - empty), Ia / 255.0 * (1 - empty), empty], axis=0)
        pf = load_pcd(os.path.join(LIDAR_DIR, keyframes[ifu]['basename']))
        xyz_f = lidar_to_lidar(pf[:, :3], t_cs, q_cs,
                               ego_trans[ifu], ego_rot[ifu], ego_trans[ia], ego_rot[ia])
        Rf, _ = rasterize(np.concatenate([xyz_f, pf[:, 3:]], axis=1))
        Rf = np.minimum(Rf, R_SCALE)          # 深度裁剪到 100m 上限，保证 sigmoid 目标有界
        reveal = (empty * (Rf > 0)).astype(np.float32)                 # 空射线里被未来帧显露
        flag = (empty * (Rf > 0.1) * (Rf < R_THRESH)).astype(np.float32)  # 隐藏近距障碍
        depth = (Rf / R_SCALE) * reveal                                # 显露表面深度(归一化)
        Xs.append(x); Fs.append(flag); Ds.append(depth); Es.append(empty); Rs.append(reveal)
    return (np.stack(Xs), np.stack(Fs), np.stack(Ds), np.stack(Es), np.stack(Rs))


class BlindSpotDataset(Dataset):
    def __init__(self, X, flag, depth, empty, reveal):
        self.X, self.flag, self.depth = X, flag, depth
        self.empty, self.reveal = empty, reveal

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i]), torch.from_numpy(self.flag[i]),
                torch.from_numpy(self.depth[i]), torch.from_numpy(self.empty[i]),
                torch.from_numpy(self.reveal[i]))


def run_epoch(model, loader, opt, device, train=True):
    model.train(train)
    tot = tot_flag = tot_rng = 0.0
    n = 0
    bce = nn.BCEWithLogitsLoss(reduction='none')
    for x, flag_gt, depth_gt, empty, reveal in loader:
        x, flag_gt, depth_gt, empty, reveal = (t.to(device) for t in
                                               (x, flag_gt, depth_gt, empty, reveal))
        flag_logit, rng_pred = model(x)
        flag_loss = (bce(flag_logit, flag_gt) * empty).sum() / empty.sum().clamp(min=1)
        range_loss = (reveal * (rng_pred - depth_gt).abs()).sum() / reveal.sum().clamp(min=1)
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
    model.eval()
    tp = fp = fn = 0
    range_err = 0.0
    range_n = 0
    for x, flag_gt, depth_gt, empty, reveal in loader:
        x, flag_gt, depth_gt, empty, reveal = (t.to(device) for t in
                                               (x, flag_gt, depth_gt, empty, reveal))
        flag_logit, rng_pred = model(x)
        pred_flag = (torch.sigmoid(flag_logit) > 0.5).float()
        e = empty.bool()
        tp += ((pred_flag[e] == 1) & (flag_gt[e] == 1)).sum().item()
        fp += ((pred_flag[e] == 1) & (flag_gt[e] == 0)).sum().item()
        fn += ((pred_flag[e] == 0) & (flag_gt[e] == 1)).sum().item()
        range_err += (reveal * (rng_pred - depth_gt).abs() * r_scale).sum().item()
        range_n += reveal.sum().item()
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * rec * prec / max(rec + prec, 1e-9)
    mae = range_err / max(range_n, 1)
    return {'recall': rec, 'precision': prec, 'f1': f1, 'depth_mae_m': mae}


@torch.no_grad()
def breakdown(model, X, flag_gt, depth_gt, empty, reveal, device, r_scale=R_SCALE):
    """按盲区类型拆解误差：近/中/远深度 + 遮挡邻接/开阔。返回各类的 flag 与深度 MAE。"""
    import torch.nn.functional as F
    model.eval()
    N = X.shape[0]
    flag_all = torch.zeros((N, N_RINGS, W_AZIM))
    rng_all = torch.zeros((N, N_RINGS, W_AZIM))
    bs = 16
    for i in range(0, N, bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        fl, rp = model(xb)
        flag_all[i:i + bs] = torch.sigmoid(fl).cpu()
        rng_all[i:i + bs] = rp.cpu()
    pred_flag = (flag_all > 0.5).float()
    obs_range = torch.from_numpy(X[:, 0] * r_scale)          # 观测 range（空射线=0）
    near_obs = ((obs_range > 0) & (obs_range < 10.0)).float()
    has_near_nb = F.max_pool2d(near_obs.unsqueeze(1), 3, 1, 1).squeeze(1)  # 3x3 邻域内有近距观测
    reveal_t = torch.from_numpy(reveal).bool()
    flag_t = torch.from_numpy(flag_gt).bool()
    depth_t = torch.from_numpy(depth_gt)
    d_m = depth_t * r_scale
    cats = {
        'near(<10m)':   reveal_t & (d_m < 10.0),
        'mid(10-30m)':  reveal_t & (d_m >= 10.0) & (d_m < 30.0),
        'far(>=30m)':   reveal_t & (d_m >= 30.0),
        'occl_adj':     reveal_t & (has_near_nb > 0),
        'open(sky)':    reveal_t & (has_near_nb == 0),
    }
    out = {}
    for cname, m in cats.items():
        m = m.bool()
        if m.sum() == 0:
            out[cname] = {'n': 0}
            continue
        tp = ((pred_flag[m] == 1) & (flag_t[m] == 1)).sum().item()
        fp = ((pred_flag[m] == 1) & (flag_t[m] == 0)).sum().item()
        fn = ((pred_flag[m] == 0) & (flag_t[m] == 1)).sum().item()
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * rec * prec / max(rec + prec, 1e-9)
        mae = (rng_all[m] - depth_t[m]).abs().mean().item() * r_scale
        out[cname] = {'n': int(m.sum().item()), 'recall': rec, 'precision': prec,
                      'f1': f1, 'depth_mae_m': mae}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta_path', default=META_PATH)
    ap.add_argument('--out_dir', default='experiments/ray_blindspot')
    ap.add_argument('--n_train', type=int, default=200)
    ap.add_argument('--n_val', type=int, default=40)
    ap.add_argument('--future_k', type=int, default=1, help='未来帧偏移(关键帧步数)')
    ap.add_argument('--drop_p', type=float, default=0.0, help='（可选）在观测上额外随机丢射线，模拟更稀疏')
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--n_layers', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_seeds', type=int, default=1, help='多种子训练，聚合 mean±std')
    ap.add_argument('--probe', action='store_true', help='只打印 GT 统计，不训练')
    ap.add_argument('--breakdown', action='store_true', help='末种子后按盲区类型拆解误差')
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

    # 连续帧对（同场景内，间隔 future_k 个关键帧）
    pairs = []
    for i in range(len(keyframes) - args.future_k):
        if keyframes[i]['scene'] == keyframes[i + args.future_k]['scene']:
            pairs.append((i, i + args.future_k))
    print(f'[帧对] 共 {len(pairs)} 对（future_k={args.future_k}）')

    # 场景隔离划分
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

    if args.probe:
        for tag, (X, F, D, E, R) in [('train', (Xtr, Ftr, Dtr, Etr, Rtr)),
                                     ('val', (Xva, Fva, Dva, Eva, Rva))]:
            n_ray = E.sum()
            n_reveal = R.sum()
            n_flag = F.sum()
            depths = D[D > 0] * R_SCALE
            print(f'[{tag}] 空射线占比={n_ray / E.size:.3f}  显露(空射线里)={n_reveal / n_ray:.3f}  '
                  f'隐藏近距障碍(空射线里)={n_flag / n_ray:.4f}')
            if depths.size:
                print(f'        显露深度(米): mean={depths.mean():.2f} median={np.median(depths):.2f} '
                      f'p90={np.percentile(depths, 90):.2f} max={depths.max():.2f}')
        return

    model_fns = [('MLPRay', lambda: MLPRay()),
                 ('MambaRay', lambda: MambaRay(args.d_model, args.n_layers)),
                 ('BiMambaRay', lambda: BiMambaRay(args.d_model, args.n_layers))]
    per_model = {name: [] for name, _ in model_fns}
    per_model_bd = {}

    for s in range(args.n_seeds):
        torch.manual_seed(args.seed + s)
        np.random.seed(args.seed + s)
        tr = DataLoader(BlindSpotDataset(Xtr, Ftr, Dtr, Etr, Rtr),
                        batch_size=args.bs, shuffle=True, num_workers=0)
        va = DataLoader(BlindSpotDataset(Xva, Fva, Dva, Eva, Rva),
                        batch_size=args.bs, shuffle=False, num_workers=0)
        for name, mf in model_fns:
            model = mf().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
            print(f'\n===== 训练 {name} seed={args.seed + s} =====')
            for ep in range(args.epochs):
                tl, tfl, trl = run_epoch(model, tr, opt, device, train=True)
                vl, vfl, vrl = run_epoch(model, va, opt, device, train=False)
                print(f'  ep{ep}: train_loss={tl:.4f} val_loss={vl:.4f} (flag {vfl:.4f} depth {vrl:.4f})')
            met = evaluate(model, va, device)
            per_model[name].append(met)
            print(f'  [结果] {name} seed={args.seed + s}: recall={met["recall"]:.4f} '
                  f'precision={met["precision"]:.4f} f1={met["f1"]:.4f} 深度MAE={met["depth_mae_m"]:.3f}m')
            if args.breakdown and s == args.n_seeds - 1:
                bd = breakdown(model, Xva, Fva, Dva, Eva, Rva, device)
                per_model_bd[name] = bd
                print(f'    [拆解] {name}:')
                for c, mm in bd.items():
                    if mm.get('n', 0) == 0:
                        print(f'      {c}: n=0')
                    else:
                        print(f'      {c}: n={mm["n"]:>7}  flag_rec={mm["recall"]:.3f} '
                              f'flag_prec={mm["precision"]:.3f} flag_f1={mm["f1"]:.3f}  深度MAE={mm["depth_mae_m"]:.2f}m')
            del model, opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {'args': vars(args), 'per_seed': per_model, 'mean_std': {}, 'breakdown': per_model_bd}
    print(f'\n===== 汇总（mean±std over {args.n_seeds} seeds）=====')
    for name, mets in per_model.items():
        summary['mean_std'][name] = {}
        line = f'  {name}: '
        for k in ['recall', 'precision', 'f1', 'depth_mae_m']:
            vals = np.array([m[k] for m in mets])
            summary['mean_std'][name][k] = {'mean': float(vals.mean()), 'std': float(vals.std())}
            line += f'{k}={vals.mean():.4f}±{vals.std():.4f}  '
        print(line)

    out_json = os.path.join(args.out_dir, 'ray_blindspot_result.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[完成] 结果已存档 → {out_json}')


if __name__ == '__main__':
    main()
