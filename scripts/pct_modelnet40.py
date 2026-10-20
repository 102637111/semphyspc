#!/usr/bin/env python3
"""PCT（Point Cloud Transformer）在 ModelNet40 上的分类实验。

目的：用纯 PyTorch（不依赖 spconv/torchsparse/MinkowskiEngine 等稀疏卷积库）的
transformer 类点云模型，验证「transformer 点云识别」在本机（RTX 5070 Ti, torch 2.12+cu130）
可跑、且可作为本科生课程实验（纯 PyTorch ⇒ 原生支持 Windows/CPU/单卡）。

模型：PCT 的核心是「对点做 self-attention + 全局 max/mean 池化分类」，
      与 PointNet++ 的层级 MLP 相比是标准的 transformer 范式。

数据：data/ModelNet/modelnet40_normal_resampled（每类一个目录，每个 .txt 一个物体，
      首行点数字头 + 每行 "x y z nx ny nz"，只取 xyz）。按类 80/20 划分 train/test。
"""
import os
import glob
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0
NUM_POINTS = 1024
DATA_ROOT = "data/ModelNet/modelnet40_normal_resampled"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ---------- 数据 ----------
def load_off_txt(path, num_points=NUM_POINTS):
    """读取一个 ModelNet40 点云 .txt（首行可能为点数字头），返回 (num_points, 3)。"""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    start = 0
    # 首行若只有一个整数（点数字头）则跳过；文件可能为逗号或空白分隔
    toks0 = lines[0].replace(',', ' ').split()
    if len(toks0) == 1:
        start = 1
    pts = []
    for ln in lines[start:]:
        v = ln.replace(',', ' ').split()
        pts.append([float(v[0]), float(v[1]), float(v[2])])
    pts = np.array(pts, dtype=np.float32)
    if len(pts) < num_points:
        # 不足则重复采样补齐
        idx = np.random.choice(len(pts), num_points, replace=True)
    else:
        idx = np.random.choice(len(pts), num_points, replace=False)
    pts = pts[idx]
    # 中心化 + 归一化到单位球（标准做法）
    pts -= pts.mean(axis=0)
    d = np.linalg.norm(pts, axis=1).max() + 1e-8
    pts /= d
    return pts


def build_split():
    """按类排序后 80/20 划分 train/test，返回 (train_files, test_files) 各 (path, label)。"""
    classes = sorted(d for d in os.listdir(DATA_ROOT)
                     if os.path.isdir(os.path.join(DATA_ROOT, d)))
    cls2id = {c: i for i, c in enumerate(classes)}
    train, test = [], []
    for c in classes:
        files = sorted(glob.glob(os.path.join(DATA_ROOT, c, "*.txt")))
        n_train = int(round(len(files) * 0.8))
        for p in files[:n_train]:
            train.append((p, cls2id[c]))
        for p in files[n_train:]:
            test.append((p, cls2id[c]))
    return classes, train, test


def augment(pts):
    """随机绕 z 轴旋转 + 缩放 + 抖动（ModelNet40 达到 92%+ 的标准增广）。"""
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    pts = pts @ R.T
    pts = pts * np.random.uniform(0.8, 1.25)
    pts = pts + np.random.normal(0, 0.02, size=pts.shape).astype(np.float32)
    return pts


# ---------- 模型（PCT，纯 PyTorch） ----------
class SA(nn.Module):
    """单层多头 self-attention（PCT 用普通 attention；论文原版是 offset attention，此处简化）。"""
    def __init__(self, d, heads=4):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d)

    def forward(self, x):  # (B, N, d)
        B, N, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.h, self.dk).transpose(1, 2)
        k = k.view(B, N, self.h, self.dk).transpose(1, 2)
        v = v.view(B, N, self.h, self.dk).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        attn = attn.softmax(dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.sa = SA(d, heads)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        x = x + self.sa(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class PCT(nn.Module):
    """PCT 分类器：embedding → 4 层 self-attention → max+mean 池化 → 分类头。"""
    def __init__(self, num_classes=40, d=128, depth=4, heads=4):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.cls = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Dropout(0.4),
                                 nn.Linear(256, num_classes))

    def forward(self, x):  # (B, N, 3)
        x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        g = torch.cat([x.max(dim=1)[0], x.mean(dim=1)], dim=-1)
        return self.cls(g)


# ---------- 训练 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt", type=str, default="results/pct_modelnet40_best.pth")
    args = ap.parse_args()

    set_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes, train_files, test_files = build_split()
    print(f"[data] {len(classes)} 类 | train {len(train_files)} | test {len(test_files)} | dev={dev}")

    # 一次性把全部点云读进内存并缓存为 npz（12k 个物体 ×1024×3 ≈ 150 MB），
    # 避免每个 epoch 重复读 12k 个 .txt（纯 Python 解析 ~10 万行/epoch 是主要开销）。
    def preload(files, cache):
        if os.path.exists(cache):
            d = np.load(cache)
            return d["pts"], d["lbl"]
        n = len(files)
        pts = np.zeros((n, NUM_POINTS, 3), dtype=np.float32)
        lbl = np.zeros((n,), dtype=np.int64)
        for i, (p, l) in enumerate(files):
            pts[i] = load_off_txt(p)
            lbl[i] = l
        np.savez_compressed(cache, pts=pts, lbl=lbl)
        return pts, lbl

    tr_pts, tr_lbl = preload(train_files, os.path.join(DATA_ROOT, "_cache_train.npz"))
    te_pts, te_lbl = preload(test_files, os.path.join(DATA_ROOT, "_cache_test.npz"))

    def batch_of(pts, lbl, aug=True):
        B = args.batch
        N = len(pts)
        idx = np.arange(N)
        if aug:
            np.random.shuffle(idx)
        for i in range(0, N, B):
            sub = idx[i:i + B]
            xb = pts[sub].copy()
            if aug:
                for k in range(len(xb)):
                    xb[k] = augment(xb[k])
            yield torch.from_numpy(xb).to(dev), torch.from_numpy(lbl[sub]).to(dev)

    model = PCT(num_classes=len(classes)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()

    best = 0.0
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, cnt, run = 0.0, 0, 0
        for x, y in batch_of(tr_pts, tr_lbl, aug=True):
            opt.zero_grad()
            out = model(x)
            loss = lossf(out, y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y); cnt += len(y); run += 1
        sched.step()
        # 每 10 轮（含最后一轮）在 test 上评一次
        if ep % 10 == 0 or ep == args.epochs:
            model.eval()
            corr = 0; n = 0
            with torch.no_grad():
                for x, y in batch_of(te_pts, te_lbl, aug=False):
                    out = model(x)
                    corr += (out.argmax(1) == y).sum().item(); n += len(y)
            acc = corr / n
            print(f"epoch {ep:3d}  train_loss {tot/cnt:.4f}  test_acc {acc:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {torch.cuda.memory_allocated()/1e9:.2f}GiB")
            if acc > best:
                best = acc
                torch.save({"model": model.state_dict(), "classes": classes, "acc": best}, args.ckpt)
    print(f"DONE best_test_acc={best:.4f}  ckpt={args.ckpt}")


if __name__ == "__main__":
    main()
