#!/usr/bin/env python3
"""D⑧ 鲁棒性：量化 3D 分割模型对四类真实扰动的敏感度（审稿风险「鲁棒性」）。

动机：论文把 3D 逐点分割作为导航软碰撞决策的前置，审稿人必然问「模型在真实部署扰动下是否稳」。
本脚本对 nuScenes val 逐点注入四类扰动，各跑一遍全量分割，输出 mIoU 与关注类 IoU 的下降：

  1. clean         —— 无扰动（上界参照）
  2. rgb_zero      —— 颜色缺失：RGB 三通道全置 0（量化「RGB 早期融合」贡献丢失的代价）
  3. sparse_50     —— 稀疏点云：随机保留 50% 点 + 重复填充（量化「采样密度」鲁棒性）
  4. occlusion     —— 遮挡：移除前方 60° 扇形（x>0 且 |y|<x·tan30°）的点（量化「被遮挡」鲁棒性）
  5. calib_010     —— 标定扰动：xyz 加零均值高斯噪声 σ=0.10 m（量化「外参标定误差」鲁棒性）

输出每种扰动的 mIoU（全 12 类）、关注类 IoU（person/two_wheeler/frangible）、致命类安全聚合
（critical/severe/safe），以及相对 clean 的 ΔmIoU。

用法:
    python3 scripts/eval_robustness.py \
        --config cfgs/seg_16k_ng6144_g050_merged.yaml \
        --ckpts experiments/seg_16k_ng6144_g050_merged/cfgs/merged_pretrain_ft/20260927-070312/ckpt-best.pth \
        --bs 4 --perturbs clean,rgb_zero,sparse_50,occlusion,calib_010 \
        --out-json results/robustness_merged_pretrain_ft.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import builder
from utils.config import get_config
from utils.logger import get_root_logger, print_log

LIDARSEG_COARSE = [
    'noise', 'animal', 'person', 'car', 'bus', 'truck',
    'two_wheeler', 'ground', 'manmade', 'vegetation', 'frangible', 'ego',
]
RARE_CLASSES = {2, 6, 10}
LETHAL = [1, 2]
FRANGIBLE = [10]
DEFORMABLE = [9]
PASSABLE = [7]
COLLIDABLE = FRANGIBLE + DEFORMABLE

# 扰动定义（在 CPU tensor data[0] 上原地构造新 tensor，(B, N, 6) xyz+rgb）
PERTURBS = {}


def _reg(name):
    def deco(fn):
        PERTURBS[name] = fn
        return fn
    return deco


@_reg('clean')
def _clean(pts, seed):
    return pts.clone()


@_reg('rgb_zero')
def _rgb_zero(pts, seed):
    out = pts.clone()
    out[:, :, 3:] = 0.0
    return out


def _sparse(pts, ratio, seed):
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    B, N, C = pts.shape
    k = int(N * ratio)
    out = pts.clone()
    for b in range(B):
        idx = torch.randperm(N, generator=g)[:k]
        kept = pts[b, idx]                       # (k, C)
        rep = torch.randint(0, k, (N,), generator=g)
        out[b] = kept[rep]                       # 重复填充保持 N
    return out


@_reg('sparse_50')
def _sparse_50(pts, seed):
    return _sparse(pts, 0.5, seed)


@_reg('sparse_25')
def _sparse_25(pts, seed):
    return _sparse(pts, 0.25, seed)


@_reg('occlusion')
def _occlusion(pts, seed):
    out = pts.clone()
    x = out[:, :, 0]
    y = out[:, :, 1]
    mask = (x > 0) & (torch.abs(y) < x * 0.577)   # 前方 60° 扇形
    out[mask, :3] = 0.0
    return out


def _calib(pts, sigma, seed):
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    out = pts.clone()
    out[:, :, :3] += torch.randn(out[:, :, :3].shape, generator=g) * sigma
    return out


@_reg('calib_005')
def _calib_005(pts, seed):
    return _calib(pts, 0.05, seed)


@_reg('calib_010')
def _calib_010(pts, seed):
    return _calib(pts, 0.10, seed)


def conf_matrix(pred, sem, num_classes):
    valid = sem >= 0
    p = pred[valid].long()
    s = sem[valid].long()
    idx = s * num_classes + p
    return torch.bincount(idx, minlength=num_classes * num_classes).view(num_classes, num_classes)


def per_class_iou(conf, num_classes):
    ious = []
    for c in range(num_classes):
        gt = conf[c].sum().item()
        pred_sum = conf[:, c].sum().item()
        tp = conf[c, c].item()
        union = gt + pred_sum - tp
        ious.append(tp / union if union > 0 else float('nan'))
    return ious


def safety_agg(conf):
    cm = conf.cpu().numpy()
    lethal_gt = int(cm[LETHAL].sum())
    agg = {'lethal_gt_points': lethal_gt}
    if lethal_gt:
        n = float(lethal_gt)
        agg['critical_rate'] = round(int(cm[np.ix_(LETHAL, COLLIDABLE)].sum()) / n, 6)
        agg['severe_rate'] = round(int(cm[np.ix_(LETHAL, PASSABLE)].sum()) / n, 6)
        agg['safe_rate'] = round((int(cm[np.ix_(LETHAL, LETHAL)].sum()) +
                                  int(cm[LETHAL].sum() - cm[np.ix_(LETHAL, LETHAL)].sum()
                                      - cm[np.ix_(LETHAL, COLLIDABLE)].sum()
                                      - cm[np.ix_(LETHAL, PASSABLE)].sum()
                                      - cm[np.ix_(LETHAL, [0])].sum())) / n, 6)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpts', required=True)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--perturbs', type=str, default='clean,rgb_zero,sparse_50,occlusion,calib_010',
                    help='扰动类型列表（逗号分隔），可选：' + ','.join(PERTURBS.keys()))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--tag', default='d8_robustness', help='结果标签')
    ap.add_argument('--out-json', default=None)
    args = ap.parse_args()

    perturb_names = [p.strip() for p in args.perturbs.split(',') if p.strip()]
    for p in perturb_names:
        assert p in PERTURBS, f'未知扰动 {p}，可选：{sorted(PERTURBS.keys())}'

    args.local_rank = 0
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.resume = False
    args.experiment_path = '/tmp/eval_robustness_exp'
    os.makedirs(args.experiment_path, exist_ok=True)
    args.log_name = os.path.basename(args.config).replace('.yaml', '')
    os.environ['LOCAL_RANK'] = '0'
    logger = get_root_logger(name='eval_robustness')

    config = get_config(args, logger=logger)
    config.dataset.val.others.bs = args.bs
    num_classes = config.model.num_classes
    assert num_classes == 12, f'本脚本只用于 12 类体系，当前 num_classes={num_classes}'

    print_log(f'加载 checkpoint: {args.ckpts}', logger=logger)
    _, val_loader = builder.dataset_builder(args, config.dataset.val)

    model = builder.model_builder(config.model)
    builder.load_model(model, args.ckpts, logger=logger)
    model = nn.DataParallel(model).cuda()
    model.eval()

    results = {}
    clean_miou = None

    for pname in perturb_names:
        conf = torch.zeros(num_classes, num_classes, dtype=torch.int64).cuda()
        with torch.no_grad():
            for i, (_, stems, data) in enumerate(val_loader):
                if args.max_frames and i >= args.max_frames:
                    break
                pts_cpu = PERTURBS[pname](data[0], args.seed)
                pts = pts_cpu.cuda()
                sem = data[1].cuda()
                ret = model(pts)
                logits = ret[0] if isinstance(ret, (tuple, list)) else ret
                pred = logits.argmax(-1)
                conf += conf_matrix(pred, sem, num_classes)

        iou = per_class_iou(conf, num_classes)
        v = [x for x in iou if not np.isnan(x)]
        miou = float(np.mean(v)) if v else 0.0
        if pname == 'clean':
            clean_miou = miou
        d = round(miou - clean_miou, 4) if clean_miou is not None else None
        results[pname] = {
            'miou_all': round(miou, 4),
            'delta_miou_vs_clean': d,
            'per_class_iou': {LIDARSEG_COARSE[c]: (None if np.isnan(iou[c]) else round(float(iou[c]), 4))
                              for c in sorted(RARE_CLASSES)},
            'safety': safety_agg(conf),
        }
        print_log(f'[{pname}] mIoU={miou:.4f}  Δvs_clean={d}', logger=logger)

    # 汇总表
    print_log('\n' + '=' * 88, logger=logger)
    print_log('D⑧ 鲁棒性：扰动 → mIoU / Δ / 关注类 IoU / 安全聚合', logger=logger)
    print_log('=' * 88, logger=logger)
    hdr = f'{"扰动":<14} {"mIoU":>8} {"Δ":>8} {"person":>8} {"two_wheel":>10} {"frangible":>10} {"safe%":>8}'
    print_log(hdr, logger=logger)
    print_log('-' * 88, logger=logger)
    for pname in perturb_names:
        r = results[pname]
        pci = r['per_class_iou']
        safe = r['safety'].get('safe_rate')
        safe_s = f'{safe * 100:.2f}%' if safe is not None else '—'
        print_log(f'{pname:<14} {r["miou_all"]:>8.4f} {str(r["delta_miou_vs_clean"]):>8} '
                  f'{str(pci.get("person")):>8} {str(pci.get("two_wheeler")):>10} '
                  f'{str(pci.get("frangible")):>10} {safe_s:>8}', logger=logger)
    print_log('=' * 88, logger=logger)

    if args.out_json:
        rec = {
            'tag': args.tag,
            'config': args.config,
            'ckpts': args.ckpts,
            'seed': args.seed,
            'perturbs': perturb_names,
            'num_classes': num_classes,
            'results': results,
        }
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        print_log(f'结果已写入 {args.out_json}', logger=logger)


if __name__ == '__main__':
    main()
