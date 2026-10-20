#!/usr/bin/env python3
"""S3 崩塌定位：在**同一批 dev 图像**上跑三个模型，唯一变量是「是否过 surgery / 是否训练」。

  A. PekingU/rtdetr_r50vd 原样（80 类，零样本）      —— 上界参照
  B. 同一权重过 surgery()（82 类，未训练，新类随机） —— 隔离 surgery 的结构性副作用
  C. experiments/s3_rtdetr_finetune/hf_best（82 类，训过 1 轮）

三者走的是训练脚本里**同一个** evaluate_recall()，所以任何差异都只能来自模型侧。

用法：
    python3 scripts/diag_s3_eval.py --out_dir experiments/s3_rtdetr_finetune --n_dev 24
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finetune_rtdetr_nuscenes as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default='data/s3_det_labels_train.npz')
    ap.add_argument('--out_dir', default='experiments/s3_rtdetr_finetune')
    ap.add_argument('--model', default='PekingU/rtdetr_r50vd')
    ap.add_argument('--n_dev', type=int, default=24)
    ap.add_argument('--dev_frac', type=float, default=0.05)
    ap.add_argument('--dev_max_items', type=int, default=600)
    ap.add_argument('--score_thr', type=float, default=0.35)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(os.path.join(args.out_dir, 'token_to_images.json')) as f:
        tok2img = json.load(f)

    items, blist, token_of = F.build_items(args.labels, tok2img)
    print(f'[diag] 有标注图像 {len(items):,}')

    # 复刻训练脚本的 dev 划分（同 seed、同顺序），保证是同一批图
    utok = sorted(set(token_of))
    rng = np.random.default_rng(0)
    rng.shuffle(utok)
    n_dev = max(1, int(len(utok) * args.dev_frac))
    dev_tok = set(utok[:n_dev])
    dv_idx = [i for i, t in enumerate(token_of) if t in dev_tok]
    if args.dev_max_items and len(dv_idx) > args.dev_max_items:
        dv_idx = list(rng.choice(dv_idx, size=args.dev_max_items, replace=False))
    dv_idx = dv_idx[:args.n_dev]
    dv_items = [items[i] for i in dv_idx]
    dv_boxes = [blist[i] for i in dv_idx]
    print(f'[diag] dev 取前 {len(dv_idx)} 张图：{os.path.basename(dv_items[0][2])} …')

    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    proc = RTDetrImageProcessor.from_pretrained(args.model)

    def run(tag, model):
        model.to(dev).eval()
        rows = F.evaluate_recall(model, proc, dv_items, dv_boxes, dev,
                                 score_thr=args.score_thr, max_items=0)
        rows = sorted(rows, key=lambda r: r['cls'])
        gt = sum(r['gt'] for r in rows)
        tp = sum(r['tp'] for r in rows)
        print(f'\n===== {tag} =====   总 GT {gt}  总 TP {tp}')
        print(f"{'cls':>4} {'类别':<13}{'GT':>6}{'TP':>6}{'FP':>6}{'召回':>9}{'精度':>9}")
        for r in rows:
            print(f"{r['cls']:>4} {r['name']:<13}{r['gt']:>6}{r['tp']:>6}{r['fp']:>6}"
                  f"{r['recall']:>9.3f}{r['precision']:>9.3f}")
        if not rows:
            print('  （一条 GT 都没有？）')
        del model
        torch.cuda.empty_cache()

    # A. 原样 80 类零样本
    mA = RTDetrForObjectDetection.from_pretrained(args.model)
    print(f'[diag] A: 原样，num_labels={mA.config.num_labels}')
    run('A 原样 80 类（零样本）', mA)

    # B. 只过 surgery，不训练
    mB = RTDetrForObjectDetection.from_pretrained(args.model)
    mB = F.surgery(mB)
    F.build_loss(mB, F.default_class_weights(4.0))
    print(f'[diag] B: surgery 后，num_labels={mB.config.num_labels}')
    run('B surgery 后 82 类（未训练）', mB)

    # C. hf_best（epoch 0 权重）
    hf = os.path.join(args.out_dir, 'hf_best')
    mC = RTDetrForObjectDetection.from_pretrained(hf)
    print(f'[diag] C: {hf}，num_labels={mC.config.num_labels}')
    run('C hf_best（训过 1 轮）', mC)


if __name__ == '__main__':
    main()
