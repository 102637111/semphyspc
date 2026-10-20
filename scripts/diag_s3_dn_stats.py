#!/usr/bin/env python3
"""统计 S3 训练里**主分支**与**去噪（DN）分支**的 VFL 监督构成——只做前向，不训练。

背景（scripts/diag_s3_collapse.py 的判决）：关掉 `num_denoising` 是唯一让 dev 召回
不崩反升的改动（66→77）。要把它写成论文里的机制，不能只给「关了就好」，得给出
可量化的差异。本脚本在**真实的训练 batch** 上钩住 `loss_labels_vfl`，逐次记录：

  nq        该次调用的 query 数（300 = 主/aux，DN 层则 = max_gt_num*2*num_groups）
  num_boxes 归一化分母（DN 那次已被乘上 dn_num_group）
  n_pos     被匹配为正样本的 query 数
  iou_pos   这些正样本与真值框的 IoU 均值 —— **这正是 VFL 里正样本的标签值
            和权重**（loss_rt_detr.py:187-192），即「模型被要求输出多高的置信度」

若 DN 正样本的 iou_pos 显著低于主分支，则 DN 在大量地教分类头输出低分，
崩塌就有了机制层面的解释（而不是「玄学」）。

用法：
    python3 scripts/diag_s3_dn_stats.py --batches 6
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
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--batches', type=int, default=6)
    ap.add_argument('--samples_per_epoch', type=int, default=12000)
    ap.add_argument('--target_per_class', type=int, default=4000)
    ap.add_argument('--class_boost', type=float, default=4.0)
    ap.add_argument('--no_split_dn', action='store_true',
                    help='复现 transformers 5.12.1 原版行为：主损失不切掉去噪 query。'
                         '默认（不加此开关）用已修正的切分。')
    args = ap.parse_args()

    with open(os.path.join(args.out_dir, 'token_to_images.json')) as f:
        tok2img = json.load(f)
    items, blist, token_of = F.build_items(args.labels, tok2img)
    utok = sorted(set(token_of))
    rng = np.random.default_rng(0)
    rng.shuffle(utok)
    dev_tok = set(utok[:max(1, int(len(utok) * 0.05))])
    tr = [i for i, t in enumerate(token_of) if t not in dev_tok]
    tr_items = [items[i] for i in tr]
    tr_boxes = [blist[i] for i in tr]

    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    from transformers.loss.loss_rt_detr import RTDetrLoss

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    proc = RTDetrImageProcessor.from_pretrained(args.model)
    model = RTDetrForObjectDetection.from_pretrained(args.model)
    model = F.surgery(model, verbose=False)
    F.build_loss(model, F.default_class_weights(args.class_boost),
                 split_dn=not args.no_split_dn)
    # 必须留在 train 模式：RT-DETR 只在 self.training 时才构造去噪组
    # （modeling_rt_detr.py 里 `if self.training and self.config.num_denoising > 0`），
    # 用 eval() 量出来的会是「没有去噪」的假象。
    model.to(dev).train()

    crit = model.loss_function.criterion
    print(f'[cfg] num_denoising = {model.config.num_denoising}  '
          f'num_queries = {model.config.num_queries}  num_labels = {model.config.num_labels}')

    recs = []
    orig = crit.loss_labels_vfl

    def hooked(outputs, targets, indices, num_boxes, log=True):
        with torch.no_grad():
            idx = crit._get_source_permutation_idx(indices)
            from transformers.loss.loss_rt_detr import center_to_corners_format, box_iou
            sb = outputs['pred_boxes'][idx]
            tb = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            if len(sb):
                _iou_mat, _ = box_iou(center_to_corners_format(sb), center_to_corners_format(tb))
                ious = torch.diag(_iou_mat)
                iou_pos = float(ious.mean())
                iou_lo = float((ious < 0.5).float().mean())
            else:
                iou_pos, iou_lo = float('nan'), float('nan')
            # 去噪 query 排在序列最前面 [0, n_dn)。主分支那次调用的 nq 是
            # num_denoising+num_queries（未切分），aux 调用则只有 num_queries——
            # 于是「被匹配为正样本的 query 里有多少落在去噪区」就成了关键判据：
            # 若主损失把去噪 query 当背景压，而去噪损失要它们输出 GT 类，两者互相矛盾。
            n_dn = max(0, int(outputs['logits'].shape[1]) - model.config.num_queries)
            qidx = idx[1]
            recs.append(dict(
                nq=int(outputs['logits'].shape[1]), num_boxes=float(num_boxes),
                n_pos=int(len(idx[0])), iou_pos=iou_pos, frac_iou_lt50=iou_lo,
                n_dn=n_dn,
                n_pos_in_dn=int((qidx < n_dn).sum()) if n_dn else 0,
                # 正样本的 VFL 权重质量 = Σ target_score（上界，未含负样本项）
                pos_mass=float(ious.sum()) if len(sb) else 0.0,
            ))
        return orig(outputs, targets, indices, num_boxes, log=log)

    crit.loss_labels_vfl = hooked

    sampler = F.RepeatFactorSampler(tr_boxes, args.samples_per_epoch, args.target_per_class)
    ds = F.NuScenesDetDataset(tr_items, tr_boxes, proc, hflip=0.5)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, sampler=sampler,
                                     num_workers=args.workers,
                                     collate_fn=lambda b: F.collate(b, proc))
    sampler.set_epoch(0)

    n_gt_all = []
    it = iter(dl)
    for bi in range(args.batches):
        pv, pm, labels = next(it)
        pv = pv.to(dev)
        pm = None if pm is None else pm.to(dev)
        labels = [{k: v.to(dev) for k, v in lb.items()} for lb in labels]
        gts = [len(lb['class_labels']) for lb in labels]
        n_gt_all.append(gts)
        recs.clear()
        with torch.no_grad():
            model(pixel_values=pv, pixel_mask=pm, labels=labels)
        nq_main = model.config.num_queries
        tot = sum(gts)
        # 归类不能只看 nq：开着去噪时 decoder 跑的是 num_denoising+num_queries 个 query，
        # 所以 **主分支那次调用的 nq 也是 500**（含去噪 query）。可靠的判据是 num_boxes：
        # DN 分支的分母被乘过 dn_num_group（loss_rt_detr.py:416）。
        dn = [r for r in recs if r['num_boxes'] > tot + 0.5]
        mn = [r for r in recs if r['num_boxes'] <= tot + 0.5]
        grp = (dn[0]['num_boxes'] / tot) if dn else float('nan')
        print(f'\n--- batch {bi}  GT/图 {gts}  max_gt_num={max(gts)}  合计 {tot} ---')
        m0 = mn[0]
        print(f'    主+aux 调用 {len(mn)} 次（nq：{[r["nq"] for r in mn]}），'
              f'正样本 {m0["n_pos"]} 个（= batch GT 数），'
              f'iou_pos 均值 {np.mean([r["iou_pos"] for r in mn]):.4f}  '
              f'IoU<0.5 占比 {np.mean([r["frac_iou_lt50"] for r in mn]):.3f}')
        if m0['n_dn']:
            print(f'      ↑ 主分支这次调用含 {m0["n_dn"]} 个去噪 query（未切分），'
                  f'被匹配为正样本的 {m0["n_pos"]} 个里只有 {m0["n_pos_in_dn"]} 个落在去噪区')
            for r in mn:
                if r['n_dn']:
                    print(f'        nq={r["nq"]}（真实 {r["nq"] - r["n_dn"]} + 去噪 {r["n_dn"]}）'
                          f'  匹配到去噪区的正样本 {r["n_pos_in_dn"]} / {r["n_pos"]}')
        if dn:
            print(f'    DN   调用 {len(dn)} 次（nq：{[r["nq"] for r in dn]}），'
                  f'num_boxes={dn[0]["num_boxes"]:.0f} = {tot} × dn_num_group {grp:.0f}，'
                  f'正样本 {dn[0]["n_pos"]} 个（= GT × 组数）')
            print(f'      iou_pos 均值 {np.mean([x["iou_pos"] for x in dn]):.4f}   '
                  f'IoU<0.5 占比 {np.mean([x["frac_iou_lt50"] for x in dn]):.3f}')
            print(f'    ⇒ 同一批正样本，主分支被要求输出 {np.mean([r["iou_pos"] for r in mn]):.3f} 的置信度，'
                  f'去噪分支 {np.mean([x["iou_pos"] for x in dn]):.3f}')

    print('\n[汇总]')
    mg = [max(g) for g in n_gt_all]
    grp = [model.config.num_denoising // m for m in mg]
    dnq = [m * 2 * gr for m, gr in zip(mg, grp)]   # 去噪 query 数 = max_gt_num*2*组数
    print(f'  各 batch max_gt_num: {mg}  （batch 内最大 GT 数，注意不是均值）')
    print(f'  ⇒ num_groups = num_denoising({model.config.num_denoising}) // max_gt_num = {grp}')
    print(f'  ⇒ 去噪 query 数 {dnq}，占全部 {model.config.num_queries} 个 query 的 '
          f'{[round(100 * d / (d + model.config.num_queries), 1) for d in dnq]}%')
    print(f'  逐图 GT 数：均值 {np.mean([np.mean(g) for g in n_gt_all]):.1f}，'
          f'中位 {np.median([np.median(g) for g in n_gt_all]):.0f}，'
          f'batch 内最大 {np.mean(mg):.1f}（后者才是 num_groups 的分母）')


if __name__ == '__main__':
    main()
