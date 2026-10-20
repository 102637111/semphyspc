#!/usr/bin/env python3
"""S3 崩塌的判决实验：从「只过 surgery、未训练」的模型起跑，逐段测 dev。

已知（scripts/diag_s3_eval.py）：
  A 原样零样本 67 TP / B 只过 surgery 未训练 66 TP / C 训过 1 轮 3 TP
  ⇒ surgery 无副作用，是训练本身在 ~1500 步内把分类分数整体压低、框也带偏。

这个脚本只跑 150 步（约 35 s/臂），在 0/10/25/50/100/150 步各测一次同一批 24 张 dev 图，
用四个臂隔离嫌疑：
  arm1 本文配置（类别加权 class_boost=4.0、lr 2e-5、hflip 0.5）—— 复现崩塌
  arm2 原版损失（无类别加权）、其余同 arm1        —— 嫌疑：类别加权
  arm3 本文损失但 lr 2e-6（1/10）                —— 嫌疑：学习率
  arm4 本文损失但 hflip=0                        —— 嫌疑：翻转增强
若四臂都在前 10 步内掉光，则问题不在这些超参，而在数据/标签侧。

用法：
    python3 scripts/diag_s3_collapse.py --steps 150 --n_dev 24
"""
import os
import sys
import json
import math
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finetune_rtdetr_nuscenes as F  # noqa: E402


def build_data(args, seed=0):
    with open(os.path.join(args.out_dir, 'token_to_images.json')) as f:
        tok2img = json.load(f)
    items, blist, token_of = F.build_items(args.labels, tok2img)
    utok = sorted(set(token_of))
    rng = np.random.default_rng(seed)
    rng.shuffle(utok)
    n_dev = max(1, int(len(utok) * args.dev_frac))
    dt = set(utok[:n_dev])
    tr = [i for i, t in enumerate(token_of) if t not in dt]
    dv = [i for i, t in enumerate(token_of) if t in dt]
    if args.dev_max_items and len(dv) > args.dev_max_items:
        dv = list(rng.choice(dv, size=args.dev_max_items, replace=False))
    return ([items[i] for i in tr], [blist[i] for i in tr],
            [items[i] for i in dv[:args.n_dev]], [blist[i] for i in dv[:args.n_dev]])


def run_arm(tag, args, data, dev_name, stock_loss=False, lr=2e-5, hflip=0.5,
            freeze_heads_only=False, box_w=None, freeze_prefix=None,
            freeze_except=(), no_dn=False, split_dn=True):
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    from transformers.loss.loss_rt_detr import RTDetrLoss

    tr_items, tr_boxes, dv_items, dv_boxes = data
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    proc = RTDetrImageProcessor.from_pretrained(args.model)
    model = RTDetrForObjectDetection.from_pretrained(args.model)
    model = F.surgery(model, verbose=False)
    if not stock_loss:
        # stock_loss=True 时不挂任何东西：PreTrainedModel.loss_function 的 property
        # 默认就返回原版 RTDetrForObjectDetectionLoss，正是我们要的对照。
        F.build_loss(model, F.default_class_weights(args.class_boost), split_dn=split_dn)
    if box_w is not None:
        _wd = model.loss_function.criterion.weight_dict
        _wd['loss_bbox'], _wd['loss_giou'] = box_w[0], box_w[1]
    if no_dn:
        # 去噪训练：每图仅约 6 个目标，num_denoising=100 会展开成 16 个去噪组，
        # 去噪 query 数远超真实正样本 ⇒ 分类头主要在被去噪目标驱动。
        model.config.num_denoising = 0
    print(f'  （生效损失 = {type(model.loss_function).__name__}，'
          f'num_denoising = {getattr(model.config, "num_denoising", "?")}）', flush=True)
    if freeze_heads_only:
        keep = ('class_embed', 'enc_score_head', 'denoising_class_embed')
        n_tr = n_fr = 0
        for n_, p in model.named_parameters():
            p.requires_grad = any(k in n_ for k in keep)
            n_tr += p.requires_grad
            n_fr += (not p.requires_grad)
        print(f'  （冻结非分类头：可训 {n_tr} 个张量 / 冻结 {n_fr} 个）', flush=True)
    if freeze_prefix:
        n_tr = n_fr = 0
        for n_, p in model.named_parameters():
            if n_.startswith(freeze_prefix) and not any(k in n_ for k in freeze_except):
                p.requires_grad = False
            n_tr += p.requires_grad
            n_fr += (not p.requires_grad)
        print(f'  （冻结前缀 {freeze_prefix}（豁免 {freeze_except}）：'
              f'可训 {n_tr} 个张量 / 冻结 {n_fr} 个）', flush=True)
    model.to(dev)

    bb, hp = [], []
    for n_, p in model.named_parameters():
        if p.requires_grad:
            (bb if n_.startswith('model.backbone') else hp).append(p)
    opt = torch.optim.AdamW([{'params': bb, 'lr': lr * 0.1}, {'params': hp, 'lr': lr}],
                            weight_decay=args.wd)
    total = 24000

    def lr_at(s):
        if s < 500:
            return (s + 1) / 500
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - 500) / (total - 500))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    sampler = F.RepeatFactorSampler(tr_boxes, args.samples_per_epoch, args.target_per_class)
    ds = F.NuScenesDetDataset(tr_items, tr_boxes, proc, hflip=hflip)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.bs, sampler=sampler, num_workers=args.workers,
        collate_fn=lambda b: F.collate(b, proc), pin_memory=True, drop_last=True)
    sampler.set_epoch(0)

    def brk(ld):
        """按 主/aux/dn × vfl/bbox/giou 拆损失，去掉 weight_dict 权重以便横向比。"""
        lf = model.loss_function
        wd = getattr(getattr(lf, 'criterion', lf), 'weight_dict', None)
        if wd is None:
            wd = {'loss_vfl': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
        s = {}
        for k, v in ld.items():
            base = k.split('_aux_')[0].split('_dn_')[0]
            kind = 'dn' if '_dn_' in k else ('aux' if '_aux_' in k else '主')
            key = f'{kind}/{base.replace("loss_", "")}'
            s[key] = s.get(key, 0.0) + float(v) / max(wd.get(base, 1.0), 1e-9)
        return s

    def score(step):
        model.eval()
        rows = F.evaluate_recall(model, proc, dv_items, dv_boxes, dev,
                                 score_thr=0.35, max_items=0)
        # 低阈值对照：若 TP@0.05 保持而 TP@0.35 掉，则是分数标定漂移而非模型失效
        rows_lo = F.evaluate_recall(model, proc, dv_items, dv_boxes, dev,
                                    score_thr=0.05, max_items=0)
        model.train()
        gt = sum(r['gt'] for r in rows)
        tp_lo = sum(r['tp'] for r in rows_lo)
        tp = sum(r['tp'] for r in rows)
        watch = [r for r in rows if r['cls'] in F.RARE_COCO]
        wr = float(np.mean([r['recall'] for r in watch])) if watch else 0.0
        return gt, tp, wr, tp_lo

    print(f'\n########## {tag} （lr={lr:g}, hflip={hflip}, '
          f'{"原版损失" if stock_loss else "类别加权 4.0"}'
          f'{", 只训分类头" if freeze_heads_only else ""}'
          f'{", box权重 " + str(box_w) if box_w else ""}）##########', flush=True)
    g0, t0, w0, t0lo = score(0)
    print(f'  step   0   TP/GT = {t0:>4}/{g0:<4}  (thr0.05: {t0lo})  '
          f'关注类平均召回 {w0:.4f}', flush=True)

    # 监测 decoder 各层 final_layer_norm 的 |weight| 均值（崩塌若源于激活尺度正反馈，这里会先涨）
    fln = [n for n, _ in model.named_parameters()
           if 'decoder.layers' in n and 'final_layer_norm.weight' in n]

    def norms():
        with torch.no_grad():
            v = [float(model.get_parameter(n).abs().mean()) for n in fln]
        return float(np.mean(v)) if v else 0.0

    print(f'  [FLN] step 0  decoder final_layer_norm |w| 均值 {norms():.5f}', flush=True)

    model.train()
    it = iter(dl)
    t_start = time.time()
    last = {}
    for step in range(1, args.steps + 1):
        pv, pm, labels = next(it)
        pv = pv.to(dev, non_blocking=True)
        pm = None if pm is None else pm.to(dev, non_blocking=True)
        labels = [{k: v.to(dev, non_blocking=True) for k, v in lb.items()} for lb in labels]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            out = model(pixel_values=pv, pixel_mask=pm, labels=labels)
            loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 0.1)
        opt.step()
        sched.step()
        if getattr(out, 'loss_dict', None):
            last = brk(out.loss_dict)
        if step in (10, 25, 50, 100, 150, 300, 600) and step <= args.steps:
            g, t, w, tlo = score(step)
            print(f'  step {step:>3}  TP/GT = {t:>4}/{g:<4}  关注类平均召回 {w:.4f}  '
                  f'(thr0.05: {tlo})  loss {float(loss):.4f}  FLN|w| {norms():.5f}  '
                  f'({time.time() - t_start:.0f}s)',
                  flush=True)
            if last:
                order = ['主/vfl', '主/bbox', '主/giou', 'aux/vfl', 'aux/bbox', 'aux/giou',
                         'dn/vfl', 'dn/bbox', 'dn/giou']
                seg = '  '.join(f'{k}={last[k]:.3f}' for k in order if k in last)
                extra = '  '.join(f'{k}={v:.3f}' for k, v in sorted(last.items()) if k not in order)
                print(f'         损失分量(去权) {seg}' + (f'  |  {extra}' if extra else ''),
                      flush=True)
    del model, opt, dl, ds
    torch.cuda.empty_cache()


def run_overfit(args, data):
    """判决测试：把同一张图反复喂 150 步。标签/目标构造正确的话，分数应当升上去；
    若连单图都拟合不了（分数反而被压低），问题就在数据/目标侧，与优化无关。"""
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

    tr_items, tr_boxes = data[0], data[1]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    proc = RTDetrImageProcessor.from_pretrained(args.model)

    # 挑一张 GT 多的图，便于观察
    k = int(np.argmax([len(b) for b in tr_boxes[:2000]]))
    one_i, one_b = [tr_items[k]], [tr_boxes[k]]
    print(f'\n########## arm5 过拟合单图 ##########\n'
          f'  {os.path.basename(tr_items[k][2])}  该相机 GT {len(tr_boxes[k])} 框  '
          f'类别 {sorted(set(tr_boxes[k][:, 4].astype(int).tolist()))}', flush=True)

    model = RTDetrForObjectDetection.from_pretrained(args.model)
    model = F.surgery(model, verbose=False)
    F.build_loss(model, F.default_class_weights(args.class_boost))
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)   # 过拟合用小网络常用的较大 lr
    ds = F.NuScenesDetDataset(one_i, one_b, proc, hflip=0.0)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True,
                                     collate_fn=lambda b: F.collate(b, proc))

    def peak():
        model.eval()
        rows = F.evaluate_recall(model, proc, one_i, one_b, dev, score_thr=0.35, max_items=0)
        model.train()
        gt = sum(r['gt'] for r in rows); tp = sum(r['tp'] for r in rows)
        # 另外看「GT 类上的最高分」，不受 0.35 阈值影响
        from PIL import Image
        with torch.no_grad():
            img = Image.open(one_i[0][2]).convert('RGB')
            enc = proc(images=[img], return_tensors='pt').to(dev)
            p = model(**enc).logits[0].sigmoid()
        gtcls = sorted(set(one_b[0][:, 4].astype(int).tolist()))
        return gt, tp, float(p.max()), {c: round(float(p[:, c].max()), 3) for c in gtcls}

    g, t, mx, per = peak()
    print(f'  step   0  TP/GT={t}/{g}  全局最高分 {mx:.3f}  逐 GT 类最高分 {per}', flush=True)
    model.train()
    it = iter(dl)
    for step in range(1, args.overfit_steps + 1):
        try:
            pv, pm, labels = next(it)
        except StopIteration:
            it = iter(dl); pv, pm, labels = next(it)
        pv = pv.to(dev); pm = None if pm is None else pm.to(dev)
        labels = [{kk: v.to(dev) for kk, v in lb.items()} for lb in labels]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            out = model(pixel_values=pv, pixel_mask=pm, labels=labels)
            loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        opt.step()
        if step % 25 == 0:
            g, t, mx, per = peak()
            print(f'  step {step:>3}  TP/GT={t}/{g}  全局最高分 {mx:.3f}  loss {float(loss.detach()):.3f}  '
                  f'逐 GT 类最高分 {per}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default='data/s3_det_labels_train.npz')
    ap.add_argument('--out_dir', default='experiments/s3_rtdetr_finetune')
    ap.add_argument('--model', default='PekingU/rtdetr_r50vd')
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--n_dev', type=int, default=24)
    ap.add_argument('--dev_frac', type=float, default=0.05)
    ap.add_argument('--dev_max_items', type=int, default=600)
    ap.add_argument('--samples_per_epoch', type=int, default=12000)
    ap.add_argument('--target_per_class', type=int, default=4000)
    ap.add_argument('--class_boost', type=float, default=4.0)
    ap.add_argument('--arms', default='1,2,3,4')
    ap.add_argument('--overfit_steps', type=int, default=0,
                    help='>0 时只跑「过拟合单图」判决测试')
    args = ap.parse_args()

    t = time.time()
    data = build_data(args)
    print(f'[数据] train {len(data[0]):,} 图 / dev 取 {len(data[2])} 图（{time.time() - t:.0f}s）')

    if args.overfit_steps > 0:
        run_overfit(args, data)
        return

    arms = {
        '1': ('arm1 本文配置', dict(stock_loss=False, lr=2e-5, hflip=0.5)),
        '2': ('arm2 原版损失', dict(stock_loss=True, lr=2e-5, hflip=0.5)),
        '3': ('arm3 lr 2e-6', dict(stock_loss=False, lr=2e-6, hflip=0.5)),
        '4': ('arm4 关掉翻转', dict(stock_loss=False, lr=2e-5, hflip=0.0)),
        '5': ('arm5 只训分类头', dict(stock_loss=False, lr=1e-4, hflip=0.5,
                                     freeze_heads_only=True)),
        '6': ('arm6 box 权重降到 1/0.5', dict(stock_loss=False, lr=2e-5, hflip=0.5,
                                             box_w=(1.0, 0.5))),
        '7': ('arm7 box 权重归零（只留分类损失）', dict(stock_loss=False, lr=2e-5, hflip=0.5,
                                                      box_w=(0.0, 0.0))),
        '10': ('arm10 去噪开着但 lr 减半（排除「梯度尺度」混淆）',
               dict(stock_loss=False, lr=1e-5, hflip=0.5)),
        '9': ('arm9 全参数微调但关掉去噪', dict(stock_loss=False, lr=2e-5, hflip=0.5,
                                                  no_dn=True)),
        '11': ('arm11 去噪开着但修正主损失切分', dict(stock_loss=False, lr=2e-5, hflip=0.5,
                                                     split_dn=True)),
        '12': ('arm12 复现原版（主损失不切分，与 arm1 同义）',
               dict(stock_loss=False, lr=2e-5, hflip=0.5, split_dn=False)),
        '8': ('arm8 冻结 decoder 非分类权重', dict(stock_loss=False, lr=2e-5, hflip=0.5,
                                                   freeze_prefix='model.decoder.',
                                                   freeze_except=('class_embed',))),
    }
    for k in args.arms.split(','):
        tag, kw = arms[k.strip()]
        run_arm(tag, args, data, k, **kw)


if __name__ == '__main__':
    main()
