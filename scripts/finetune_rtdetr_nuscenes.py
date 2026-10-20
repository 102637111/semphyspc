#!/usr/bin/env python3
"""S3 第 2 步：在 nuScenes **train split** 的投影 2D 标签上微调 RT-DETR。

目的（§4.13 的 S2 分支）：COCO 零样本检测器只认 person / bicycle / motorcycle，
barrier(9) 与 trafficcone(12) 完全没有对应类，且 3D 实例级召回只有 0.205
（而用评测集真值投影的 oracle 上界是 0.861）。这里用**训练集**标注做标准有监督微调，
把检测器换成 82 类（COCO 80 类 + barrier + trafficcone），
前 80 行权重完整继承 COCO 预训练，只有新类两行是随机初始化。

方法论要点（避免重蹈 §4.13 的泄漏覆辙）：
  - 标签只来自 **train split**，评测 split 的标注**全程不参与**训练与选型；
  - 从 train split 里切 5% 作为 dev，用于监控 early-stop，绝不碰 val。

头部改造（transformers 5.12.1 的 RT-DETR 结构）：
  model.model.decoder.class_embed.{0..5}  nn.Linear(256, 80)  -> (256, 82)
  model.model.enc_score_head              nn.Linear(256, 80)  -> (256, 82)   # 两阶段提案
  model.model.denoising_class_embed       nn.Embedding(81, 256, padding_idx=80)
                                          -> nn.Embedding(83, 256, padding_idx=82)
其中 no-object 是**最后一行**（源码 `nn.Embedding(num_labels+1, d_model, padding_idx=num_labels)`），
所以旧的第 80 行要搬到新的第 82 行。改完必须重建 loss（`RTDetrLoss` 在 __init__ 里缓存
num_labels），且**绝不能重跑 post_init**（`_init_weights` 会把分类头按新 num_labels 重新初始化，
把预训练权重全部冲掉）。

用法：
    python3 scripts/finetune_rtdetr_nuscenes.py \
        --labels data/s3_det_labels_train.npz --dataroot /mnt/data/nuscenes \
        --out_dir experiments/s3_rtdetr_finetune --epochs 8 --bs 8 \
        --samples_per_epoch 8000 --dev_frac 0.05
"""
import os
import sys
import json
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CAM_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
NUM_CLASSES = 82
NEW_CLASSES = {'barrier': 80, 'trafficcone': 81}

# 需要过采样的类（COCO 里没有或本就稀少的），键是 COCO id
RARE_COCO = [1, 3, 80, 81]          # bicycle, motorcycle, barrier, trafficcone


def cls_name(cid):
    coco = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    return coco.get(cid, {80: 'barrier', 81: 'trafficcone'}.get(cid, f'id{cid}'))


# --------------------------------------------------------------------------- 头部改造
def surgery(model, n_new=NUM_CLASSES, verbose=True):
    """80 类 COCO 检测器 -> n_new 类，保留预训练权重，no-object 行搬到末位。"""
    old_num = model.config.num_labels
    assert n_new > old_num, f'只支持扩类：{old_num} -> {n_new}'
    d = model.config.d_model

    def grow_linear(lin):
        new = nn.Linear(lin.in_features, n_new)
        with torch.no_grad():
            new.weight[:old_num] = lin.weight
            new.bias[:old_num] = lin.bias
            # 新类的偏置先给一个低先验（对应 prior_prob≈1/(n+1)），避免一上来就狂报
            nn.init.normal_(new.weight[old_num:], std=0.01)
            nn.init.constant_(new.bias[old_num:], -4.0)
        return new

    # 1) 解码器 6 个分类头
    ce = model.model.decoder.class_embed
    for i in range(len(ce)):
        ce[i] = grow_linear(ce[i])

    # 2) 编码器两阶段提案头
    model.model.enc_score_head = grow_linear(model.model.enc_score_head)

    # 3) 去噪训练的类别嵌入：末位是 no-object
    old_emb = model.model.denoising_class_embed
    new_emb = nn.Embedding(n_new + 1, d, padding_idx=n_new)
    with torch.no_grad():
        new_emb.weight[:old_num] = old_emb.weight[:old_num]      # 旧的真实类
        nn.init.xavier_uniform_(new_emb.weight[old_num:n_new])   # 新类：用原始初始化分布
        new_emb.weight[n_new] = old_emb.weight[old_num]          # 旧 no-object -> 新末位
    model.model.denoising_class_embed = new_emb

    model.config.num_labels = n_new
    id2label = {i: cls_name(i) for i in range(n_new)}
    model.config.id2label = id2label
    model.config.label2id = {v: k for k, v in id2label.items()}

    if verbose:
        print(f'[surgery] {old_num} -> {n_new} 类；'
              f'class_embed×{len(ce)}、enc_score_head、denoising_class_embed(末位=no-object) 已扩')
    return model


def build_loss(model, class_weights=None, split_dn=True):
    """重建分类损失：RTDetrLoss 在 __init__ 缓存 num_labels，必须重建；
    并在 VFL 上对正样本按类别加权（缓解 person/car 主导）。

    三个必须注意的 transformers 5.x 细节（旧写法都会**静默失效**）：
      1. `loss_function` 是 PreTrainedModel 上的 **property**，forward 里
         `self.loss_function(...)` 永远走 property；`model.loss_function = X` 会被
         nn.Module 塞进 _modules 并被 property 遮蔽，取不回来。property 优先读
         `self._loss_function`，所以必须写 **`model._loss_function = ...`**。
      2. 5.x 的损失是**函数式接口** `f(logits, labels, device, pred_boxes, config, ...)
         -> (loss, loss_dict, auxiliary_outputs)`，不是老的 `RTDetrLoss(outputs, targets)`。
         所以外层要包一个 Module 复刻 `RTDetrForObjectDetectionLoss` 的主体，
         内层才是我们改写 loss_labels_vfl 的那个 RTDetrLoss 子类。
      3. **原版 `RTDetrForObjectDetectionLoss` 的主分支没有切掉去噪 query**（见
         下方 forward 里的长注释），会让匈牙利匹配把正样本几乎全配给去噪 query。
         `split_dn=True`（默认）修正这一点；只在复现旧行为/对照实验时置 False。
    """
    import torch.nn as nn
    from transformers.loss.loss_rt_detr import RTDetrLoss, _set_aux_loss

    class WeightedRTDetrLoss(RTDetrLoss):
        def __init__(self, config, cw):
            super().__init__(config)
            self.register_buffer('cls_weight', torch.as_tensor(cw, dtype=torch.float32))

        def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
            from transformers.loss.loss_for_object_detection import box_iou
            from transformers.image_transforms import center_to_corners_format
            idx = self._get_source_permutation_idx(indices)
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(center_to_corners_format(src_boxes.detach()),
                              center_to_corners_format(target_boxes))
            ious = torch.diag(ious)

            src_logits = outputs['logits']
            dtype = src_logits.dtype
            tc_orig = torch.cat([t['class_labels'][i] for t, (_, i) in zip(targets, indices)])
            tc = torch.full(src_logits.shape[:2], self.num_classes,
                            dtype=torch.int64, device=src_logits.device)
            tc[idx] = tc_orig
            target = F.one_hot(tc, num_classes=self.num_classes + 1)[..., :-1]

            tgt_score_orig = torch.zeros_like(tc, dtype=dtype)
            tgt_score_orig[idx] = ious.to(dtype)
            target_score = tgt_score_orig.unsqueeze(-1) * target

            pred_score = F.sigmoid(src_logits.detach())
            weight = (self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score).to(dtype)
            # 仅对**正样本所在通道**乘类别权重，不放大背景通道的惩罚
            cw = self.cls_weight[:self.num_classes][None, None, :]
            weight = weight * (1.0 + (cw - 1.0) * target)

            loss = F.binary_cross_entropy_with_logits(src_logits, target_score,
                                                      weight=weight, reduction='none')
            return {'loss_vfl': loss.mean(1).sum() * src_logits.shape[1] / num_boxes}

    class WeightedRTDetrForObjectDetectionLoss(nn.Module):
        """复刻 transformers 5.x 的 RTDetrForObjectDetectionLoss 主体，
        只把内部的 criterion 换成带类别加权的 WeightedRTDetrLoss。"""

        def __init__(self, config, cw):
            super().__init__()
            self.config = config
            self.criterion = WeightedRTDetrLoss(config, cw)

        def forward(self, logits, labels, device, pred_boxes, config=None,
                    outputs_class=None, outputs_coord=None, enc_topk_logits=None,
                    enc_topk_bboxes=None, denoising_meta_values=None, **kwargs):
            self.criterion.to(device)
            # ---- 主分支必须先切掉去噪 query（HF 5.12.1 原版没有切，是个真 bug）----
            # 开着去噪时 decoder 跑的是 num_denoising + num_queries 个 query，所以传进来的
            # logits/pred_boxes 宽度是 500 而不是 300。RTDetrForObjectDetectionLoss 在
            # loss_rt_detr.py:450 直接 `outputs_loss["logits"] = logits`（未切分），
            # 却在 :455-456 给 aux 分支做了 torch.split——主分支因此把 200 个去噪 query
            # 当普通 query 一起算匈牙利匹配。去噪 query 的参考点就是加噪后的 GT 框，
            # 匹配时几乎必胜，实测**主分支 90~100% 的正样本都配给了去噪 query**
            # （scripts/diag_s3_dn_stats.py），于是推理时真正使用的 300 个真实 query
            # 被大面积当成背景压——分类分数整体塌掉而框回归照常变好，正是 S3 崩塌的成因。
            # 这里按 dn_num_split 切回真实 query，与 aux / 去噪两个分支对齐。
            if denoising_meta_values is not None and split_dn:
                n_dn = int(denoising_meta_values['dn_num_split'][0])
                if logits.shape[1] > n_dn:
                    logits = logits[:, n_dn:]
                    pred_boxes = pred_boxes[:, n_dn:]
            outputs_loss = {'logits': logits, 'pred_boxes': pred_boxes}
            auxiliary_outputs = None
            cfg = config if config is not None else self.config
            if cfg.auxiliary_loss:
                if denoising_meta_values is not None:
                    split = denoising_meta_values['dn_num_split']
                    dn_out_coord, outputs_coord = torch.split(outputs_coord, split, dim=2)
                    dn_out_class, outputs_class = torch.split(outputs_class, split, dim=2)
                auxiliary_outputs = _set_aux_loss(outputs_class[:, :-1].transpose(0, 1),
                                                  outputs_coord[:, :-1].transpose(0, 1))
                outputs_loss['auxiliary_outputs'] = auxiliary_outputs
                outputs_loss['auxiliary_outputs'].extend(
                    _set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))
                if denoising_meta_values is not None:
                    outputs_loss['dn_auxiliary_outputs'] = _set_aux_loss(
                        dn_out_class.transpose(0, 1), dn_out_coord.transpose(0, 1))
                    outputs_loss['denoising_meta_values'] = denoising_meta_values
            loss_dict = self.criterion(outputs_loss, labels)
            return sum(loss_dict.values()), loss_dict, auxiliary_outputs

    if class_weights is None:
        class_weights = np.ones(NUM_CLASSES, dtype=np.float32)
    loss = WeightedRTDetrForObjectDetectionLoss(model.config, class_weights)
    model._loss_function = loss
    return loss


def default_class_weights(boost=4.0, boost_set=(1, 3, 80, 81)):
    cw = np.ones(NUM_CLASSES, dtype=np.float32)
    for c in boost_set:
        cw[c] = boost
    return cw


# --------------------------------------------------------------------------- 数据
class NuScenesDetDataset(Dataset):
    """索引 = (lidar_token, cam_idx) 图像，标签 = 该图上的 2D 框。"""

    def __init__(self, items, boxes_by_item, processor, hflip=0.5):
        self.items = items                      # list[(lidar_token, cam_idx, img_path)]
        self.boxes = boxes_by_item              # list[np.ndarray (n,5) = x,y,w,h,cls]
        self.proc = processor
        self.hflip = hflip

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        tok, cam, path = self.items[i]
        img = Image.open(path).convert('RGB')
        W, H = img.size
        b = self.boxes[i].astype(np.float32).copy()

        if self.hflip > 0 and np.random.rand() < self.hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            b[:, 0] = W - b[:, 0] - b[:, 2]

        # xywh(像素) -> 归一化 cxcywh
        cx = (b[:, 0] + b[:, 2] / 2) / W
        cy = (b[:, 1] + b[:, 3] / 2) / H
        bw = b[:, 2] / W
        bh = b[:, 3] / H
        boxes = np.stack([cx, cy, bw, bh], axis=1).clip(0.0, 1.0)
        keep = (boxes[:, 2] > 1e-3) & (boxes[:, 3] > 1e-3)
        boxes, labels = boxes[keep], b[keep, 4].astype(np.int64)

        return dict(pixel_values=img, boxes=torch.from_numpy(boxes),
                    class_labels=torch.from_numpy(labels))


def collate(batch, processor):
    """RT-DETR 要求 labels 是 list[dict]，所以自己 collate。"""
    imgs = [x['pixel_values'] for x in batch]
    enc = processor(images=imgs, return_tensors='pt')
    labels = [{'class_labels': x['class_labels'], 'boxes': x['boxes']} for x in batch]
    return enc['pixel_values'], enc.get('pixel_mask'), labels


def build_items(labels_npz, token_to_imgs, min_box_px=2.0):
    """npz -> (items, boxes_by_item)。items 与 boxes 一一对应。"""
    d = np.load(labels_npz, allow_pickle=True)
    tokens = [str(t) for t in d['tokens']]
    offsets, boxes = d['offsets'], d['boxes']
    items, blist, token_of = [], [], []
    for i, tok in enumerate(tokens):
        sl = boxes[offsets[i]:offsets[i + 1]]
        if len(sl) == 0:
            continue
        paths = token_to_imgs.get(tok)
        if paths is None:
            continue
        for cam in range(len(CAM_CHANNELS)):
            m = sl[sl[:, 7].astype(int) == cam]
            if len(m) == 0:
                continue
            m = m[(m[:, 2] >= min_box_px) & (m[:, 3] >= min_box_px)]
            if len(m) == 0:
                continue
            items.append((tok, cam, paths[cam]))
            blist.append(np.stack([m[:, 0], m[:, 1], m[:, 2], m[:, 3], m[:, 4]], axis=1))
            token_of.append(tok)
    return items, blist, token_of


class RepeatFactorSampler(torch.utils.data.Sampler):
    """按类目标出现次数对图像过采样：让 barrier/trafficcone 等稀有类每轮被看到足够多次。"""

    def __init__(self, boxes_by_item, samples_per_epoch, target_per_class=3000, max_rep=20, seed=0):
        n = len(boxes_by_item)
        cnt = np.zeros(NUM_CLASSES, dtype=np.int64)
        present = [[] for _ in range(NUM_CLASSES)]
        for i, b in enumerate(boxes_by_item):
            for c in np.unique(b[:, 4].astype(int)):
                cnt[c] += 1
                present[c].append(i)
        rep = np.ones(NUM_CLASSES, dtype=np.float64)
        for c in RARE_COCO:
            if cnt[c] > 0:
                rep[c] = min(max_rep, max(1.0, target_per_class / cnt[c]))
        w = np.ones(n, dtype=np.float64)
        for c in RARE_COCO:
            if cnt[c] > 0:
                w[present[c]] = np.maximum(w[present[c]], rep[c])
        self.w = w / w.sum()
        self.n = n
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        print(f'[sampler] 图像数 {n:,}；稀有类出现图像数 ' +
              ', '.join(f'{cls_name(c)}={cnt[c]:,}(×{rep[c]:.1f})' for c in RARE_COCO))

    def set_epoch(self, e):
        self.epoch = e
        rng = np.random.default_rng(self.seed + e)
        self.idx = rng.choice(self.n, size=self.samples_per_epoch, p=self.w)

    def __iter__(self):
        return iter(self.idx.tolist())

    def __len__(self):
        return self.samples_per_epoch


# --------------------------------------------------------------------------- dev 评测
@torch.no_grad()
def evaluate_recall(model, processor, items, boxes_by_item, device,
                    score_thr=0.35, iou_thr=0.5, batch=8, max_items=0):
    """按类统计 2D 召准：贪心匹配 IoU>=iou_thr 视为命中。直接对齐 S2 关心的「召回」。"""
    from PIL import Image
    from transformers.loss.loss_for_object_detection import box_iou
    from transformers.image_transforms import center_to_corners_format
    model.eval()
    n = len(items) if max_items <= 0 else min(len(items), max_items)
    tp = np.zeros(NUM_CLASSES); fp = np.zeros(NUM_CLASSES); ngt = np.zeros(NUM_CLASSES)

    for s in range(0, n, batch):
        chunk = list(range(s, min(s + batch, n)))
        imgs = [Image.open(items[i][2]).convert('RGB') for i in chunk]
        sizes = [(im.size[1], im.size[0]) for im in imgs]
        enc = processor(images=imgs, return_tensors='pt').to(device)
        outs = model(**enc)
        res = processor.post_process_object_detection(
            outs, target_sizes=torch.tensor(sizes, device=device), threshold=score_thr)

        for k, i in enumerate(chunk):
            W, H = imgs[k].size
            g = boxes_by_item[i]
            gt_xyxy = torch.tensor(
                np.stack([g[:, 0], g[:, 1], g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]], 1), device=device)
            gt_cls = torch.tensor(g[:, 4].astype(np.int64), device=device)
            pb = res[k]['boxes']; pl = res[k]['labels']; ps = res[k]['scores']
            if len(pb) == 0:
                for c in np.unique(gt_cls.cpu().numpy()):
                    ngt[c] += int((gt_cls.cpu().numpy() == c).sum())
                continue
            iou, _ = box_iou(pb, gt_xyxy)   # 注意 box_iou 返回 (iou, union) 元组
            matched = torch.zeros(len(gt_xyxy), dtype=torch.bool, device=device)
            order = torch.argsort(-ps)
            for j in order.tolist():
                c = int(pl[j])
                same = (gt_cls == c) & (~matched)
                if not same.any():
                    fp[c] += 1
                    continue
                cand = torch.where(same)[0]
                best = cand[torch.argmax(iou[j, cand])]
                if iou[j, best] >= iou_thr:
                    matched[best] = True
                    tp[c] += 1
                else:
                    fp[c] += 1
            for c in np.unique(gt_cls.cpu().numpy()):
                ngt[c] += int((gt_cls.cpu().numpy() == c).sum())

    model.train()
    rows = []
    for c in range(NUM_CLASSES):
        if ngt[c] == 0:
            continue
        rows.append(dict(cls=c, name=cls_name(c), gt=int(ngt[c]), tp=int(tp[c]),
                         fp=int(fp[c]), recall=float(tp[c] / ngt[c]),
                         precision=float(tp[c] / max(1, tp[c] + fp[c]))))
    return rows


# --------------------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--dataroot', default='/mnt/data/nuscenes')
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--model', default='PekingU/rtdetr_r50vd')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--samples_per_epoch', type=int, default=8000)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--backbone_lr_scale', type=float, default=0.1)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--clip', type=float, default=0.1)
    ap.add_argument('--dev_frac', type=float, default=0.05)
    ap.add_argument('--dev_max_items', type=int, default=400)
    ap.add_argument('--target_per_class', type=int, default=3000)
    ap.add_argument('--class_boost', type=float, default=4.0)
    ap.add_argument('--hflip', type=float, default=0.5)
    ap.add_argument('--no_denoising', action='store_true',
                    help='关掉对比去噪训练（num_denoising=0）。**一般不需要**：去噪本身有益，'
                         '崩塌的真因是 build_loss 修正掉的那处主损失未切分（见其 docstring）。'
                         '留着仅供对照实验：dev 召回 arm11 修正切分+去噪开 81 / arm9 关去噪 77。')
    ap.add_argument('--freeze', default='none',
                    choices=['none', 'heads', 'decoder_noclass'],
                    help='none=全参数微调（默认，配合 --no_denoising）；'
                         'heads=只训分类头；decoder_noclass=冻结 decoder 非分类权重')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--eval_every', type=int, default=1)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--smoke', action='store_true', help='只跑 20 步验证流程')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    from nuscenes.nuscenes import NuScenes
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True

    # ---- 图像路径索引（缓存到 out_dir，省得每次重建）
    cache_f = os.path.join(args.out_dir, 'token_to_images.json')
    d0 = np.load(args.labels, allow_pickle=True)
    tokens = [str(t) for t in d0['tokens']]
    if os.path.exists(cache_f):
        with open(cache_f) as f:
            token_to_imgs = json.load(f)
        print(f'[data] 复用图像索引 {cache_f}')
    else:
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
        token_to_imgs = {}
        for i, tok in enumerate(tokens):
            sd = nusc.get('sample_data', tok)
            sample = nusc.get('sample', sd['sample_token'])
            token_to_imgs[tok] = [os.path.join(nusc.dataroot,
                                               nusc.get('sample_data', sample['data'][cam])['filename'])
                                  for cam in CAM_CHANNELS]
            if (i + 1) % 5000 == 0:
                print(f'  ... 图像索引 {i + 1}/{len(tokens)}', flush=True)
        with open(cache_f, 'w') as f:
            json.dump(token_to_imgs, f)
        print(f'[data] 图像索引已缓存 {cache_f}')

    items, blist, token_of = build_items(args.labels, token_to_imgs)
    print(f'[data] 有标注的图像 {len(items):,}（来自 {len(set(token_of)):,} 帧）')

    # ---- train / dev 按**帧**切分，绝不让同一帧的两个相机分居两侧
    utok = sorted(set(token_of))
    rng = np.random.default_rng(0)
    rng.shuffle(utok)
    n_dev = max(1, int(len(utok) * args.dev_frac))
    dev_tok = set(utok[:n_dev])
    tr_idx = [i for i, t in enumerate(token_of) if t not in dev_tok]
    dv_idx = [i for i, t in enumerate(token_of) if t in dev_tok]
    if args.dev_max_items and len(dv_idx) > args.dev_max_items:
        dv_idx = list(rng.choice(dv_idx, size=args.dev_max_items, replace=False))
    print(f'[data] train 图像 {len(tr_idx):,} / dev 图像 {len(dv_idx):,}'
          f'（dev 帧 {n_dev:,}，与 train 帧不重叠）')

    if args.smoke:
        tr_idx = tr_idx[:64]
        dv_idx = dv_idx[:16]
        args.samples_per_epoch = 64
        args.epochs = 1
        args.warmup = 5

    # ---- 模型 + 头部改造
    processor = RTDetrImageProcessor.from_pretrained(args.model)
    model = RTDetrForObjectDetection.from_pretrained(args.model)
    model = surgery(model)
    id2label = model.config.id2label
    assert id2label[80] == 'barrier' and id2label[81] == 'trafficcone'
    cw = default_class_weights(args.class_boost)
    build_loss(model, cw)
    if args.no_denoising:
        model.config.num_denoising = 0
        print('[dn] 已关闭对比去噪训练（num_denoising=0）')

    # 冻结策略。诊断（scripts/diag_s3_collapse.py）表明崩塌是**分类分数整体下压**，
    # 与几何无关：把框损失权重归零、几何全程冻结（arm7）照样在 0.35 阈值上掉光，
    # 而同一批模型在 0.05 阈值上的命中数反而上升（92→101）——排序没坏，尺度被压低了。
    # 真因是主损失未切掉去噪 query（已由 build_loss 的 split_dn 修正），修正后全参数
    # 微调即可提升（arm11：66→81）。因此默认不冻结；--freeze 保留给隔离变量的对照实验。
    if args.freeze != 'none':
        if args.freeze == 'heads':
            keep = ('class_embed', 'enc_score_head', 'denoising_class_embed')
            want = lambda n_: any(k in n_ for k in keep)
        elif args.freeze == 'decoder_noclass':
            want = lambda n_: not (n_.startswith('model.decoder.') and 'class_embed' not in n_)
        else:
            raise SystemExit(f'未知 --freeze {args.freeze}')
        n_tr = n_fr = 0
        for n_, p in model.named_parameters():
            p.requires_grad = want(n_)
            n_tr += p.requires_grad
            n_fr += (not p.requires_grad)
        print(f'[freeze] {args.freeze}：可训 {n_tr} 个张量 / 冻结 {n_fr} 个')
    model.to(dev)

    # 断点续训：last.pt 里有 epoch/step/opt，把调度器也快进过去。
    # 只恢复权重不恢复 step 的话，余弦调度会从 0 重启，学习率被顶回峰值。
    resume_sd, start_ep, step0 = None, 0, 0
    if args.resume and os.path.exists(args.resume):
        resume_sd = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(resume_sd['model'])
        start_ep = int(resume_sd.get('epoch', -1)) + 1
        step0 = int(resume_sd.get('step', 0))
        print(f'[resume] {args.resume} → 从 epoch {start_ep}/{args.epochs} 继续（step {step0}）')

    # ---- 优化器：主干用小学习率
    backbone_params, head_params = [], []
    for n_, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if n_.startswith('model.backbone') else head_params).append(p)
    opt = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * args.backbone_lr_scale},
        {'params': head_params, 'lr': args.lr}], weight_decay=args.wd)
    print(f'[opt] 主干参数 {len(backbone_params)} 组（lr {args.lr * args.backbone_lr_scale:g}），'
          f'其余 {len(head_params)} 组（lr {args.lr:g}）')

    total_steps = math.ceil(args.samples_per_epoch / args.bs) * args.epochs
    if args.smoke:
        total_steps = 20

    def lr_at(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    if resume_sd is not None:
        # 恢复优化器动量，并把调度器快进到断点 step（LambdaLR 按 last_epoch 求 lr，
        # 主循环里 step 计数为 s 时生效的是 lr_at(s)）
        if 'opt' in resume_sd:
            opt.load_state_dict(resume_sd['opt'])
        sched.last_epoch = step0
        # 此时 _last_lr 还没重算（要等下一次 sched.step()），直接按公式报
        print(f'[resume] 优化器状态已恢复，调度器快进到 step {step0}，'
              f'下一批 lr {args.lr * lr_at(step0):.2e}')
    scaler = torch.amp.GradScaler('cuda', enabled=False)   # bf16 不需要

    sampler = RepeatFactorSampler([blist[i] for i in tr_idx], args.samples_per_epoch,
                                  args.target_per_class)
    tr_items = [items[i] for i in tr_idx]
    tr_boxes = [blist[i] for i in tr_idx]
    dv_items = [items[i] for i in dv_idx]
    dv_boxes = [blist[i] for i in dv_idx]
    ds = NuScenesDetDataset(tr_items, tr_boxes, processor, hflip=args.hflip)
    dl = DataLoader(ds, batch_size=args.bs, sampler=sampler, num_workers=args.workers,
                    collate_fn=lambda b: collate(b, processor), pin_memory=True,
                    persistent_workers=args.workers > 0, drop_last=True)

    log_f = open(os.path.join(args.out_dir, 'train.log'), 'a')
    # 续训时把 best 从 best.pt 捡回来，否则重启后的第一个 epoch 会无条件覆盖 hf_best
    best = -1.0
    _bp = os.path.join(args.out_dir, 'best.pt')
    if resume_sd is not None and os.path.exists(_bp):
        try:
            best = float(torch.load(_bp, map_location='cpu').get('score', -1.0))
            print(f'[resume] 沿用已有最佳关注类召回 {best:.4f}（不会被更差的 epoch 覆盖）')
        except Exception as e:
            print(f'[resume][warn] best.pt 读取失败，best 从 -1 重新开始：{e}')
    step = step0
    t0 = time.time()
    model.train()
    for ep in range(start_ep, args.epochs):
        sampler.set_epoch(ep)
        run = {}
        for it, (pv, pm, labels) in enumerate(dl):
            pv = pv.to(dev, non_blocking=True)
            pm = None if pm is None else pm.to(dev, non_blocking=True)
            labels = [{k: v.to(dev, non_blocking=True) for k, v in lb.items()} for lb in labels]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                out = model(pixel_values=pv, pixel_mask=pm, labels=labels)
                loss = out.loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], args.clip)
            opt.step()
            sched.step()
            step += 1
            for k, v in out.loss_dict.items():
                run[k] = run.get(k, 0.0) + float(v.detach())
            if (it + 1) % 50 == 0:
                el = time.time() - t0
                msg = (f'ep {ep} it {it + 1}/{len(dl)} step {step} '
                       f'lr {sched.get_last_lr()[-1]:.2e} loss {float(loss):.4f} '
                       f'{el / step:.2f}s/it')
                print(msg, flush=True)
                print(msg, file=log_f, flush=True)
            if args.smoke and step >= 20:
                break

        msg = f'== epoch {ep} 平均 ' + ' '.join(f'{k}={v / max(1, it + 1):.4f}' for k, v in run.items())
        print(msg, flush=True); print(msg, file=log_f, flush=True)

        ck = os.path.join(args.out_dir, 'last.pt')
        # 连优化器状态一起存（AdamW 两个动量 ≈ 340 MB），续训才能接着走
        torch.save({'model': model.state_dict(), 'config': model.config.to_dict(),
                    'epoch': ep, 'step': step, 'opt': opt.state_dict()}, ck)

        if (ep + 1) % args.eval_every == 0 or args.smoke:
            rows = evaluate_recall(model, processor, dv_items, dv_boxes, dev,
                                   max_items=args.dev_max_items)
            print(f'--- dev（train split 内 5% 留出，{len(dv_items)} 图）---', flush=True)
            print(f"{'cls':>4} {'类别':<13}{'GT':>7}{'TP':>7}{'FP':>7}{'召回':>9}{'精度':>9}", flush=True)
            for r in rows:
                line = (f"{r['cls']:>4} {r['name']:<13}{r['gt']:>7}{r['tp']:>7}{r['fp']:>7}"
                        f"{r['recall']:>9.3f}{r['precision']:>9.3f}")
                print(line, flush=True); print(line, file=log_f, flush=True)
            watch = [r for r in rows if r['cls'] in RARE_COCO]
            score = float(np.mean([r['recall'] for r in watch])) if watch else 0.0
            print(f'--- 关注类平均召回（bicycle/motorcycle/barrier/trafficcone）= {score:.4f} ---',
                  flush=True)
            print(f'--- 关注类平均召回 = {score:.4f} ---', file=log_f, flush=True)
            if score > best:
                best = score
                torch.save({'model': model.state_dict(), 'config': model.config.to_dict(),
                            'epoch': ep, 'step': step, 'score': score},
                           os.path.join(args.out_dir, 'best.pt'))
                # 同时存成 HF 目录，供 run_2d_detector.py --model <这个目录> 直接用
                hf_dir = os.path.join(args.out_dir, 'hf_best')
                model.save_pretrained(hf_dir)
                processor.save_pretrained(hf_dir)
                print(f'    -> best.pt / hf_best 已更新（{score:.4f}）', flush=True)
            with open(os.path.join(args.out_dir, f'dev_ep{ep}.json'), 'w') as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
        if args.smoke:
            break

    try:
        processor.save_pretrained(args.out_dir)
    except Exception as e:
        print(f'[warn] processor 保存失败：{e}')
    print(f'[done] 最佳关注类召回 {best:.4f}；输出 {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
