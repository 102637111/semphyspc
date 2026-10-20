#!/usr/bin/env python3
"""零样本跨数据集迁移评测：nuScenes 32 类模型 -> Semantic3D 8 类。

背景：
  RGB+XYZ 早期融合分割前置在 nuScenes(32 类) 上训练后，不经任何再训练直接在
  Semantic3D(8 类) 上预测，衡量 backbone 特征是否跨数据集迁移。

类别映射（nuScenes-lidarseg 32 类 -> Semantic3D 8 类，文档化、确定性）：
    noise(0)                      -> scanning artefacts(6)
    vehicle.car(17)               -> cars(7)
    flat.driveable_surface(24)    -> man-made terrain(0)
    flat.sidewalk(26)             -> man-made terrain(0)
    flat.other(25)                -> hard scape(5)
    flat.terrain(27)              -> natural terrain(1)
    static.manmade(28)            -> buildings(4)
    static.vegetation(30)         -> high vegetation(2)   [low veg(3) 无法区分，见备注]
    其余 24 类无对应 -> 映射为 -1（该点忽略）

指标：OA（映射后整体准确率）、逐类 recall、映射覆盖率（模型预测落入可映射类的比例）。
诚实备注：Semantic3D 的 high/low vegetation 在 nuScenes 只有单一 vegetation 类，
故 low veg(3) 的 recall 恒为 0，只报告 high veg；man-made terrain / hard scape /
buildings 之间也存在类别边界差异，属于跨数据集标签体系固有的不可比性。

用法:
    python scripts/eval_semantic3d_zeroshot.py \
        --config cfgs/seg_nuscenes_full_16k.yaml \
        --ckpt experiments/seg_nuscenes_full_16k/cfgs/singletask_sem/20260821-221520/ckpt-best.pth \
        --s3d_dir /mnt/data/datasets/Semantic3D/seg_npz_labeled/train
"""

import os, sys, glob, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import builder
from utils.config import get_config
from utils.logger import get_root_logger, print_log

S3D_NAMES = ['man-made terrain', 'natural terrain', 'high vegetation', 'low vegetation',
             'buildings', 'hard scape', 'scanning artefacts', 'cars']

# nuScenes 32 类预测 -> Semantic3D 8 类
NUSC_TO_S3D = {
    0: 6,    # noise -> scanning artefacts
    17: 7,   # vehicle.car -> cars
    24: 0,   # flat.driveable_surface -> man-made terrain
    25: 5,   # flat.other -> hard scape
    26: 0,   # flat.sidewalk -> man-made terrain
    27: 1,   # flat.terrain -> natural terrain
    28: 4,   # static.manmade -> buildings
    30: 2,   # static.vegetation -> high vegetation
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--s3d_dir', required=True)
    ap.add_argument('--npoints', type=int, default=16384)
    args = ap.parse_args()

    args.local_rank = 0
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.resume = False
    args.experiment_path = '/tmp/eval_s3d_zs'
    os.makedirs(args.experiment_path, exist_ok=True)
    args.log_name = 'eval_s3d_zeroshot'
    os.environ['LOCAL_RANK'] = '0'
    logger = get_root_logger(name='eval_s3d_zeroshot')

    config = get_config(args, logger=logger)
    config.model.num_classes = 32  # 预训练为 32 类
    model = builder.model_builder(config.model)
    builder.load_model(model, args.ckpt, logger=logger)
    model = model.cuda().eval()

    files = sorted(glob.glob(os.path.join(args.s3d_dir, '*.npz')))
    print_log(f'[zero-shot] {len(files)} 个 Semantic3D 场景', logger=logger)

    # 混淆矩阵在 8 类上（映射后）
    conf = np.zeros((8, 8), dtype=np.int64)
    n_valid = n_total = 0

    with torch.no_grad():
        for f in files:
            d = np.load(f)
            xyz = d['xyz'].astype(np.float32)
            rgb = d['rgb'].astype(np.float32)
            sem = d['sem'].astype(np.int64)
            N = xyz.shape[0]
            if N > args.npoints:
                idx = np.random.choice(N, args.npoints, replace=False)
                xyz, rgb, sem = xyz[idx], rgb[idx], sem[idx]
            pts = torch.from_numpy(np.concatenate([xyz, rgb], 1)).float().unsqueeze(0).cuda()
            logits = model(pts)[0]
            pred32 = logits.argmax(-1)[0].cpu().numpy().astype(np.int64)
            pred8 = np.array([NUSC_TO_S3D.get(int(p), -1) for p in pred32], dtype=np.int64)
            m = (pred8 >= 0) & (sem >= 0) & (sem < 8)
            n_total += sem.size
            n_valid += m.sum()
            for c in range(8):
                gt_m = m & (sem == c)
                if gt_m.any():
                    conf[c, :] += np.bincount(pred8[gt_m], minlength=8)

    print_log('\n' + '=' * 66, logger=logger)
    print_log('零样本迁移结果（nuScenes 32 类 -> Semantic3D 8 类）', logger=logger)
    print_log('=' * 66, logger=logger)
    print_log(f'{"类":<20s} {"recall":>8s} {"IoU":>7s} {"GT点数":>9s}', logger=logger)
    recalls, ious = [], []
    for c in range(8):
        gt = conf[c].sum()
        tp = conf[c, c]
        pred_sum = conf[:, c].sum()
        union = gt + pred_sum - tp
        rec = tp / gt if gt else float('nan')
        iou = tp / union if union else float('nan')
        if gt:
            recalls.append(rec)
            ious.append(iou if not np.isnan(iou) else 0.0)
        print_log(f'{S3D_NAMES[c]:<20s} {rec:>8.3f} {iou:>7.3f} {gt:>9,}', logger=logger)

    oa = np.trace(conf) / max(conf.sum(), 1)
    coverage = n_valid / max(n_total, 1)
    print_log('-' * 66, logger=logger)
    print_log(f'OA（映射后整体准确率）      = {oa:.4f}', logger=logger)
    print_log(f'平均 recall（有 GT 类）      = {np.mean(recalls):.4f}', logger=logger)
    print_log(f'mIoU（有 GT 类）            = {np.mean(ious):.4f}', logger=logger)
    print_log(f'映射覆盖率（可映射点占比）   = {coverage:.4f}', logger=logger)
    print_log('=' * 66, logger=logger)
    print_log('备注: low vegetation(3) 与 high vegetation(2) 在 nuScenes 中同属单一 '
             'vegetation 类，故 low veg recall=0；跨数据集标签边界固有不可比。', logger=logger)


if __name__ == '__main__':
    main()
