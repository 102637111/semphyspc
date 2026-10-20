#!/usr/bin/env python3
"""12 类体系 2D/3D 门控投票融合：验证 2D person 检测能否降低致命类安全误判（critical_rate）。

动机：合并类方向三腿（merged_v1 / coarse_agg / 跨域迁移）都证明 critical_rate 没降
（5.30% / 5.18% / 5.15%）。本脚本把「3D 误判为可撞类（frangible/vegetation）的 person 点」
用 2D person 检测回填成 person(2)，直接攻击论文核心安全指标。

与 eval_gated_voting.py（32 类）的差异：
  - 12 类体系：bbox 缓存的 sem_id（32 类）经 COARSE_MAP 映射到 12 类
  - 罕见类集合换成 {person(2), two_wheeler(6), frangible(10)}
  - 新增安全聚合（critical_rate/severe_rate/safe_rate）对比 baseline vs fused
  - val bs 默认 4（ng6144 需 bs=4 防 OOM）
  - 去掉 group_refine（行人子类已合并成 person，无意义）与 score 门控（缓存无 score 列）

用法:
    python3 scripts/eval_gated_voting_12c.py \
        --config cfgs/seg_16k_ng6144_g050_merged.yaml \
        --ckpts experiments/seg_16k_ng6144_g050_merged/cfgs/merged_v1/<ts>/ckpt-best.pth \
        --proj data/nuscenes_projection_index.npz \
        --bbox data/nuscenes_2d_bbox_cache_rtdetr_s3_val_size.npz \
        --bs 4
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

IMG_W, IMG_H = 1600, 900

# 合并相似类后的 12 类（物理交互粗粒度，索引顺序，与 eval_seg_full.py 一致）
LIDARSEG_COARSE = [
    'noise', 'animal', 'person', 'car', 'bus', 'truck',
    'two_wheeler', 'ground', 'manmade', 'vegetation', 'frangible', 'ego',
]

# 32 → 12 映射（与 NuscenesSegFull16kMerged.yaml 的 CLASS_MAP 一致）。
COARSE_MAP = [0, 1, 2, 2, 2, 2, 2, 2, 2, 10, 10, 10, 10, 8, 6, 4, 4, 3, 5, 3, 3, 6, 5, 5, 7, 7, 7, 7, 8, 8, 9, 11]

# 12 类体系下门控投票关注的类：RT-DETR 能检测类的映射
# person(2, 致命) / two_wheeler(6, bicycle/motorcycle) / frangible(10, barrier/trafficcone)
RARE_CLASSES = {2, 6, 10}

# 安全聚合（选择性碰撞的核心）：与 eval_seg_full.py 的 12 类口径一致
LETHAL = [1, 2]          # animal + person（致命）
FRANGIBLE = [10]         # frangible（护栏/碎屑/锥桶）
DEFORMABLE = [9]         # vegetation
PASSABLE = [7]           # ground（路面/人行道/地形）
COLLIDABLE = FRANGIBLE + DEFORMABLE  # 决策模块推荐撞击的集合


def project(xyz, K, T):
    """xyz (N,3) lidar -> (u, v, depth, valid)。"""
    N = len(xyz)
    pts_h = np.concatenate([xyz, np.ones((N, 1))], axis=1).astype(np.float64)
    pts_cam = (T @ pts_h.T).T  # N,4
    depth = pts_cam[:, 2]
    uv = (K @ pts_cam[:, :3].T)  # 2,N
    u = uv[0] / uv[2]
    v = uv[1] / uv[2]
    valid = (depth > 0) & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    return u, v, depth, valid


def assign_2d_sem(xyz, K_all, T_all, boxes, depth_margin):
    """返回 (point2d_sem, point2d_score)，形状 (N,)：
    - point2d_sem：每个点归属的罕见类 id（**32 类**，-1=不在任何深度门控 bbox 内）
    - point2d_score：命中框的检测 score（缓存有第 9 列时），未命中为 -1
    """
    N = len(xyz)
    out = np.full(N, -1, dtype=np.int64)
    out_score = np.full(N, -1.0, dtype=np.float32)
    for cam_idx in range(K_all.shape[0]):
        u, v, depth, valid = project(xyz, K_all[cam_idx], T_all[cam_idx])
        for row in boxes:
            if int(row[7]) != cam_idx:
                continue
            x, y, w, h, sem_id = row[0], row[1], row[2], row[3], int(row[4])
            in_box = valid & (u >= x) & (u < x + w) & (v >= y) & (v < y + h)
            if row.shape[0] >= 7:
                dmin, dmax = row[5], row[6]
                in_box &= (depth >= dmin - depth_margin) & (depth <= dmax + depth_margin)
            if in_box.any():
                out[in_box] = sem_id
                if row.shape[0] >= 9:
                    out_score[in_box] = float(row[8])
    return out, out_score


def _override_mask(in_bbox, point2d, pred3d, conf3d, theta=None, theta_class=None):
    """构造覆盖掩码：override = in_bbox & (conf3d < θ)。
    12 类下无 group_refine（行人子类已合并成 person=2）。
    """
    override = in_bbox
    if theta_class is not None:
        th = theta_class[point2d.clamp(min=0)]
        override = override & (conf3d < th)
    elif theta is not None:
        override = override & (conf3d < theta)
    return override


def conf_matrix(pred, sem, num_classes):
    """pred/sem (N,) 长整型 -> (num_classes, num_classes) 混淆矩阵（行=GT, 列=pred）。"""
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
    """从 12 类混淆矩阵算致命类安全聚合（与 eval_seg_full.py 口径一致）。"""
    cm = conf.cpu().numpy()
    lethal_gt = int(cm[LETHAL].sum())
    agg = {
        'lethal_gt_points': lethal_gt,
        'lethal_to_lethal': int(cm[np.ix_(LETHAL, LETHAL)].sum()),
        'lethal_to_collidable': int(cm[np.ix_(LETHAL, COLLIDABLE)].sum()),
        'lethal_to_frangible': int(cm[np.ix_(LETHAL, FRANGIBLE)].sum()),
        'lethal_to_deformable': int(cm[np.ix_(LETHAL, DEFORMABLE)].sum()),
        'lethal_to_passable': int(cm[np.ix_(LETHAL, PASSABLE)].sum()),
        'lethal_to_hard': int(cm[LETHAL].sum() - cm[np.ix_(LETHAL, LETHAL)].sum()
                              - cm[np.ix_(LETHAL, COLLIDABLE)].sum()
                              - cm[np.ix_(LETHAL, PASSABLE)].sum()
                              - cm[np.ix_(LETHAL, [0])].sum()),
    }
    if lethal_gt:
        n = float(lethal_gt)
        agg['critical_rate'] = round(agg['lethal_to_collidable'] / n, 6)
        agg['severe_rate'] = round(agg['lethal_to_passable'] / n, 6)
        agg['safe_rate'] = round((agg['lethal_to_lethal'] + agg['lethal_to_hard']) / n, 6)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpts', required=True)
    ap.add_argument('--proj', required=True)
    ap.add_argument('--bbox', required=True)
    ap.add_argument('--depth_margin', type=float, default=0.5)
    ap.add_argument('--thetas', type=str, default='0.5,0.7,0.9,1.0',
                    help='门控置信度阈值列表（逗号分隔）')
    ap.add_argument('--bs', type=int, default=4, help='val 批大小（ng6144 需 4 防 OOM）')
    ap.add_argument('--max-frames', type=int, default=0, help='只评测前 N 帧（0=全部），用于冒烟测试')
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--tag', default='run', help='结果标签，写进 --out-json')
    ap.add_argument('--out-json', default=None, help='把 per-class IoU / 安全聚合存成 json')
    args = ap.parse_args()

    thetas = [float(t) for t in args.thetas.split(',')]

    args.local_rank = 0
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.resume = False
    args.experiment_path = '/tmp/eval_gated_12c_exp'
    os.makedirs(args.experiment_path, exist_ok=True)
    args.log_name = os.path.basename(args.config).replace('.yaml', '')
    os.environ['LOCAL_RANK'] = '0'
    logger = get_root_logger(name='eval_gated_12c')

    config = get_config(args, logger=logger)
    config.dataset.val.others.bs = args.bs
    num_classes = config.model.num_classes
    assert num_classes == 12, f'本脚本只用于 12 类体系，当前 config num_classes={num_classes}'

    coarse_map_t = torch.tensor(COARSE_MAP, dtype=torch.long).cuda()

    # 加载投影索引 + 2D bbox 缓存
    proj = np.load(args.proj)
    bbox = np.load(args.bbox)
    proj_idx = {t: i for i, t in enumerate(proj['tokens'])}
    bbox_idx = {t: i for i, t in enumerate(bbox['tokens'])}
    bbox_offsets = bbox['offsets']
    bbox_boxes = bbox['boxes']

    print_log(f'加载 checkpoint: {args.ckpts}', logger=logger)
    _, val_loader = builder.dataset_builder(args, config.dataset.val)

    model = builder.model_builder(config.model)
    builder.load_model(model, args.ckpts, logger=logger)
    model = nn.DataParallel(model).cuda()
    model.eval()

    conf_base = torch.zeros(num_classes, num_classes, dtype=torch.int64).cuda()
    conf_fused = {t: torch.zeros(num_classes, num_classes, dtype=torch.int64).cuda()
                  for t in thetas}
    n_miss = 0  # token 缺失的样本数

    with torch.no_grad():
        for i, (_, stems, data) in enumerate(val_loader):
            if args.max_frames and i >= args.max_frames:
                break
            pts = data[0].cuda()
            sem = data[1].cuda()
            ret = model(pts)
            logits = ret[0] if isinstance(ret, (tuple, list)) else ret
            probs = logits.softmax(-1)
            conf3d = probs.max(-1).values  # (B,N)
            pred3d = logits.argmax(-1)     # (B,N)

            B = pts.shape[0]
            xyz_cpu = pts[:, :, :3].cpu().numpy().astype(np.float64)  # (B,N,3) xyz
            point2d = torch.full((B, pts.shape[1]), -1, dtype=torch.long).cuda()
            for b in range(B):
                tok = stems[b]
                if tok not in proj_idx or tok not in bbox_idx:
                    n_miss += 1
                    continue
                K_all = proj['K'][proj_idx[tok]].astype(np.float64)  # (6,3,3)
                T_all = proj['T'][proj_idx[tok]].astype(np.float64)  # (6,4,4)
                bi = bbox_idx[tok]
                boxes = bbox_boxes[bbox_offsets[bi]:bbox_offsets[bi + 1]]
                sem2d, _ = assign_2d_sem(xyz_cpu[b], K_all, T_all, boxes, args.depth_margin)
                point2d[b] = torch.from_numpy(sem2d).cuda()

            # 32 类 sem_id -> 12 类（-1 保持）
            point2d_12 = torch.where(point2d >= 0, coarse_map_t[point2d.clamp(min=0)], point2d)

            in_bbox = point2d >= 0
            conf_base += conf_matrix(pred3d, sem, num_classes)
            for t in thetas:
                override = _override_mask(in_bbox, point2d_12, pred3d, conf3d, theta=t)
                pred_fused = torch.where(override, point2d_12, pred3d)
                conf_fused[t] += conf_matrix(pred_fused, sem, num_classes)

    # 报告
    iou_base = per_class_iou(conf_base, num_classes)
    agg_base = safety_agg(conf_base)

    print_log('\n' + '=' * 92, logger=logger)
    print_log('12 类门控投票：baseline vs fused（关注类 IoU + 安全聚合）', logger=logger)
    print_log('=' * 92, logger=logger)
    hdr = f'{"class":<16} {"base":>7}'
    for t in thetas:
        hdr += f' {"θ=" + str(t):>8}'
    print_log(hdr, logger=logger)
    print_log('-' * 92, logger=logger)
    for c in sorted(RARE_CLASSES):
        line = f'{LIDARSEG_COARSE[c]:<16} {iou_base[c]:>7.3f}'
        for t in thetas:
            line += f' {per_class_iou(conf_fused[t], num_classes)[c]:>8.3f}'
        print_log(line, logger=logger)
    print_log('-' * 92, logger=logger)

    def _miou(ious):
        v = [x for x in ious if not np.isnan(x)]
        return float(np.mean(v)) if v else 0.0

    line = f'{"mIoU(全部)":<16} {_miou(iou_base):>7.4f}'
    for t in thetas:
        line += f' {_miou(per_class_iou(conf_fused[t], num_classes)):>8.4f}'
    print_log(line, logger=logger)
    print_log('=' * 92, logger=logger)

    # 安全聚合对比
    print_log('\n' + '=' * 74, logger=logger)
    print_log('安全聚合（核心指标）：致命类 GT 被判到哪一级', logger=logger)
    print_log('=' * 74, logger=logger)
    print_log(f"致命类（animal+person）GT 共 {agg_base['lethal_gt_points']:,} 点", logger=logger)

    def _print_safety(tag, agg):
        print_log(f'  [{tag}] critical = {agg["critical_rate"] * 100:.3f}%　'
                  f'severe = {agg["severe_rate"] * 100:.3f}%　'
                  f'safe = {agg["safe_rate"] * 100:.3f}%', logger=logger)

    _print_safety('base  ', agg_base)
    for t in thetas:
        _print_safety(f'θ={t}', safety_agg(conf_fused[t]))
    print_log('=' * 74, logger=logger)
    print_log(f'token 缺失样本数 = {n_miss}', logger=logger)

    if args.out_json:
        rec = {
            'tag': args.tag,
            'bbox_cache': os.path.basename(args.bbox),
            'ckpts': args.ckpts,
            'depth_margin': args.depth_margin,
            'n_miss': int(n_miss),
            'focus_classes': {str(c): LIDARSEG_COARSE[c] for c in sorted(RARE_CLASSES)},
            'base': {
                'per_class_iou': {LIDARSEG_COARSE[c]: (None if np.isnan(iou_base[c]) else round(float(iou_base[c]), 4))
                                  for c in sorted(RARE_CLASSES)},
                'miou_all': round(_miou(iou_base), 4),
                'safety': agg_base,
            },
            'fused': {},
        }
        for t in thetas:
            iou_t = per_class_iou(conf_fused[t], num_classes)
            rec['fused'][str(t)] = {
                'per_class_iou': {LIDARSEG_COARSE[c]: (None if np.isnan(iou_t[c]) else round(float(iou_t[c]), 4))
                                  for c in sorted(RARE_CLASSES)},
                'miou_all': round(_miou(iou_t), 4),
                'safety': safety_agg(conf_fused[t]),
            }
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        print_log(f'结果已写入 {args.out_json}', logger=logger)


if __name__ == '__main__':
    main()
