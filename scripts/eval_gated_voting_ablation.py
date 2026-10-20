#!/usr/bin/env python3
"""C⑥ 检测辅助消融梯度：拆解 2D 检测回填的每个组件，量化「深度门控」与「置信度门控」的增量贡献。

动机：gated_voting 主实验（eval_gated_voting_12c.py）验证了「深度门控 + 置信度门控」完整融合能降
critical_rate（5.30%→3.80%），但没拆开「到底是哪个组件在起作用」。本脚本做四路消融梯度：

  1. base        —— 无回填（纯 3D 分割预测，上界参照）
  2. direct      —— 直接回填：bbox 2D 框内所有投影点无条件覆盖为检测类别
                     （无深度门控 + 无置信度门控，最激进，暴露「没门控」的代价）
  3. conf_gate   —— 置信度门控：bbox 内 + 3D 置信度 < θ 才回填（无深度门控）
                     （隔离「置信度门控」的增量：防低置信度点被误覆盖）
  4. conf_depth  —— 置信度 + 深度门控：完整融合（现有主实验做法）
                     （隔离「深度门控」的增量：防 bbox 2D 框内不同深度的背景点被误回填）

输出四种方式的关注类 IoU（person/two_wheeler/frangible）+ 安全聚合（critical/severe/safe），
以及「相对上一级」的增量，直接服务论文「检测辅助组件消融」小节。

用法（与 eval_gated_voting_12c.py 相同，固定 θ=0.9 最优安全点）:
    python3 scripts/eval_gated_voting_ablation.py \
        --config cfgs/seg_16k_ng6144_g050_merged.yaml \
        --ckpts experiments/seg_16k_ng6144_g050_merged/cfgs/merged_pretrain_ft/20260927-070312/ckpt-best.pth \
        --proj data/nuscenes_projection_index.npz \
        --bbox data/nuscenes_2d_bbox_cache_rtdetr_s3_val_size.npz \
        --bs 4 --out-json results/gated_voting_ablation_merged_pretrain_ft.json
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

LIDARSEG_COARSE = [
    'noise', 'animal', 'person', 'car', 'bus', 'truck',
    'two_wheeler', 'ground', 'manmade', 'vegetation', 'frangible', 'ego',
]

COARSE_MAP = [0, 1, 2, 2, 2, 2, 2, 2, 2, 10, 10, 10, 10, 8, 6, 4, 4, 3, 5, 3, 3, 6, 5, 5, 7, 7, 7, 7, 8, 8, 9, 11]

RARE_CLASSES = {2, 6, 10}   # person / two_wheeler / frangible

LETHAL = [1, 2]
FRANGIBLE = [10]
DEFORMABLE = [9]
PASSABLE = [7]
COLLIDABLE = FRANGIBLE + DEFORMABLE


def project(xyz, K, T):
    N = len(xyz)
    pts_h = np.concatenate([xyz, np.ones((N, 1))], axis=1).astype(np.float64)
    pts_cam = (T @ pts_h.T).T
    depth = pts_cam[:, 2]
    uv = (K @ pts_cam[:, :3].T)
    u = uv[0] / uv[2]
    v = uv[1] / uv[2]
    valid = (depth > 0) & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    return u, v, depth, valid


def assign_2d_sem_both(xyz, K_all, T_all, boxes, depth_margin):
    """一次投影同时算两路回填掩码，避免重复 project：
    返回 (point2d_nodepth, point2d_depth)，形状 (N,)：
      - point2d_nodepth：bbox 2D 框内所有投影点 → 检测类（32 类 id，无深度门控）
      - point2d_depth  ：仅深度落在 bbox [dmin-margin, dmax+margin] 的点 → 检测类
    二者均为 -1=不在任何框内。
    """
    N = len(xyz)
    out_nd = np.full(N, -1, dtype=np.int64)
    out_d = np.full(N, -1, dtype=np.int64)
    for cam_idx in range(K_all.shape[0]):
        u, v, depth, valid = project(xyz, K_all[cam_idx], T_all[cam_idx])
        for row in boxes:
            if int(row[7]) != cam_idx:
                continue
            x, y, w, h, sem_id = row[0], row[1], row[2], row[3], int(row[4])
            in_box = valid & (u >= x) & (u < x + w) & (v >= y) & (v < y + h)
            if in_box.any():
                out_nd[in_box] = sem_id
            if row.shape[0] >= 7:
                dmin, dmax = row[5], row[6]
                in_box_d = in_box & (depth >= dmin - depth_margin) & (depth <= dmax + depth_margin)
                if in_box_d.any():
                    out_d[in_box_d] = sem_id
    return out_nd, out_d


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
    agg = {
        'lethal_gt_points': lethal_gt,
        'lethal_to_lethal': int(cm[np.ix_(LETHAL, LETHAL)].sum()),
        'lethal_to_collidable': int(cm[np.ix_(LETHAL, COLLIDABLE)].sum()),
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
    ap.add_argument('--theta', type=float, default=0.9, help='置信度门控阈值（默认 0.9 最优安全点）')
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--tag', default='c6_gated_voting_ablation', help='结果标签')
    ap.add_argument('--out-json', default=None)
    args = ap.parse_args()

    args.local_rank = 0
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.resume = False
    args.experiment_path = '/tmp/eval_gv_ablation_exp'
    os.makedirs(args.experiment_path, exist_ok=True)
    args.log_name = os.path.basename(args.config).replace('.yaml', '')
    os.environ['LOCAL_RANK'] = '0'
    logger = get_root_logger(name='eval_gv_ablation')

    config = get_config(args, logger=logger)
    config.dataset.val.others.bs = args.bs
    num_classes = config.model.num_classes
    assert num_classes == 12, f'本脚本只用于 12 类体系，当前 num_classes={num_classes}'

    coarse_map_t = torch.tensor(COARSE_MAP, dtype=torch.long).cuda()
    theta = args.theta

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

    # 四路消融的混淆矩阵：base / direct / conf_gate / conf_depth
    conf = {k: torch.zeros(num_classes, num_classes, dtype=torch.int64).cuda()
            for k in ['base', 'direct', 'conf_gate', 'conf_depth']}
    n_miss = 0

    with torch.no_grad():
        for i, (_, stems, data) in enumerate(val_loader):
            if args.max_frames and i >= args.max_frames:
                break
            pts = data[0].cuda()
            sem = data[1].cuda()
            ret = model(pts)
            logits = ret[0] if isinstance(ret, (tuple, list)) else ret
            probs = logits.softmax(-1)
            conf3d = probs.max(-1).values
            pred3d = logits.argmax(-1)

            B = pts.shape[0]
            xyz_cpu = pts[:, :, :3].cpu().numpy().astype(np.float64)
            p2d_nd = torch.full((B, pts.shape[1]), -1, dtype=torch.long).cuda()
            p2d_d = torch.full((B, pts.shape[1]), -1, dtype=torch.long).cuda()
            for b in range(B):
                tok = stems[b]
                if tok not in proj_idx or tok not in bbox_idx:
                    n_miss += 1
                    continue
                K_all = proj['K'][proj_idx[tok]].astype(np.float64)
                T_all = proj['T'][proj_idx[tok]].astype(np.float64)
                bi = bbox_idx[tok]
                boxes = bbox_boxes[bbox_offsets[bi]:bbox_offsets[bi + 1]]
                nd, d = assign_2d_sem_both(xyz_cpu[b], K_all, T_all, boxes, args.depth_margin)
                p2d_nd[b] = torch.from_numpy(nd).cuda()
                p2d_d[b] = torch.from_numpy(d).cuda()

            # 32 类 -> 12 类
            p2d_nd_12 = torch.where(p2d_nd >= 0, coarse_map_t[p2d_nd.clamp(min=0)], p2d_nd)
            p2d_d_12 = torch.where(p2d_d >= 0, coarse_map_t[p2d_d.clamp(min=0)], p2d_d)
            in_nd = p2d_nd >= 0
            in_d = p2d_d >= 0

            conf['base'] += conf_matrix(pred3d, sem, num_classes)
            # direct：无门控直接回填（无深度门控）
            conf['direct'] += conf_matrix(torch.where(in_nd, p2d_nd_12, pred3d), sem, num_classes)
            # conf_gate：置信度门控（无深度门控）
            ov_c = in_nd & (conf3d < theta)
            conf['conf_gate'] += conf_matrix(torch.where(ov_c, p2d_nd_12, pred3d), sem, num_classes)
            # conf_depth：置信度 + 深度门控（完整）
            ov_cd = in_d & (conf3d < theta)
            conf['conf_depth'] += conf_matrix(torch.where(ov_cd, p2d_d_12, pred3d), sem, num_classes)

    order = ['base', 'direct', 'conf_gate', 'conf_depth']
    names = ['无回填', '直接回填', '置信度门控', '置信度+深度门控']

    print_log('\n' + '=' * 96, logger=logger)
    print_log(f'C⑥ 检测辅助消融梯度（θ={theta}）：四种回填方式', logger=logger)
    print_log('=' * 96, logger=logger)

    # 关注类 IoU
    print_log(f'{"方式":<18} {"person":>8} {"two_wheel":>10} {"frangible":>10} {"mIoU(全部)":>11}', logger=logger)
    print_log('-' * 96, logger=logger)
    for k, nm in zip(order, names):
        iou = per_class_iou(conf[k], num_classes)
        v = [x for x in iou if not np.isnan(x)]
        miou = float(np.mean(v)) if v else 0.0
        print_log(f'{nm:<18} {iou[2]:>8.3f} {iou[6]:>10.3f} {iou[10]:>10.3f} {miou:>11.4f}', logger=logger)
    print_log('-' * 96, logger=logger)

    # 安全聚合
    print_log('\n' + '=' * 78, logger=logger)
    print_log('安全聚合（核心指标）：致命类 GT 被判到哪一级', logger=logger)
    print_log('=' * 78, logger=logger)
    agg_base = safety_agg(conf['base'])
    print_log(f"致命类（animal+person）GT 共 {agg_base['lethal_gt_points']:,} 点", logger=logger)
    print_log(f'{"方式":<18} {"critical%":>10} {"severe%":>9} {"safe%":>8}', logger=logger)
    print_log('-' * 78, logger=logger)
    for k, nm in zip(order, names):
        a = safety_agg(conf[k])
        print_log(f'{nm:<18} {a["critical_rate"] * 100:>9.3f}% {a["severe_rate"] * 100:>8.3f}% {a["safe_rate"] * 100:>7.3f}%', logger=logger)
    print_log('=' * 78, logger=logger)
    print_log(f'token 缺失样本数 = {n_miss}', logger=logger)

    if args.out_json:
        rec = {
            'tag': args.tag,
            'config': args.config,
            'ckpts': args.ckpts,
            'bbox_cache': os.path.basename(args.bbox),
            'depth_margin': args.depth_margin,
            'theta': theta,
            'n_miss': int(n_miss),
            'focus_classes': {str(c): LIDARSEG_COARSE[c] for c in sorted(RARE_CLASSES)},
            'modes': {},
        }
        for k, nm in zip(order, names):
            iou = per_class_iou(conf[k], num_classes)
            v = [x for x in iou if not np.isnan(x)]
            rec['modes'][k] = {
                'name': nm,
                'per_class_iou': {LIDARSEG_COARSE[c]: (None if np.isnan(iou[c]) else round(float(iou[c]), 4))
                                  for c in sorted(RARE_CLASSES)},
                'miou_all': round(float(np.mean(v)) if v else 0.0, 4),
                'safety': safety_agg(conf[k]),
            }
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        print_log(f'结果已写入 {args.out_json}', logger=logger)


if __name__ == '__main__':
    main()
