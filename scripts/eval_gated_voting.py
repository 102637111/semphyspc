#!/usr/bin/env python3
"""端到端 2D/3D 门控投票融合验证：验证 2D 罕见类检测能否提升 3D 分割罕见类精度。

流程：
  1. 加载 3D 分割模型，逐点推理 val 集，得到 logits/预测/置信度
  2. 把每个点投影到 CAM_FRONT，用「深度门控的 2D bbox」判断点是否落在罕见类框内
  3. 门控投票：对落在罕见类框内 且 3D 置信度 < θ 的点，用 2D 类别回填
  4. 对比 baseline vs fused 的罕见类 per-class IoU / mIoU

关键：DataLoader 返回 (name, lidar_token, (pts, sem))，token 用于查投影索引；
xyz 直接从 batch 的 pts[:, :3] 取出（in_channels=6），与模型预测的点一一对应，
不受 __getitem__ 随机下采样影响。

用法:
    python scripts/eval_gated_voting.py \
        --config cfgs/seg_nuscenes_full_16k.yaml \
        --ckpts experiments/seg_nuscenes_full_16k/cfgs/seg_nuscenes_full_16k/20260815-152228/ckpt-best.pth \
        --proj data/nuscenes_projection_index.npz \
        --bbox data/nuscenes_2d_bbox_cache.npz \
        --depth_margin 0.5
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

LIDARSEG_32 = [
    'noise', 'animal',
    'human.pedestrian.adult', 'human.pedestrian.child',
    'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer', 'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.barrier', 'movable_object.debris',
    'movable_object.pushable_pullable', 'movable_object.trafficcone',
    'static_object.bicycle_rack', 'vehicle.bicycle',
    'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car',
    'vehicle.construction', 'vehicle.emergency.ambulance',
    'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer',
    'vehicle.truck', 'flat.driveable_surface', 'flat.other',
    'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other',
    'static.vegetation', 'vehicle.ego',
]

RARE_CLASSES = {2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 21}


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
    - point2d_sem：每个点归属的罕见类 id（-1=不在任何深度门控 bbox 内）
    - point2d_score：命中框的检测 score（缓存有第 9 列时），未命中为 -1

    6 相机并集：点只要在任意相机的深度门控 bbox 内即归属该类。
    K_all/T_all 为 (6,3,3)/(6,4,4)，boxes 第 7 列为 cam_idx、第 8 列(若有)为 score。
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


def _override_mask(in_bbox, point2d, pred3d, conf3d, point2d_score, group_refine,
                   theta=None, score_thr=None, theta_class=None):
    """构造覆盖掩码（§4.15 融合规则改进的核心）。

    三种规则（后两种是 §4.15 提出的改进，相对「无条件覆盖」各补一条约束）：
      - 无条件（theta 非 None）：override = in_bbox & (conf3d < theta)
      - score 门控（score_thr 非 None）：再要求 score2d >= score_thr
      - 按类 θ（theta_class 非 None）：conf3d < theta_class[point2d]，忽略全局 theta

    注：§4.15 原文还提到「仅在 2D 与 3D 预测不一致时覆盖」，但该条件对输出是**空操作**
    ——`torch.where(override, point2d, pred3d)` 在 point2d==pred3d 时无论 override 真假
    结果相同，故不单独实现。真正的修复是 score 门控（拒低分误检）与按类 θ（3D 本就
    对的类少覆盖）。

    theta_class：长度 num_classes 的向量，point2d 已保证 >= 0（in_bbox）。
    """
    override = in_bbox
    if theta_class is not None:
        th = theta_class[point2d.clamp(min=0)]
        override = override & (conf3d < th)
    elif theta is not None:
        override = override & (conf3d < theta)
    if score_thr is not None:
        override = override & (point2d_score >= score_thr)
    if group_refine:
        ped_refine = override & (point2d == 2) & (pred3d >= 2) & (pred3d <= 8)
        override = override & ~ped_refine
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpts', required=True)
    ap.add_argument('--proj', required=True)
    ap.add_argument('--bbox', required=True)
    ap.add_argument('--depth_margin', type=float, default=0.5)
    ap.add_argument('--thetas', type=str, default='0.5,0.7,0.9,1.0',
                    help='门控置信度阈值列表（逗号分隔）')
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--group-refine', action='store_true',
                    help='COCO 检测器只有 person 粗粒度（映射到 adult=2）：当 2D 给出 2 且 3D 预测'
                         '本身落在行人组 2..8 内时，保留 3D 的子类，仅用 2D 判定"是不是人"。'
                         '仅对真实检测器缓存有意义；GT 缓存（含精细子类）不要开。')
    ap.add_argument('--tag', default='run', help='结果标签，写进 --out-json')
    ap.add_argument('--out-json', default=None, help='把 per-class IoU 与 mIoU 汇总存成 json')
    ap.add_argument('--score-thrs', type=str, default='',
                    help='2D 检测分数阈值列表（逗号分隔）。非空时额外输出 score 门控规则：'
                         'override = in_bbox & (score2d >= s) & (conf3d < θ)，对每个 s 扫一遍 θ。'
                         '需缓存含第 9 列 score。')
    ap.add_argument('--perclass-thetas', type=str, default='',
                    help='按类自适应 θ（JSON，lidar 类 id -> 3D 置信度阈值），如 \'{"2":0.95,"9":0.5}\'。'
                         '非空时额外输出 per-class 规则：override = in_bbox & (conf3d < θ_class[point2d])。'
                         '这是 §4.15 提出的「3D 本来就对的类少覆盖、3D 错的类多覆盖」的直接实现。')
    ap.add_argument('--perclass-default', type=float, default=1.0,
                    help='per-class 规则下未在 --perclass-thetas 中给出的类的默认 θ（默认 1.0=不覆盖）')
    args = ap.parse_args()

    thetas = [float(t) for t in args.thetas.split(',')]
    score_thrs = [float(s) for s in args.score_thrs.split(',')] if args.score_thrs else []
    perclass_map = json.loads(args.perclass_thetas) if args.perclass_thetas else {}

    args.local_rank = 0
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.resume = False
    args.experiment_path = '/tmp/eval_gated_exp'
    os.makedirs(args.experiment_path, exist_ok=True)
    args.log_name = os.path.basename(args.config).replace('.yaml', '')
    os.environ['LOCAL_RANK'] = '0'
    logger = get_root_logger(name='eval_gated')

    config = get_config(args, logger=logger)
    config.dataset.val.others.bs = 16
    num_classes = config.model.num_classes

    # 按类自适应 θ 向量（§4.15 改进）：默认值填 args.perclass_default，命中 map 的类用给定值
    theta_class = None
    if perclass_map:
        theta_class = torch.full((num_classes,), args.perclass_default, dtype=torch.float32).cuda()
        for k, v in perclass_map.items():
            theta_class[int(k)] = float(v)
        print_log(f'按类 θ：{dict(sorted({int(k): float(v) for k, v in perclass_map.items()}.items()))}'
                  f'，其余类默认 {args.perclass_default}', logger=logger)

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
    conf_score = {(s, t): torch.zeros(num_classes, num_classes, dtype=torch.int64).cuda()
                  for s in score_thrs for t in thetas}
    conf_perclass = torch.zeros(num_classes, num_classes, dtype=torch.int64).cuda()
    n_miss = 0  # token 缺失（投影索引/bbox 缓存）的样本数

    with torch.no_grad():
        for i, (_, stems, data) in enumerate(val_loader):
            pts = data[0].cuda()
            sem = data[1].cuda()
            ret = model(pts)
            logits = ret[0] if isinstance(ret, (tuple, list)) else ret
            probs = logits.softmax(-1)
            conf3d = probs.max(-1).values  # (B,N)
            pred3d = logits.argmax(-1)     # (B,N)

            B = pts.shape[0]
            # 逐样本算 point2d_sem + point2d_score（CPU）
            xyz_cpu = pts[:, :, :3].cpu().numpy().astype(np.float64)  # (B,N,3) xyz
            point2d = torch.full((B, pts.shape[1]), -1, dtype=torch.long).cuda()
            point2d_score = torch.full((B, pts.shape[1]), -1.0, dtype=torch.float32).cuda()
            for b in range(B):
                tok = stems[b]
                if tok not in proj_idx or tok not in bbox_idx:
                    n_miss += 1
                    continue
                K_all = proj['K'][proj_idx[tok]].astype(np.float64)  # (6,3,3)
                T_all = proj['T'][proj_idx[tok]].astype(np.float64)  # (6,4,4)
                bi = bbox_idx[tok]
                boxes = bbox_boxes[bbox_offsets[bi]:bbox_offsets[bi + 1]]
                sem2d, score2d = assign_2d_sem(xyz_cpu[b], K_all, T_all, boxes, args.depth_margin)
                point2d[b] = torch.from_numpy(sem2d).cuda()
                point2d_score[b] = torch.from_numpy(score2d).cuda()

            in_bbox = point2d >= 0
            # baseline
            conf_base += conf_matrix(pred3d, sem, num_classes)
            # 规则 1：无条件覆盖（现有基线），扫全局 θ
            for t in thetas:
                override = _override_mask(in_bbox, point2d, pred3d, conf3d, point2d_score,
                                          args.group_refine, theta=t)
                pred_fused = torch.where(override, point2d, pred3d)
                conf_fused[t] += conf_matrix(pred_fused, sem, num_classes)
            # 规则 2：score 门控，扫 (2D 分数阈值 × 全局 θ)
            for s in score_thrs:
                for t in thetas:
                    override = _override_mask(in_bbox, point2d, pred3d, conf3d, point2d_score,
                                              args.group_refine, theta=t, score_thr=s)
                    pred_fused = torch.where(override, point2d, pred3d)
                    conf_score[(s, t)] += conf_matrix(pred_fused, sem, num_classes)
            # 规则 3：按类自适应 θ（单条规则，无 θ 扫描）
            if theta_class is not None:
                override = _override_mask(in_bbox, point2d, pred3d, conf3d, point2d_score,
                                          args.group_refine, theta_class=theta_class)
                pred_fused = torch.where(override, point2d, pred3d)
                conf_perclass += conf_matrix(pred_fused, sem, num_classes)

    # 报告
    iou_base = per_class_iou(conf_base, num_classes)
    print_log('\n' + '=' * 92, logger=logger)
    print_log('罕见类 per-class IoU：baseline vs fused（门控投票）', logger=logger)
    print_log('=' * 92, logger=logger)
    hdr = f'{"class":<30} {"base":>7}'
    for t in thetas:
        hdr += f' {"θ=" + str(t):>9}'
    print_log(hdr, logger=logger)
    print_log('-' * 92, logger=logger)
    for c in sorted(RARE_CLASSES):
        line = f'{LIDARSEG_32[c]:<30} {iou_base[c]:>7.3f}'
        for t in thetas:
            line += f' {per_class_iou(conf_fused[t], num_classes)[c]:>9.3f}'
        print_log(line, logger=logger)
    print_log('-' * 92, logger=logger)

    # mIoU 汇总（罕见类 vs 全部）
    def _miou(ious):
        v = [x for x in ious if not np.isnan(x)]
        return float(np.mean(v)) if v else 0.0
    def _miou_rare(ious):
        v = [ious[c] for c in RARE_CLASSES if not np.isnan(ious[c])]
        return float(np.mean(v)) if v else 0.0

    miou_base = _miou(iou_base)
    line = f'{"mIoU(全部)":<30} {miou_base:>7.4f}'
    for t in thetas:
        line += f' {_miou(per_class_iou(conf_fused[t], num_classes)):>9.4f}'
    print_log(line, logger=logger)

    line = f'{"mIoU(罕见类)":<30} {_miou_rare(iou_base):>7.4f}'
    for t in thetas:
        line += f' {_miou_rare(per_class_iou(conf_fused[t], num_classes)):>9.4f}'
    print_log(line, logger=logger)
    print_log('=' * 92, logger=logger)
    print_log(f'token 缺失样本数 = {n_miss}', logger=logger)

    # 规则 2：score 门控汇总（对每个 2D 分数阈值扫 θ）
    if score_thrs:
        print_log('\n' + '=' * 92, logger=logger)
        print_log('score 门控（2D score 阈值 × θ）mIoU(罕见类) / mIoU(全部)：', logger=logger)
        hdr = f'{"score x theta":<12}'
        for t in thetas:
            hdr += f' {"θ=" + str(t):>18}'
        print_log(hdr, logger=logger)
        print_log('-' * 92, logger=logger)
        for s in score_thrs:
            line = f'{str(s):<10}'
            for t in thetas:
                iou_st = per_class_iou(conf_score[(s, t)], num_classes)
                line += f' {_miou_rare(iou_st):>8.4f}/{_miou(iou_st):<8.4f}'
            print_log(line, logger=logger)
        print_log('-' * 92, logger=logger)

    # 规则 3：按类自适应 θ 汇总（单条规则）
    if theta_class is not None:
        print_log('\n' + '=' * 92, logger=logger)
        print_log(f'按类自适应 θ（θ_class）mIoU(罕见类)={_miou_rare(per_class_iou(conf_perclass, num_classes)):.4f}  '
                  f'mIoU(全部)={_miou(per_class_iou(conf_perclass, num_classes)):.4f}', logger=logger)
        for c in sorted(RARE_CLASSES):
            iou_pc = per_class_iou(conf_perclass, num_classes)
            print_log(f'  {LIDARSEG_32[c]:<30} θ={theta_class[c].item():.2f}  IoU={iou_pc[c]:.4f}'
                      f'  (base {iou_base[c]:.4f})', logger=logger)
        print_log('=' * 92, logger=logger)

    if args.out_json:
        rec = {
            'tag': args.tag,
            'bbox_cache': os.path.basename(args.bbox),
            'ckpts': args.ckpts,
            'depth_margin': args.depth_margin,
            'group_refine': bool(args.group_refine),
            'n_miss': int(n_miss),
            'rare_classes': sorted(RARE_CLASSES),
            'base': {
                'per_class_iou': {LIDARSEG_32[c]: (None if np.isnan(iou_base[c]) else round(float(iou_base[c]), 4))
                                  for c in sorted(RARE_CLASSES)},
                'miou_all': round(_miou(iou_base), 4),
                'miou_rare': round(_miou_rare(iou_base), 4),
            },
            'fused': {},
            'fused_score': {},
            'fused_perclass': None,
        }
        for t in thetas:
            iou_t = per_class_iou(conf_fused[t], num_classes)
            rec['fused'][str(t)] = {
                'per_class_iou': {LIDARSEG_32[c]: (None if np.isnan(iou_t[c]) else round(float(iou_t[c]), 4))
                                  for c in sorted(RARE_CLASSES)},
                'miou_all': round(_miou(iou_t), 4),
                'miou_rare': round(_miou_rare(iou_t), 4),
            }
        for s in score_thrs:
            rec['fused_score'][str(s)] = {}
            for t in thetas:
                iou_st = per_class_iou(conf_score[(s, t)], num_classes)
                rec['fused_score'][str(s)][str(t)] = {
                    'per_class_iou': {LIDARSEG_32[c]: (None if np.isnan(iou_st[c]) else round(float(iou_st[c]), 4))
                                      for c in sorted(RARE_CLASSES)},
                    'miou_all': round(_miou(iou_st), 4),
                    'miou_rare': round(_miou_rare(iou_st), 4),
                }
        if theta_class is not None:
            iou_pc = per_class_iou(conf_perclass, num_classes)
            rec['fused_perclass'] = {
                'theta_class': {LIDARSEG_32[c]: round(float(theta_class[c].item()), 4)
                                for c in sorted(RARE_CLASSES)},
                'per_class_iou': {LIDARSEG_32[c]: (None if np.isnan(iou_pc[c]) else round(float(iou_pc[c]), 4))
                                  for c in sorted(RARE_CLASSES)},
                'miou_all': round(_miou(iou_pc), 4),
                'miou_rare': round(_miou_rare(iou_pc), 4),
            }
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        print_log(f'结果已写入 {args.out_json}', logger=logger)


if __name__ == '__main__':
    main()
