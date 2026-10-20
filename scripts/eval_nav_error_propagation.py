#!/usr/bin/env python3
"""B② 感知误差传递：把分割预测误差翻译成「导航代价」层面的危险混淆。

动机（导航证据链补强项 B②）：导航用分割预测语义生成代价图，但分割 mIoU 只有 ~0.62。
本脚本量化「感知误差 → 导航代价误差」的传递：三组语义（GT 零误差上界 / 3D 分割预测 /
3D+2D 检测门控投票融合）各自映射到导航代价五档（方案 a 口径），输出完整代价混淆矩阵，
聚焦「致命类（lethal）被漏检成可穿越软代价/自由地面」的危险混淆率，以及 2D 检测辅助能救回多少。

与 eval_gated_voting_12c.py 的关系：后者验证「门控投票能否降 critical」，输出 safety 三标量；
本脚本是它的「导航化」扩展——额外输出完整 5x5 导航代价混淆矩阵 + hard 漏检率，直接服务论文
「感知误差传递」小节。推理/融合逻辑复用同一套（模型 + 投影 + bbox 深度门控 + 置信度门控）。

导航代价五档（方案 a：lethal 与 hard 都标 254 绝对障碍，仅 frangible/deformable 保留软代价）：
  lethal(254) = animal(1) + person(2)            —— 碰撞致命，绝对障碍
  hard(254)   = car/bus/truck/two_wheeler/manmade/ego —— 实体不可穿越，绝对障碍
  soft(46)    = vegetation(9) + frangible(10)    —— 可穿越软代价（评估尺子 46）
  free(0)     = ground(7)                        —— 完全可穿越
  noise       = noise(0)                         —— 无语义标注，单独一档不参与安全聚合

用法:
    python3 scripts/eval_nav_error_propagation.py \
        --config cfgs/seg_16k_ng6144_g050_merged.yaml \
        --ckpts experiments/seg_16k_ng6144_g050_merged/cfgs/merged_pretrain_ft/20260927-070312/ckpt-best.pth \
        --proj data/nuscenes_projection_index.npz \
        --bbox data/nuscenes_2d_bbox_cache_rtdetr_s3_val_size.npz \
        --bs 4 --out-json results/nav_error_propagation_merged_pretrain_ft.json
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

# 合并相似类后的 12 类（与 eval_seg_full.py / eval_gated_voting_12c.py 一致）
LIDARSEG_COARSE = [
    'noise', 'animal', 'person', 'car', 'bus', 'truck',
    'two_wheeler', 'ground', 'manmade', 'vegetation', 'frangible', 'ego',
]

# 32 → 12 映射（与 NuscenesSegFull16kMerged.yaml 的 CLASS_MAP 一致）。
COARSE_MAP = [0, 1, 2, 2, 2, 2, 2, 2, 2, 10, 10, 10, 10, 8, 6, 4, 4, 3, 5, 3, 3, 6, 5, 5, 7, 7, 7, 7, 8, 8, 9, 11]

# 门控投票关注的类：RT-DETR 能检测类的映射（person/two_wheeler/frangible）
RARE_CLASSES = {2, 6, 10}

# ===== 导航代价五档（方案 a 口径）=====
# 顺序即混淆矩阵的行/列顺序；每档是 12 类 coarse id 集合。
NAV_ORDER = ['lethal', 'hard', 'soft', 'free', 'noise']
NAV_GROUPS = {
    'lethal': [1, 2],              # animal + person —— 致命
    'hard':   [3, 4, 5, 6, 8, 11],  # car/bus/truck/two_wheeler/manmade/ego —— 实体
    'soft':   [9, 10],             # vegetation + frangible —— 可穿越软代价
    'free':   [7],                 # ground —— 自由地面
    'noise':  [0],                 # 未标注
}
NAV_COST_LABEL = {'lethal': 254, 'hard': 254, 'soft': 46, 'free': 0, 'noise': None}


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
    """覆盖掩码：override = in_bbox & (conf3d < θ)。"""
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


def nav_conf_from(conf12):
    """12x12 混淆矩阵 -> 5x5 导航代价混淆矩阵（行=GT, 列=pred，绝对计数）。"""
    cm = conf12.cpu().numpy()
    n = len(NAV_ORDER)
    out = np.zeros((n, n), dtype=np.int64)
    for gi, g in enumerate(NAV_ORDER):
        for gj, h in enumerate(NAV_ORDER):
            out[gi, gj] = int(cm[np.ix_(NAV_GROUPS[g], NAV_GROUPS[h])].sum())
    return out


def row_norm(m):
    """按行归一化（每行 GT 总数 -> 比例），空行置 0 避免 NaN。"""
    m = m.astype(np.float64)
    r = m.sum(axis=1, keepdims=True)
    r[r == 0] = 1.0
    return m / r


def nav_summary(nav_conf):
    """从 5x5 导航代价混淆矩阵提取聚焦指标。

    返回 dict：
      - lethal：致命类（animal+person）漏检成可穿越软代价/自由地面的危险混淆率
        critical_rate = lethal→soft（行人被当成植被/护栏，可穿越）
        severe_rate   = lethal→free（行人被当成地面，最危险）
        safe_rate     = lethal→lethal/hard（仍标绝对障碍，安全）
      - hard：实体类（车/墙/建筑）漏检率（同样关系导航安全）
    """
    m = nav_conf.astype(np.float64)
    order = NAV_ORDER
    li, hi, si, fi, ni = (order.index(g) for g in ['lethal', 'hard', 'soft', 'free', 'noise'])
    out = {}
    for key, gi in [('lethal', li), ('hard', hi)]:
        total = m[gi].sum()
        rec = {
            'gt_points': int(total),
            'to_lethal': int(m[gi, li]), 'to_hard': int(m[gi, hi]),
            'to_soft': int(m[gi, si]), 'to_free': int(m[gi, fi]), 'to_noise': int(m[gi, ni]),
        }
        if total > 0:
            rec['critical_rate'] = round(float(m[gi, si]) / total, 6)
            rec['severe_rate'] = round(float(m[gi, fi]) / total, 6)
            rec['safe_rate'] = round(float((m[gi, li] + m[gi, hi])) / total, 6)
        else:
            rec['critical_rate'] = 0.0
            rec['severe_rate'] = 0.0
            rec['safe_rate'] = 0.0
        out[key] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpts', required=True)
    ap.add_argument('--proj', required=True)
    ap.add_argument('--bbox', required=True)
    ap.add_argument('--depth_margin', type=float, default=0.5)
    ap.add_argument('--thetas', type=str, default='0.5,0.7,0.9',
                    help='门控置信度阈值列表（逗号分隔）')
    ap.add_argument('--bs', type=int, default=4, help='val 批大小（ng6144 需 4 防 OOM）')
    ap.add_argument('--max-frames', type=int, default=0, help='只评测前 N 帧（0=全部），用于冒烟测试')
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--tag', default='b2_nav_error_propagation', help='结果标签')
    ap.add_argument('--out-json', default=None, help='把导航代价混淆矩阵存成 json')
    args = ap.parse_args()

    thetas = [float(t) for t in args.thetas.split(',')]

    args.local_rank = 0
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.resume = False
    args.experiment_path = '/tmp/eval_nav_prop_exp'
    os.makedirs(args.experiment_path, exist_ok=True)
    args.log_name = os.path.basename(args.config).replace('.yaml', '')
    os.environ['LOCAL_RANK'] = '0'
    logger = get_root_logger(name='eval_nav_prop')

    config = get_config(args, logger=logger)
    config.dataset.val.others.bs = args.bs
    num_classes = config.model.num_classes
    assert num_classes == 12, f'本脚本只用于 12 类体系，当前 config num_classes={num_classes}'

    coarse_map_t = torch.tensor(COARSE_MAP, dtype=torch.long).cuda()

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

            point2d_12 = torch.where(point2d >= 0, coarse_map_t[point2d.clamp(min=0)], point2d)

            in_bbox = point2d >= 0
            conf_base += conf_matrix(pred3d, sem, num_classes)
            for t in thetas:
                override = _override_mask(in_bbox, point2d_12, pred3d, conf3d, theta=t)
                pred_fused = torch.where(override, point2d_12, pred3d)
                conf_fused[t] += conf_matrix(pred_fused, sem, num_classes)

    # ===== 导航代价混淆矩阵 =====
    nav_base = nav_conf_from(conf_base)
    nav_fused = {t: nav_conf_from(conf_fused[t]) for t in thetas}
    gt_dist = nav_base.sum(axis=1)  # GT 侧 = 零误差导航代价分布

    def _build_rec(nav_conf):
        return {
            'nav_confusion': nav_conf.tolist(),
            'nav_confusion_row_norm': [[round(v, 6) for v in row]
                                       for row in row_norm(nav_conf).tolist()],
            'summary': nav_summary(nav_conf),
        }

    rec = {
        'tag': args.tag,
        'config': args.config,
        'ckpts': args.ckpts,
        'bbox_cache': os.path.basename(args.bbox),
        'depth_margin': args.depth_margin,
        'n_miss': int(n_miss),
        'nav_order': NAV_ORDER,
        'nav_groups': NAV_GROUPS,
        'nav_cost_label': NAV_COST_LABEL,
        'gt_nav_distribution': {NAV_ORDER[k]: int(v) for k, v in enumerate(gt_dist)},
        'base': _build_rec(nav_base),
        'fused': {str(t): _build_rec(nav_fused[t]) for t in thetas},
    }

    # ===== 打印报告 =====
    print_log('\n' + '=' * 90, logger=logger)
    print_log('B② 感知误差传递：导航代价五档混淆矩阵（行=GT，列=预测/融合）', logger=logger)
    print_log('=' * 90, logger=logger)

    gt_pred_label = "GT / pred"
    hdr = f'{gt_pred_label:<14}' + ''.join(f'{g:>10}' for g in NAV_ORDER)
    print_log(hdr, logger=logger)

    def _print_nav(tag, nav_conf):
        m = nav_conf.astype(np.float64)
        r = m.sum(axis=1, keepdims=True); r[r == 0] = 1.0
        p = m / r
        print_log(f'  [{tag}]', logger=logger)
        for gi, g in enumerate(NAV_ORDER):
            row = f'    {g:<12}'
            for gj in range(len(NAV_ORDER)):
                row += f'{p[gi, gj] * 100:>9.2f}%'
            print_log(row, logger=logger)

    _print_nav('base', nav_base)
    for t in thetas:
        _print_nav(f'fused θ={t}', nav_fused[t])

    print_log('-' * 90, logger=logger)
    print_log('聚焦指标（危险混淆率）：', logger=logger)

    def _print_focus(tag, nav_conf):
        s = nav_summary(nav_conf)
        for key, label in [('lethal', '致命类(lethal)'), ('hard', '实体类(hard)')]:
            d = s[key]
            print_log(f'  [{tag}] {label}: critical(→soft)={d["critical_rate"] * 100:.3f}%　'
                      f'severe(→free)={d["severe_rate"] * 100:.3f}%　'
                      f'safe(→254)={d["safe_rate"] * 100:.3f}%　(GT {d["gt_points"]:,} 点)',
                      logger=logger)

    _print_focus('base  ', nav_base)
    for t in thetas:
        _print_focus(f'θ={t}', nav_fused[t])

    print_log('=' * 90, logger=logger)
    print_log(f"GT 导航代价分布（零误差上界）: " +
              ', '.join(f'{NAV_ORDER[k]}={int(v):,}' for k, v in enumerate(gt_dist)), logger=logger)
    print_log(f'token 缺失样本数 = {n_miss}', logger=logger)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        print_log(f'结果已写入 {args.out_json}', logger=logger)


if __name__ == '__main__':
    main()
