#!/usr/bin/env python3
"""框级逐类召回：把 `run_2d_detector.py --stage det` 的输出与投影 GT 对照。

用途是把 §4.13 的「检测器召回」做成**同口径可比**的数字：零样本 COCO 检测器
（data/det_rtdetr_val.npz）与 S3 微调后（data/det_rtdetr_s3_val.npz）跑同一份
data/s3_det_labels_val.npz（nuScenes val split 的 3D 真值框 2D 投影），
同一 score 阈值、同一 IoU 阈值，得到逐类 recall / precision。

两侧的 token 都是**帧级**（lidar sample token），GT 的每一行带 cam_idx。

用法:
    python3 scripts/eval_det_recall.py --det data/det_rtdetr_val.npz \
        --gt data/s3_det_labels_val.npz --score_thr 0.35 --iou_thr 0.5
"""

import argparse
import numpy as np


# GT（data/s3_det_labels_val.npz）用的是 **82 类标签空间**的索引。
TARGET = {0: 'person', 1: 'bicycle', 3: 'motorcycle', 80: 'barrier', 81: 'trafficcone'}

# 而 run_2d_detector.py 存进 det npz 的是 COCO_TO_NUSC 的值，即 **nuScenes lidar id**
# （person→2、bicycle→14、motorcycle→21、barrier→9、trafficcone→12）。
# 两侧空间不同，比较前必须重映射，否则会得到「召回全 0」的假象。
LIDAR2LABEL = {2: 0, 14: 1, 21: 3, 9: 80, 12: 81}


def xywh_iou(a, b):
    """a (N,4) b (M,4) xywh -> (N,M) IoU。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float64)
    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(ax1[:, None], bx1[None, :]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None, :]) - np.maximum(ay1[:, None], by1[None, :]), 0, None)
    inter = iw * ih
    ua = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-9), 0.0)


def greedy_match(ious, iou_thr):
    """按 IoU 从高到低贪心匹配，返回匹配上的 (det_idx, gt_idx) 对数。该 IoU 矩阵的
    行/列都只属于同一个类，所以逐类匹配与全局匹配等价。"""
    if ious.size == 0:
        return 0
    order = np.argsort(-ious, axis=None)
    rd, cd = np.unravel_index(order, ious.shape)
    ud, uc = set(), set()
    n = 0
    for d, c in zip(rd, cd):
        if ious[d, c] < iou_thr:
            break
        if d in ud or c in uc:
            continue
        ud.add(d); uc.add(c); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--score_thr', type=float, default=0.35)
    ap.add_argument('--iou_thr', type=float, default=0.5)
    ap.add_argument('--tag', default='')
    ap.add_argument('--det_space', default='lidar', choices=['lidar', 'label'],
                    help='det npz 的类 id 空间：lidar=COCO_TO_NUSC 的值（run_2d_detector 的默认输出）')
    args = ap.parse_args()

    det = np.load(args.det, allow_pickle=True)
    gt = np.load(args.gt, allow_pickle=True)

    d_off, d_rows = det['offsets'], det['dets']
    d_idx = {str(t): i for i, t in enumerate(det['tokens'])}
    g_off, g_boxes = gt['offsets'], gt['boxes']
    g_idx = {str(t): i for i, t in enumerate(gt['tokens'])}

    # 分母是**全部 GT 帧**：run_2d_detector 只把有检测的帧写进 npz，若只遍历两边都有的帧，
    # 会把「检测器整帧无输出」这类必然漏检从分母里剔除，从而系统性虚高召回。
    n_empty = sum(1 for t in g_idx if t not in d_idx)
    print(f'[recall] det 帧 {len(d_idx)}   gt 帧 {len(g_idx)}   '
          f'其中检测器无输出的帧 {n_empty}（按零检测计入分母）')

    stat = {c: dict(gt=0, tp=0, det=0) for c in TARGET}
    for t in g_idx:
        gi = g_idx[t]
        gb = g_boxes[g_off[gi]:g_off[gi + 1]]
        di = d_idx.get(t)
        db = d_rows[d_off[di]:d_off[di + 1]] if di is not None else np.zeros((0, 7), np.float32)
        if len(db):
            db = db[db[:, 5] >= args.score_thr]
        if len(db) and args.det_space == 'lidar':
            db = db.copy()
            db[:, 4] = [LIDAR2LABEL.get(int(v), -1) for v in db[:, 4]]
            db = db[db[:, 4] >= 0]

        for c in TARGET:
            g = gb[gb[:, 4].astype(int) == c][:, :4]
            d = db[db[:, 4].astype(int) == c][:, :4] if len(db) else np.zeros((0, 4), np.float32)
            stat[c]['gt'] += len(g)
            stat[c]['det'] += len(d)
            stat[c]['tp'] += greedy_match(xywh_iou(g, d), args.iou_thr)

    print(f'[recall] score_thr={args.score_thr}  iou_thr={args.iou_thr}  {args.tag}')
    print(f'{"cls":>4} {"name":<13} {"GT":>7} {"det":>7} {"TP":>7} {"recall":>7} {"prec":>7}')
    for c, name in TARGET.items():
        s = stat[c]
        r = s['tp'] / s['gt'] if s['gt'] else float('nan')
        p = s['tp'] / s['det'] if s['det'] else float('nan')
        print(f'{c:>4} {name:<13} {s["gt"]:>7} {s["det"]:>7} {s["tp"]:>7} {r:>7.4f} {p:>7.4f}')
    tot_g = sum(s['gt'] for s in stat.values())
    tot_tp = sum(s['tp'] for s in stat.values())
    print(f'{"":>4} {"合计":<12} {tot_g:>7} {"":>7} {tot_tp:>7} {tot_tp/tot_g:>7.4f}')
    rare = [1, 3, 80, 81]
    rr = np.mean([stat[c]['tp'] / stat[c]['gt'] if stat[c]['gt'] else np.nan for c in rare])
    print(f'[recall] 关注类（bicycle/motorcycle/barrier/trafficcone）平均召回 = {rr:.4f}')


if __name__ == '__main__':
    main()
