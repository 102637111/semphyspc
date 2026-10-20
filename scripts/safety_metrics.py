#!/usr/bin/env python3
"""安全通道指标：致命类在固定误报率下的召回（recall @ FPPI），而不是 IoU。

为什么换指标：门控投票现在报 IoU，但对一个安全通道来说，漏掉一个行人和漏掉一个锥桶
的代价完全不同。行人检测领域的标准口径是 MR-FPPI（miss rate vs false positives per
image），这里报 recall @ FPPI ∈ {0.1, 1.0} 和 log-average miss rate。

忽略区（ignore region）：按通行做法，高度 < --min_h 像素的 GT 框（远处小目标，标注本身
不可靠）不计入召回，且落在忽略区里的检测也不计为误报。

输入格式（两边的列序不同，已核对源码，别搞混）：
  --det  run_2d_detector.py --stage det 产出，列 = [x, y, w, h, sem_id, score, cam_idx]
  --gt   build_s3_det_labels.py 产出，列 = [x, y, w, h, coco_id, dmin, dmax, cam_idx, n_pts]
匹配统一在 COCO 类别空间做（nuScenes 的 bus.bendy / bus.rigid 都映射到 COCO bus）。

用法：
    python3 scripts/safety_metrics.py --det data/det_rtdetr_val_veh.npz \
        --gt data/s3_det_labels_val.npz --n_frames 6019 --out results/safety_s2_zeroshot.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

# det 里的 sem_id → COCO id（与 run_2d_detector.py 的 COCO_TO_NUSC 互为逆）
SEM_TO_COCO = {2: 0, 14: 1, 21: 3, 9: 80, 12: 81,
               17: 2, 16: 5, 15: 5, 23: 7, 22: 7}

FATAL = {0: 'person', 1: 'bicycle', 3: 'motorcycle', 2: 'car', 5: 'bus', 7: 'truck'}
NONFATAL = {80: 'barrier', 81: 'trafficcone'}


def iou_matrix(a, b):
    """a: (n,4) b: (m,4)，都是 xywh。返回 (n,m) 的 IoU。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.minimum(ax1[:, None], bx1[None, :]) - np.maximum(a[:, 0][:, None], b[:, 0][None, :])
    ih = np.minimum(ay1[:, None], by1[None, :]) - np.maximum(a[:, 1][:, None], b[:, 1][None, :])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    ua = a[:, 2] * a[:, 3]
    ub = b[:, 2] * b[:, 3]
    union = ua[:, None] + ub[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_frames', type=int, default=6019, help='评测帧数（算 FPPI 的分母）')
    ap.add_argument('--n_cams', type=int, default=6)
    ap.add_argument('--iou_thr', type=float, default=0.5)
    ap.add_argument('--min_h', type=float, default=20.0, help='GT 框低于这个像素高度视为忽略区')
    ap.add_argument('--tag', default='safety')
    args = ap.parse_args()

    d = np.load(args.det, allow_pickle=True)
    dtok, doff, dets = d['tokens'], d['offsets'], d['dets']
    g = np.load(args.gt, allow_pickle=True)
    gtok, goff, gbox = g['tokens'], g['offsets'], g['boxes']
    print(f'[det] {len(dtok)} 帧 / {len(dets)} 框   [gt] {len(gtok)} 帧 / {len(gbox)} 框')

    det_by = defaultdict(list)
    for i, t in enumerate(dtok):
        for r in dets[doff[i]:doff[i + 1]]:
            c = SEM_TO_COCO.get(int(r[4]))
            if c is not None:
                det_by[(str(t), int(r[6]))].append((float(r[5]), c, r[:4]))
    gt_by = defaultdict(list)
    for i, t in enumerate(gtok):
        for r in gbox[goff[i]:goff[i + 1]]:
            gt_by[(str(t), int(r[7]))].append((int(r[4]), r[:4]))

    keys = set(det_by) | set(gt_by)
    print(f'[配对] (帧,相机) 组合 {len(keys)}')

    # 每个类收集 (score, is_tp)，忽略区的检测直接丢弃
    per_cls = defaultdict(list)
    n_gt = defaultdict(int)
    for k in keys:
        gts = gt_by.get(k, [])
        dts = det_by.get(k, [])
        for c in set([x[0] for x in gts]) | set([x[1] for x in dts]):
            gc = np.array([x[1] for x in gts if x[0] == c], dtype=np.float32).reshape(-1, 4)
            dc = [(s, b) for (s, cc, b) in dts if cc == c]
            keep = gc[:, 3] >= args.min_h
            gv, gi = gc[keep], gc[~keep]
            n_gt[c] += len(gv)
            if not dc and len(gv) == 0:
                continue
            dc.sort(key=lambda x: -x[0])
            db = np.array([b for _, b in dc], dtype=np.float32).reshape(-1, 4)
            iv = iou_matrix(db, gv)
            ii = iou_matrix(db, gi)
            used_v, used_i = set(), set()
            for j, (s, _) in enumerate(dc):
                # 先看能不能匹配到有效 GT
                best, bj = 0.0, -1
                for q in range(len(gv)):
                    if q not in used_v and iv[j, q] > best:
                        best, bj = iv[j, q], q
                if best >= args.iou_thr:
                    used_v.add(bj)
                    per_cls[c].append((s, 1))
                    continue
                # 再看是不是落在忽略区里
                hit_ig = any(ii[j, q] >= args.iou_thr and q not in used_i
                             for q in range(len(gi)))
                if hit_ig:
                    continue
                per_cls[c].append((s, 0))

    n_img = args.n_frames * args.n_cams

    def curve(c):
        rows = sorted(per_cls[c], key=lambda x: -x[0])
        if not rows:
            return np.array([]), np.array([]), np.array([])
        tp = np.cumsum([r[1] for r in rows])
        fp = np.cumsum([1 - r[1] for r in rows])
        sc = np.array([r[0] for r in rows])
        recall = tp / max(1, n_gt[c])
        fppi = fp / n_img
        return sc, recall, fppi

    def at_fppi(c, target):
        """在 FPPI <= target 的约束下能达到的最大召回。"""
        sc, rc, fp = curve(c)
        ok = fp <= target
        return float(rc[ok].max()) if ok.any() else 0.0

    def lamr(c):
        """log-average miss rate：9 个对数均匀 FPPI 点上漏检率的平均（COCO 口径）。"""
        sc, rc, fp = curve(c)
        if len(fp) == 0:
            return float('nan')
        pts = np.logspace(np.log10(1e-2), np.log10(1.0), 9)
        mrs = []
        for p in pts:
            ok = fp <= p
            r = float(rc[ok].max()) if ok.any() else 0.0
            mrs.append(max(1e-10, 1.0 - r))
        return float(np.exp(np.mean(np.log(mrs))))

    out = {'tag': args.tag, 'det': args.det, 'gt': args.gt, 'iou_thr': args.iou_thr,
           'min_h': args.min_h, 'n_images': n_img, 'classes': {}}
    for group, names in (('fatal', FATAL), ('nonfatal', NONFATAL)):
        print(f'\n=== {group}（致命类：漏一个的代价最高）===')
        print(f"{'类别':<13}{'GT':>8}{'recall@FPPI0.1':>16}{'recall@FPPI1.0':>16}"
              f"{'log-avg miss':>14}")
        print('-' * 68)
        for c, nm in names.items():
            if n_gt[c] == 0:
                print(f'{nm:<13}{0:>8}   —— GT 为 0，不可测')
                out['classes'][nm] = {'group': group, 'n_gt': 0}
                continue
            r01, r10, la = at_fppi(c, 0.1), at_fppi(c, 1.0), lamr(c)
            print(f'{nm:<13}{n_gt[c]:>8,}{r01:>16.4f}{r10:>16.4f}{la:>14.4f}')
            out['classes'][nm] = {'group': group, 'n_gt': n_gt[c],
                                  'recall_at_fppi_0.1': r01, 'recall_at_fppi_1.0': r10,
                                  'log_avg_miss_rate': la}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n已写出：{args.out}')


if __name__ == '__main__':
    main()
