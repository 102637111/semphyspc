#!/usr/bin/env python3
"""测 2D→3D 关联的质量：把 val 集每个点按门控投票规则关联到 2D 框类别后，
与逐点真值比对，统计关联 precision / 罕见类 recall / 覆盖率。

不依赖 3D 模型（只用 npz 里的 xyz 与 sem），因此可对多个 2D 框缓存做纯对比，
用来回答「深度门控到底有没有用」——这是 §4.13 中 S2 三种深度来源消融的依据。

关联 precision 的定义：在所有被 2D 框覆盖且真值属于罕见类的点中，
2D 给出的类别与真值一致的比例。注意它衡量的是**关联**质量（像素↔点的对应
是否落在同一物体上），不是端到端 mIoU —— 后者还取决于门控投票是否触发。

用法:
    python3 scripts/eval_2d_assoc_precision.py \
        --proj data/nuscenes_projection_index.npz \
        --cache GT=data/nuscenes_2d_bbox_cache.npz \
                lidar=data/nuscenes_2d_bbox_cache_rtdetr_val_lidar.npz \
                size=data/nuscenes_2d_bbox_cache_rtdetr_val_size.npz \
                none=data/nuscenes_2d_bbox_cache_rtdetr_val_none.npz
"""
import os
import sys
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMG_W, IMG_H = 1600, 900
RARE_CLASSES = {2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 21}


def project(xyz, K, T):
    N = len(xyz)
    pts_h = np.concatenate([xyz, np.ones((N, 1))], axis=1).astype(np.float64)
    pts_cam = (T @ pts_h.T).T
    depth = pts_cam[:, 2]
    uv = K @ pts_cam[:, :3].T
    u = uv[0] / uv[2]
    v = uv[1] / uv[2]
    valid = (depth > 0) & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    return u, v, depth, valid


def assign(xyz, K_all, T_all, boxes, depth_margin, use_depth):
    out = np.full(len(xyz), -1, dtype=np.int64)
    for cam_idx in range(K_all.shape[0]):
        u, v, depth, valid = project(xyz, K_all[cam_idx], T_all[cam_idx])
        for row in boxes:
            if int(row[7]) != cam_idx:
                continue
            x, y, w, h, sem_id = row[0], row[1], row[2], row[3], int(row[4])
            m = valid & (u >= x) & (u < x + w) & (v >= y) & (v < y + h)
            if use_depth and row.shape[0] >= 7:
                m &= (depth >= row[5] - depth_margin) & (depth <= row[6] + depth_margin)
            out[m] = sem_id
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proj', required=True)
    ap.add_argument('--val-dir', default='data/nuscenes_seg_full_16k/val')
    ap.add_argument('--cache', nargs='+', required=True, help='名字=路径，可多个')
    ap.add_argument('--depth-margin', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 帧（0=全部）')
    args = ap.parse_args()

    proj = np.load(args.proj, allow_pickle=True)
    proj_idx = {str(t): i for i, t in enumerate(proj['tokens'])}
    K_all, T_all = proj['K'], proj['T']

    caches = []
    for spec in args.cache:
        name, path = spec.split('=', 1)
        d = np.load(path, allow_pickle=True)
        caches.append((name, path,
                       {str(t): i for i, t in enumerate(d['tokens'])},
                       d['offsets'], d['boxes']))

    files = sorted(glob.glob(os.path.join(args.val_dir, '*.npz')))
    if args.limit:
        files = files[:args.limit]
    print(f'[assoc] {len(files)} 帧, depth_margin={args.depth_margin}', flush=True)

    # 统计量: 对每个 cache × {use_depth=True/False}
    stats = {}
    for name, _, _, _, _ in caches:
        for ud in (True, False):
            stats[(name, ud)] = dict(n=0, assigned=0, correct=0,
                                     rare_gt=0, rare_assigned=0, rare_correct=0)

    for fi, fp in enumerate(files):
        tok = os.path.basename(fp)[:-4]
        if tok not in proj_idx:
            continue
        d = np.load(fp)
        xyz = d['xyz'].astype(np.float64)
        sem = d['sem'].astype(np.int64)
        pi = proj_idx[tok]
        for name, _, cidx, coff, cbox in caches:
            if tok not in cidx:
                continue
            bi = cidx[tok]
            boxes = cbox[coff[bi]:coff[bi + 1]]
            rare_gt = np.isin(sem, list(RARE_CLASSES))
            for ud in (True, False):
                if ud:
                    b = boxes
                else:
                    # 无深度门控：只把深度区间放宽到无穷，其余规则完全不变
                    b = boxes.copy()
                    if b.shape[1] >= 7:
                        b[:, 5], b[:, 6] = -1e9, 1e9
                p2d = assign(xyz, K_all[pi], T_all[pi], b, args.depth_margin, True)
                s = stats[(name, ud)]
                asg = p2d >= 0
                ok = asg & (p2d == sem)
                s['n'] += len(sem)
                s['assigned'] += int(asg.sum())
                s['correct'] += int(ok.sum())
                s['rare_gt'] += int(rare_gt.sum())
                s['rare_assigned'] += int((asg & rare_gt).sum())
                s['rare_correct'] += int((ok & rare_gt).sum())
        if (fi + 1) % 500 == 0:
            print(f'  ... {fi + 1}/{len(files)}', flush=True)

    print()
    print('=' * 104)
    print(f"{'2D 框来源':<10}{'深度门控':<10}{'关联 precision':>16}{'罕见类 precision':>18}"
          f"{'罕见类 recall':>16}{'点覆盖率':>12}")
    print('-' * 104)
    for name, path, _, _, _ in caches:
        for ud in (True, False):
            s = stats[(name, ud)]
            if s['assigned'] == 0:
                continue
            pr = s['correct'] / s['assigned']
            pr_r = s['rare_correct'] / s['rare_assigned'] if s['rare_assigned'] else float('nan')
            rc_r = s['rare_correct'] / s['rare_gt'] if s['rare_gt'] else float('nan')
            cov = s['assigned'] / s['n']
            print(f'{name:<10}{"有" if ud else "无":<10}{pr:>16.3f}{pr_r:>18.3f}'
                  f'{rc_r:>16.3f}{cov:>12.3f}')
        print('-' * 104)


if __name__ == '__main__':
    main()
