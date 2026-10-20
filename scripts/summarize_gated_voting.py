#!/usr/bin/env python3
"""汇总 eval_gated_voting.py 产出的多个 JSON，打印论文用的对比表。

用法:
    python3 scripts/summarize_gated_voting.py results/gated_voting_s1_oracle.json \
                                                results/gated_voting_s2_rtdetr.json
"""
import json
import sys
import os

RARE_ORDER = [
    'human.pedestrian.adult', 'human.pedestrian.child',
    'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer', 'human.pedestrian.stroller',
    'human.pedestrian.wheelchair', 'movable_object.barrier',
    'movable_object.trafficcone', 'vehicle.bicycle', 'vehicle.motorcycle',
]


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(v):
    return '  —  ' if v is None else f'{v:.4f}'


def main():
    recs = [load(p) for p in sys.argv[1:]]

    print('=' * 100)
    print('设置概览')
    print('=' * 100)
    for r in recs:
        print(f"  {r['tag']:<22} bbox={r['bbox_cache']:<45} group_refine={r['group_refine']}")
        print(f"  {'':<22} base: 罕见类 mIoU={r['base']['miou_rare']:.4f}  全类 mIoU={r['base']['miou_all']:.4f}")
    print()

    # mIoU 汇总表
    thetas = sorted({t for r in recs for t in r['fused']}, key=float)
    print('=' * 100)
    print('mIoU 汇总（行=设置与指标，列=θ）')
    print('=' * 100)
    hdr = f"{'设置':<24}{'指标':<16}" + ''.join(f"{'θ=' + t:>10}" for t in thetas)
    print(hdr)
    print('-' * 100)
    for r in recs:
        for metric, key in [('罕见类 mIoU', 'miou_rare'), ('全类 mIoU', 'miou_all')]:
            line = f"{r['tag']:<24}{metric:<16}"
            for t in thetas:
                v = r['fused'].get(t, {}).get(key)
                line += f"{fmt(v):>10}"
            print(line)
        print('-' * 100)

    # per-class 对比（对每个设置取罕见类 mIoU 最高的 θ）
    print()
    print('=' * 100)
    print('逐类 IoU（每个设置取其罕见类 mIoU 最优的 θ）')
    print('=' * 100)
    hdr = f"{'类别':<38}" + ''.join(f"{r['tag'][:16]:>18}" for r in recs)
    print(hdr)
    print('-' * 100)
    print(f"{'[base 3D]':<38}" + ''.join(f"{fmt(r['base']['miou_rare']):>18}" for r in recs))
    for r in recs:
        best_t = max(r['fused'], key=lambda t: r['fused'][t]['miou_rare'])
        r['_best_t'] = best_t
    print(f"{'[最优 θ 的罕见类 mIoU]':<38}" + ''.join(f"{r['_best_t']:>18}" for r in recs))
    print('-' * 100)
    for cls in RARE_ORDER:
        line = f'{cls:<38}'
        for r in recs:
            line += f"{fmt(r['base']['per_class_iou'].get(cls)):>18}"
        print(line)
    print('-' * 100)
    for cls in RARE_ORDER:
        line = f'{cls + " (fused)":<38}'
        for r in recs:
            v = r['fused'][r['_best_t']]['per_class_iou'].get(cls)
            line += f"{fmt(v):>18}"
        print(line)
    print('=' * 100)
    for r in recs:
        bt = r['_best_t']
        print(f"  {r['tag']}: 最优 θ={bt}  罕见类 mIoU {r['base']['miou_rare']:.4f} -> "
              f"{r['fused'][bt]['miou_rare']:.4f}  全类 {r['base']['miou_all']:.4f} -> {r['fused'][bt]['miou_all']:.4f}")


if __name__ == '__main__':
    main()
