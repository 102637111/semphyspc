#!/usr/bin/env python3
"""实验B：物体级多数投票 vs 逐点预测，量化「先分割再判类」能救回多少动态类。

背景：
  语义 32 类端到端 mIoU 只有 0.2342，尤其动态物体(车/行人/骑行者)逐点分割困难。
  级联思路：先地面几何分割(实验A已证 42% 场景)，再对非地面物体 crop 做物体级判类。
  本实验用「真值实例分割(inst) + 现有语义模型逐点预测多数投票」来上界地衡量：
  如果物体切分是完美的，物体级判类能把动态类的准确率提到多少。

方法：
  1. 加载语义单任务模型，对 val 子集逐点推理得 pred (argmax)。
  2. 用真值 inst 标签把点聚成物体实例(inst>0 的 thing 类)，每个实例:
     - 真类别 = 实例内 sem 的众数
     - 物体级预测 = 实例内 pred 的众数（多数投票，相当于把逐点噪声抹平）
  3. 对比：thing 类点的「逐点准确率」 vs 「物体级准确率」，分整体与逐类。

注意：这是多数投票(用同一个弱模型)，只消除「逐点噪声」，不引入更强的分类器；
     若这里已经有明显提升，则换一个专门的物体分类器收益更大。

用法:
    python scripts/analyze_object_vote.py \
        --data data/nuscenes_seg_full_16k/val \
        --sem_cfg experiments/seg_nuscenes_full_16k/cfgs/singletask_sem/20260821-221520/config.yaml \
        --sem_ckpt experiments/seg_nuscenes_full_16k/cfgs/singletask_sem/20260821-221520/ckpt-best.pth \
        --n 500 --seed 0
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 非地面、非 noise 的「物体(thing)」类：动态 + 可移动物 + 静态 object(bicycle_rack)。
# 地面 {24,25,26,27} 与 static 的 manmade/other/vegetation {28,29,30} 属 stuff，不参与。
THING_CLASSES = set(range(1, 24)) | {28, 29, 30}  # 这里只对 inst>0 的实例判类，stuff 自然被排除

CLASS_NAMES = {
    0: 'noise', 1: 'animal',
    2: 'ped.adult', 3: 'ped.child', 4: 'ped.worker', 5: 'ped.mobility',
    6: 'ped.police', 7: 'ped.stroller', 8: 'ped.wheelchair',
    9: 'barrier', 10: 'debris', 11: 'pushable', 12: 'trafficcone',
    13: 'bicycle_rack', 14: 'bicycle',
    15: 'bus.bendy', 16: 'bus.rigid', 17: 'car', 18: 'construction',
    19: 'ambulance', 20: 'police_car', 21: 'motorcycle', 22: 'trailer',
    23: 'truck', 24: 'driveable', 25: 'flat_other', 26: 'sidewalk', 27: 'terrain',
    28: 'manmade', 29: 'other', 30: 'vegetation', 31: 'ego',
}


def build_model(cfg_path, ckpt_path):
    from models.point_mamba_seg import PointMambaSeg
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = PointMambaSeg(EasyDict(cfg['model']))
    sd = torch.load(ckpt_path, map_location='cpu')
    base = sd.get('base_model') or sd.get('model')
    base = {k.replace('module.', ''): v for k, v in base.items()}
    model.load_state_dict(base, strict=False)
    model.cuda().eval()
    return model


@torch.no_grad()
def infer(model, pts):
    logits, _ = model(pts)
    return logits.argmax(-1)[0].cpu().numpy().astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--sem_cfg', required=True)
    ap.add_argument('--sem_ckpt', required=True)
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    rng = np.random.default_rng(args.seed)
    if args.n < len(files):
        files = rng.choice(files, args.n, replace=False).tolist()

    print(f"[experimentB] 载入模型 + {len(files)} 个 val 样本", flush=True)
    model = build_model(args.sem_cfg, args.sem_ckpt)

    # 逐点统计：thing 类点
    n_pt_correct = 0
    n_pt_total = 0
    # 物体级统计
    n_obj_correct = 0
    n_obj_total = 0
    # 逐类
    pt_correct = np.zeros(32, dtype=np.int64)
    pt_total = np.zeros(32, dtype=np.int64)
    obj_correct = np.zeros(32, dtype=np.int64)
    obj_total = np.zeros(32, dtype=np.int64)

    t0 = time.time()
    for fi, fp in enumerate(files):
        d = np.load(fp)
        xyz = d['xyz'].astype(np.float32)
        rgb = d['rgb'].astype(np.float32)
        sem = d['sem'].astype(np.int64)
        inst = d['inst'].astype(np.int64)

        pts = torch.from_numpy(np.concatenate([xyz, rgb], axis=1)).float().unsqueeze(0).cuda()
        pred = infer(model, pts)  # (N,)

        valid = sem >= 0
        # 逐点：所有 thing 类有效点
        thing_mask = valid & np.isin(sem, list(range(1, 24)))
        pt_total_c = np.bincount(sem[thing_mask], minlength=32)
        pt_correct_c = np.bincount(
            sem[thing_mask][pred[thing_mask] == sem[thing_mask]], minlength=32)
        pt_total += pt_total_c
        pt_correct += pt_correct_c
        n_pt_correct += int((pred[thing_mask] == sem[thing_mask]).sum())
        n_pt_total += int(thing_mask.sum())

        # 物体级：按 inst 分组，仅 thing 类实例 (inst>0 且 sem in 1..23)
        obj_mask = thing_mask & (inst > 0)
        if obj_mask.any():
            ids = np.unique(inst[obj_mask])
            for oid in ids:
                m = obj_mask & (inst == oid)
                true_cls = int(np.bincount(sem[m]).argmax())  # 实例真类众数
                vote = int(np.bincount(pred[m]).argmax())       # 实例内多数投票
                obj_total[true_cls] += 1
                if vote == true_cls:
                    obj_correct[true_cls] += 1
                    n_obj_correct += 1
                n_obj_total += 1

        if (fi + 1) % 100 == 0:
            print(f"  [{fi+1}/{len(files)}] 用时 {time.time()-t0:.0f}s", flush=True)

    print("\n================ 实验B 结果（物体级多数投票 vs 逐点）================", flush=True)
    print(f"  样本数 {len(files)}, 物体实例总数 {n_obj_total}", flush=True)
    print(f"  thing 类逐点准确率 : {n_pt_correct/max(n_pt_total,1):.4f}", flush=True)
    print(f"  物体级多数投票准确率: {n_obj_correct/max(n_obj_total,1):.4f}  (提升 {n_obj_correct/max(n_obj_total,1)-n_pt_correct/max(n_pt_total,1):+.4f})", flush=True)

    print("\n  {:2s} {:<16s} {:>10s} {:>10s} {:>10s}".format(
        'id', 'class', 'pt_acc', 'obj_acc', 'obj数'), flush=True)
    # 只打印实例数 >= 5 的类
    for c in range(1, 24):
        if obj_total[c] < 5:
            continue
        pa = pt_correct[c] / max(pt_total[c], 1)
        oa = obj_correct[c] / max(obj_total[c], 1)
        print(f"  {c:2d} {CLASS_NAMES.get(c, str(c)):<16s} "
              f"{pa:>10.4f} {oa:>10.4f} {obj_total[c]:>10d}", flush=True)

    print("======================================================================", flush=True)


if __name__ == '__main__':
    main()
