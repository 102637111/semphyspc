#!/usr/bin/env python3
"""在最佳 checkpoint 上做逐类语义分割评估（mIoU / per-class IoU / OA）。

用法:
    python3 scripts/eval_seg_full.py --config cfgs/seg_nuscenes_full.yaml \
        --ckpts experiments/seg_nuscenes_full/cfgs/seg_full_v1/<ts>/ckpt-best.pth
"""
import os, sys, argparse, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import builder
from utils.config import get_config
from utils.logger import get_root_logger, print_log

# nuScenes-lidarseg 32 类（索引顺序）
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

# 合并相似类后的 12 类（物理交互粗粒度，索引顺序）
LIDARSEG_COARSE = [
    'noise', 'animal', 'person', 'car', 'bus', 'truck',
    'two_wheeler', 'ground', 'manmade', 'vegetation', 'frangible', 'ego',
]

# 32 → 12 映射（与 NuscenesSegFull16kMerged.yaml 的 CLASS_MAP 一致）。
# 用于把「32 类模型」的预测/GT 聚合到 12 类，和「直接学 12 类」做同一标准下的公平对比。
COARSE_MAP = [0, 1, 2, 2, 2, 2, 2, 2, 2, 10, 10, 10, 10, 8, 6, 4, 4, 3, 5, 3, 3, 6, 5, 5, 7, 7, 7, 7, 8, 8, 9, 11]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpts', required=True)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--out_json', type=str, default=None, help='导出结构化结果 JSON 存档')
    ap.add_argument('--exp_tag', type=str, default='baseline', help='实验标签（写入 JSON）')
    ap.add_argument('--bs', type=int, default=16, help='val 批大小（细粒度 num_group 大时需调小防 OOM）')
    # 可选：把罕见类的「GT 类 → 预测类」分布导出来。默认关闭，不影响原有调用。
    # 动机：child/stroller 的 IoU 精确为 0，但模型到底是把它们预测成了 ped.adult
    # （=能定位、只是没学会高度阈值 → HAG 显式特征有用）还是 noise（=簇根本不形成
    # → 实例级后处理那条路断了）。这个分布是区分两种情况的唯一直接证据。
    ap.add_argument('--dump_confusion', action='store_true')
    ap.add_argument('--confusion_topk', type=int, default=6)
    ap.add_argument('--confusion_out', type=str, default=None)
    # 把「32 类模型」的预测/GT 聚合到 12 类后评测，与「直接学 12 类」同一标准对比。
    ap.add_argument('--coarse_agg', action='store_true')
    args = ap.parse_args()

    args.local_rank = 0
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.resume = False
    args.experiment_path = '/tmp/eval_seg_exp'
    os.makedirs(args.experiment_path, exist_ok=True)
    args.log_name = os.path.basename(args.config).replace('.yaml', '')
    os.environ['LOCAL_RANK'] = '0'
    logger = get_root_logger(name='eval_seg')

    config = get_config(args, logger=logger)
    config.dataset.val.others.bs = args.bs
    num_classes = config.model.num_classes

    # --coarse_agg：模型输出仍是 32 类，但把预测/GT 都映射到 12 类后评测
    if args.coarse_agg:
        assert num_classes == 32, '--coarse_agg 只用于 32 类模型（12 类模型直接用 merged config 评测）'
        num_classes = 12

    # 类别名：32 类用 lidarseg 全名，合并后（12 类）用粗粒度名
    CLASS_NAMES = LIDARSEG_32 if num_classes == 32 else LIDARSEG_COARSE
    # 关注的类：32 类下看罕见 thing 类 + ped.adult 作参照；合并后看全部类
    WATCH = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 21] if num_classes == 32 else list(range(num_classes))

    print_log(f'加载 checkpoint: {args.ckpts}', logger=logger)
    _, val_loader = builder.dataset_builder(args, config.dataset.val)

    model = builder.model_builder(config.model)
    builder.load_model(model, args.ckpts, logger=logger)
    model = nn.DataParallel(model).cuda()
    model.eval()

    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64).cuda()
    with torch.no_grad():
        for i, (_, _, data) in enumerate(val_loader):
            pts = data[0].cuda()
            sem = data[1].cuda()
            ret = model(pts)
            # PointMambaSeg.forward 返回 (sem_logits, mat_logits) 元组（材料头加入后），
            # DataParallel 也会把元组按元素 gather —— 两者都只取语义分支
            logits = ret[0] if isinstance(ret, (tuple, list)) else ret
            pred = logits.argmax(-1)
            if args.coarse_agg:
                coarse_map = torch.tensor(COARSE_MAP, device=pred.device)
                pred = coarse_map[pred]
                sem = torch.where(sem >= 0, coarse_map[sem.clamp(min=0)], sem)
            valid = sem >= 0
            pred_v = pred[valid].long()
            sem_v = sem[valid].long()
            for c in range(num_classes):
                m = (sem_v == c)
                if m.sum() == 0:
                    continue
                conf[c] += torch.bincount(pred_v[m], minlength=num_classes)

    # 逐类 IoU
    print_log('\n' + '=' * 70, logger=logger)
    print_log(f'逐类 IoU（{num_classes} 类，val 集）', logger=logger)
    print_log('=' * 70, logger=logger)
    print_log(f'{"idx":>3} {"class":<38} {"IoU":>7} {"GT点数":>10}', logger=logger)
    ious = []
    for c in range(num_classes):
        gt = conf[c].sum().item()
        pred_sum = conf[:, c].sum().item()
        tp = conf[c, c].item()
        union = gt + pred_sum - tp
        iou = tp / union if union > 0 else float('nan')
        ious.append(iou)
        print_log(f'{c:>3} {CLASS_NAMES[c]:<38} {iou:>7.3f} {gt:>10,}', logger=logger)

    valid_ious = [i for i in ious if not np.isnan(i)]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    oa = conf.diag().sum().item() / conf.sum().item()
    print_log('-' * 70, logger=logger)
    print_log(f'mIoU (排除无 GT 类) = {miou:.4f}', logger=logger)
    print_log(f'OA  (整体准确率)     = {oa:.4f}', logger=logger)
    print_log('=' * 70, logger=logger)

    if args.dump_confusion:
        print_log('\n' + '=' * 70, logger=logger)
        print_log(f'罕见类 GT → 预测 分布（top-{args.confusion_topk}，val 集）', logger=logger)
        print_log('=' * 70, logger=logger)
        cm = conf.cpu().numpy()

        # ---- 安全聚合：论文核心是「可选择碰撞」，所以最要命的不是 mIoU，而是
        # 「该躲的（行人/动物）被预测成可以撞的（可破坏/可变形）」这种致命误判。
        # 这里按 §3.3 的 5 级安全等级聚合混淆矩阵。逐点统计（不按实例），
        # 口径与逐类 IoU 一致；实例级版本见级联判类那条线。
        if num_classes == 32:
            LETHAL = [1, 2, 3, 4, 5, 6, 7, 8]         # 动物 + 全部行人子类
            FRANGIBLE = [9, 10, 11, 12]               # 护栏/碎屑/可推动/锥桶
            DEFORMABLE = [30]                         # 植被（27 地形 2026-09-14 移入 PASSABLE）
            PASSABLE = [24, 25, 26, 27]               # 路面/其他平面/人行道/地形
        else:
            # 合并后 12 类：person 已含全部行人子类，animal 单独
            LETHAL = [1, 2]                           # animal + person
            FRANGIBLE = [10]                          # frangible（护栏/碎屑/锥桶等）
            DEFORMABLE = [9]                          # vegetation
            PASSABLE = [7]                            # ground（路面/人行道/地形）
        COLLIDABLE = FRANGIBLE + DEFORMABLE           # 决策模块推荐撞击的集合
        lethal_gt = int(cm[LETHAL].sum())
        agg = {'lethal_gt_points': lethal_gt,
               'lethal_to_lethal': int(cm[np.ix_(LETHAL, LETHAL)].sum()),
               'lethal_to_collidable': int(cm[np.ix_(LETHAL, COLLIDABLE)].sum()),
               'lethal_to_frangible': int(cm[np.ix_(LETHAL, FRANGIBLE)].sum()),
               'lethal_to_deformable': int(cm[np.ix_(LETHAL, DEFORMABLE)].sum()),
               'lethal_to_passable': int(cm[np.ix_(LETHAL, PASSABLE)].sum()),
               'lethal_to_hard': int(cm[LETHAL].sum() - cm[np.ix_(LETHAL, LETHAL)].sum()
                                     - cm[np.ix_(LETHAL, COLLIDABLE)].sum()
                                     - cm[np.ix_(LETHAL, PASSABLE)].sum()
                                     - cm[np.ix_(LETHAL, [0])].sum())}
        if lethal_gt:
            n = float(lethal_gt)
            agg['critical_rate'] = round(agg['lethal_to_collidable'] / n, 6)   # 会被主动撞上
            agg['severe_rate'] = round(agg['lethal_to_passable'] / n, 6)       # 会被当成地面
            agg['safe_rate'] = round((agg['lethal_to_lethal'] + agg['lethal_to_hard']) / n, 6)
        print_log('\n' + '=' * 74, logger=logger)
        print_log('安全聚合（选择性碰撞的核心指标）：致命类 GT 被判到哪一级', logger=logger)
        print_log('=' * 74, logger=logger)
        print_log(f"致命类（动物+行人）GT 共 {lethal_gt:,} 点", logger=logger)
        for k, name in (('lethal_to_lethal', '判为致命类（安全）'),
                        ('lethal_to_hard', '判为硬刚性（安全：被排除）'),
                        ('lethal_to_collidable', '**判为可撞（致命误判）**'),
                        ('lethal_to_frangible', '   其中 可破坏 frangible'),
                        ('lethal_to_deformable', '   其中 可变形 deformable'),
                        ('lethal_to_passable', '**判为可通过（会被当路面）**')):
            v = agg[k]
            print_log(f'  {name:<32}{v:>10,}  {v / max(lethal_gt, 1) * 100:>7.3f}%', logger=logger)
        if lethal_gt:
            print_log(f"  ⇒ 致命误判率 critical = {agg['critical_rate'] * 100:.3f}%"
                      f"　严重误判率 severe = {agg['severe_rate'] * 100:.3f}%"
                      f"　安全率 = {agg['safe_rate'] * 100:.3f}%", logger=logger)
        print_log('=' * 74, logger=logger)

        rows = {}
        for c in WATCH:
            gt = int(cm[c].sum())
            if gt == 0:
                print_log(f'{CLASS_NAMES[c]:<38} GT 点数为 0，不可测', logger=logger)
                rows[CLASS_NAMES[c]] = {'gt': 0, 'top': []}
                continue
            order = np.argsort(-cm[c])[:args.confusion_topk]
            top = [(CLASS_NAMES[int(j)], int(cm[c, j]),
                    round(float(cm[c, j]) / gt, 4)) for j in order if cm[c, j] > 0]
            rec = float(cm[c, c]) / gt
            print_log(f'{CLASS_NAMES[c]:<38} GT {gt:>8,}  自身召回 {rec:.4f}', logger=logger)
            for name, cnt, frac in top:
                print_log(f'      {name:<38}{cnt:>8,}  {frac * 100:>6.2f}%', logger=logger)
            rows[CLASS_NAMES[c]] = {'gt': gt, 'self_recall': rec, 'top': top}
        print_log('=' * 70, logger=logger)
        if args.confusion_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.confusion_out)), exist_ok=True)
            with open(args.confusion_out, 'w') as f:
                # 存**完整**混淆矩阵 + 安全聚合：只存罕见类 top-k 的话，
                # 「致命类 → 可撞类」这个核心安全指标就复算不出来了。
                json.dump({'exp_tag': args.exp_tag, 'ckpt': args.ckpts, 'rows': rows,
                           'safety': agg, 'classes': CLASS_NAMES,
                           'confusion_full': cm.astype(np.int64).tolist()},
                          f, indent=2, ensure_ascii=False)
            print_log(f'混淆分布已存档（含完整矩阵与安全聚合）→ {args.confusion_out}',
                      logger=logger)

    if args.out_json:
        gt_per_class = [int(conf[c].sum().item()) for c in range(num_classes)]
        result = {
            'exp_tag': args.exp_tag,
            'config': args.config,
            'ckpt': args.ckpts,
            'num_classes': num_classes,
            'miou': miou,
            'oa': oa,
            'valid_class_count': len(valid_ious),
            'per_class_iou': {CLASS_NAMES[c]: (None if np.isnan(ious[c]) else float(ious[c]))
                              for c in range(num_classes)},
            'gt_points_per_class': {CLASS_NAMES[c]: gt_per_class[c] for c in range(num_classes)},
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print_log(f'结果已存档 → {args.out_json}', logger=logger)


if __name__ == '__main__':
    main()
