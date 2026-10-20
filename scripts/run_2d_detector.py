#!/usr/bin/env python3
"""用**真实 2D 检测器**（COCO 预训练，零样本，不碰任何 nuScenes 标注）替换
§4.13 门控投票里来自 GT 3D 框的 2D 框，产出与 `build_2d_bbox_cache.py` **同 schema**
的缓存，使 `eval_gated_voting.py` 无需改动即可直接复用。

为什么必须换：原缓存 (`build_2d_bbox_cache.py`) 的框来自 `nusc.sample_annotation`
（真值 3D 框投影），推理阶段用真值 = 数据泄漏。本脚本的框全部来自相机图像上
COCO 预训练检测器的**预测**，不含任何 nuScenes 监督信号，是可部署设置。

两阶段（检测结果会缓存，便于反复调门控策略而不用重跑检测器）：
  阶段1  det-cache: 跑检测器 -> data/det_<model>_<split>.npz
  阶段2  voting-cache: 检测框 + 深度门控 -> npz。schema 为 9 列
         [x,y,w,h,sem_id,dmin,dmax,cam_idx,score]，比 build_2d_bbox_cache 的 8 列
         **多保留末列 score**（融合阶段做 2D 分数扫描用）；下游 reader 按索引读
         row[0..7]，8/9 列完全向后兼容。

深度门控（2D 检测器**不给深度**，必须自己估，这是与 GT 版最大的区别）：
  --gate-mode lidar : 框内 LiDAR 点的深度分位数 [p05, p90]（推荐，用真实传感器测距）
  --gate-mode size  : 由框高反推深度 d = f_y * H_class / h_px（纯单目，不依赖 LiDAR）
  --gate-mode none  : 不设深度门控（下界对照，复现"无门控 precision 0.358"）

类别映射（COCO -> nuScenes 32 类 id）：
  person     -> 2  (human.pedestrian.adult；COCO 无行人子类粒度，
                    评测侧用 --group-refine 让 3D 在 2..8 内细化子类)
  bicycle    -> 14 (vehicle.bicycle)
  motorcycle -> 21 (vehicle.motorcycle)
  COCO 无 barrier(9) / trafficcone(12) 对应类 -> 零样本设置下覆盖不到，如实报告

用法:
    # 先跑检测（约 30-40 分钟 / val 6019 帧 x 6 相机）
    python3 scripts/run_2d_detector.py --stage det \
        --dataroot /mnt/data/nuscenes --version v1.0-trainval \
        --proj data/nuscenes_projection_index.npz \
        --split-file data/nuscenes_seg_full_16k/val \
        --out data/det_rtdetr_val.npz

    # 再转成投票缓存（秒级，可反复调参）
    python3 scripts/run_2d_detector.py --stage voting \
        --dataroot /mnt/data/nuscenes \
        --det data/det_rtdetr_val.npz \
        --proj data/nuscenes_projection_index.npz \
        --gate-mode lidar \
        --out data/nuscenes_2d_bbox_cache_rtdetr_val.npz
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

CAM_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
IMG_W, IMG_H = 1600, 900

# COCO 类名 -> nuScenes 32 类分割 id（仅保留 nuScenes 罕见类里有对应语义的）
# 注意命名有两套：COCO 官方 'motorcycle'/'airplane' vs Pascal VOC 风格
# 'motorbike'/'aeroplane'。PekingU/rtdetr_* 用的是后者，两个别名都要收。
COCO_TO_NUSC = {
    'person': 2,        # human.pedestrian.adult
    'bicycle': 14,      # vehicle.bicycle
    'motorcycle': 21,   # vehicle.motorcycle
    'motorbike': 21,    # 同上（VOC 命名别名）
    # 下列两类是 S3（在 nuScenes train 上微调）新增的，COCO 零样本模型不会输出
    'barrier': 9,       # movable_object.barrier
    'trafficcone': 12,  # movable_object.trafficcone
}

# 深度门控 size 模式用的类别真实高度先验（米）
CLASS_HEIGHT_M = {2: 1.70, 14: 1.50, 21: 1.60, 9: 1.00, 12: 0.60}


# ----------------------------------------------------------------------------
# 阶段 1：跑检测器
# ----------------------------------------------------------------------------

def build_token_to_images(nusc, tokens):
    """lidar_token -> [6 个相机图像的绝对路径]（按 CAM_CHANNELS 顺序）。"""
    out = {}
    for tok in tokens:
        sd = nusc.get('sample_data', tok)
        sample = nusc.get('sample', sd['sample_token'])
        paths = []
        for cam in CAM_CHANNELS:
            cam_sd = nusc.get('sample_data', sample['data'][cam])
            paths.append(os.path.join(nusc.dataroot, cam_sd['filename']))
        out[tok] = paths
    return out


def stage_det(args):
    import torch
    from PIL import Image
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    from nuscenes.nuscenes import NuScenes

    proj = np.load(args.proj, allow_pickle=True)
    all_tokens = [str(t) for t in proj['tokens']]

    # 只跑指定 split 的帧（split-file 是分割数据集里 val 目录，文件名即 lidar token）
    if args.split_file:
        want = {f[:-4] for f in os.listdir(args.split_file) if f.endswith('.npz')}
        tokens = [t for t in all_tokens if t in want]
    else:
        tokens = all_tokens
    print(f'[det] 待推理帧数 {len(tokens)}（总索引 {len(all_tokens)}）', flush=True)

    print(f'[det] 加载检测器 {args.model} ...', flush=True)
    proc = RTDetrImageProcessor.from_pretrained(args.model)
    model = RTDetrForObjectDetection.from_pretrained(args.model)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # COCO 关注类的 label id
    want_ids = {i: n for i, n in model.config.id2label.items() if n in COCO_TO_NUSC}
    print(f'[det] 目标类: {want_ids}', flush=True)

    print('[det] 建立 lidar_token -> 图像路径映射 ...', flush=True)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    tok2imgs = build_token_to_images(nusc, tokens)

    det_by_tok = {}
    bs = args.batch_size
    from tqdm import tqdm
    for i in tqdm(range(0, len(tokens), bs), desc='detecting'):
        chunk = tokens[i:i + bs]
        # 展平成 (帧 x 6 相机) 一批
        flat = []
        for tok in chunk:
            for ci, p in enumerate(tok2imgs[tok]):
                flat.append((tok, ci, p))
        imgs = []
        for _, _, p in flat:
            with Image.open(p) as im:
                imgs.append(im.convert('RGB').copy())

        inputs = proc(images=imgs, return_tensors='pt').to(device)
        with torch.no_grad():
            outs = model(**inputs)
        target_sizes = torch.tensor([[IMG_H, IMG_W]] * len(imgs), device=device)
        results = proc.post_process_object_detection(
            outs, target_sizes=target_sizes, threshold=args.score_thr)

        for (tok, ci, _), res in zip(flat, results):
            rows = []
            for s, l, b in zip(res['scores'], res['labels'], res['boxes']):
                name = model.config.id2label[int(l)]
                if name not in COCO_TO_NUSC:
                    continue
                x0, y0, x1, y1 = [float(v) for v in b.tolist()]
                x0 = max(0.0, x0); y0 = max(0.0, y0)
                x1 = min(float(IMG_W - 1), x1); y1 = min(float(IMG_H - 1), y1)
                if x1 <= x0 or y1 <= y0:
                    continue
                rows.append([x0, y0, x1 - x0, y1 - y0,
                             COCO_TO_NUSC[name], float(s), float(ci)])
            if rows:
                det_by_tok.setdefault(tok, []).extend(rows)

    # 打包成紧凑 npz
    toks = sorted(det_by_tok)
    offsets = np.zeros(len(toks) + 1, dtype=np.int64)
    flat_rows = []
    for i, t in enumerate(toks):
        rows = det_by_tok[t]
        offsets[i + 1] = offsets[i] + len(rows)
        flat_rows.extend(rows)
    dets = np.array(flat_rows, dtype=np.float32) if flat_rows else np.zeros((0, 7), np.float32)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(args.out, tokens=np.array(toks), offsets=offsets, dets=dets)
    print(f'\n[det] 保存 -> {args.out}', flush=True)
    print(f'[det] 有检测的帧 {len(toks)} / {len(tokens)}，检测框总数 {len(dets)}', flush=True)
    if len(dets):
        sem = dets[:, 4].astype(int)
        for sid in sorted(set(sem)):
            print(f'    sem_id {sid:>2}: {(sem == sid).sum():,} 框', flush=True)


# ----------------------------------------------------------------------------
# 阶段 2：检测框 + 深度门控 -> 投票缓存
# ----------------------------------------------------------------------------

def load_lidar_points(nusc, lidar_token):
    """返回该帧 LiDAR 点在 lidar 系的 (N,3) 坐标（用它做深度门控，非 GT）。"""
    from nuscenes.utils.data_classes import LidarPointCloud
    sd = nusc.get('sample_data', lidar_token)
    p = os.path.join(nusc.dataroot, sd['filename'])
    pc = LidarPointCloud.from_file(p)
    return pc.points[:3, :].T.astype(np.float64)  # (N,3)


def project_pts(xyz, K, T):
    """lidar xyz (N,3) -> (u, v, depth, valid)。"""
    N = len(xyz)
    pts_h = np.concatenate([xyz, np.ones((N, 1))], axis=1)
    pts_cam = (T @ pts_h.T).T
    depth = pts_cam[:, 2]
    uv = K @ pts_cam[:, :3].T
    u = uv[0] / uv[2]
    v = uv[1] / uv[2]
    valid = (depth > 0.5) & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    return u, v, depth, valid


def stage_voting(args):
    from nuscenes.nuscenes import NuScenes

    proj = np.load(args.proj, allow_pickle=True)
    proj_idx = {str(t): i for i, t in enumerate(proj['tokens'])}
    K_all = proj['K']
    T_all = proj['T']

    det = np.load(args.det, allow_pickle=True)
    det_tokens = [str(t) for t in det['tokens']]
    det_offsets = det['offsets']
    dets = det['dets']

    print(f'[voting] 门控模式 = {args.gate_mode}', flush=True)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    from tqdm import tqdm
    out_by_tok = {}
    n_drop = 0
    n_fallback = 0
    for i, tok in enumerate(tqdm(det_tokens, desc='gating')):
        if tok not in proj_idx:
            continue
        Ks = K_all[proj_idx[tok]].astype(np.float64)  # (6,3,3)
        Ts = T_all[proj_idx[tok]].astype(np.float64)  # (6,4,4)
        rows = dets[det_offsets[i]:det_offsets[i + 1]]

        xyz = None
        if args.gate_mode == 'lidar':
            xyz = load_lidar_points(nusc, tok)
            # 预算一次 6 相机的投影，避免每个框重算
            proj_cache = []
            for ci in range(6):
                proj_cache.append(project_pts(xyz, Ks[ci], Ts[ci]))

        kept = []
        for r in rows:
            x, y, w, h, sem_id, score, cam_idx = r
            sem_id = int(sem_id); cam_idx = int(cam_idx)
            if score < args.min_score:
                continue
            if args.gate_mode == 'none':
                dmin, dmax = -1e9, 1e9
            elif args.gate_mode == 'size':
                fy = float(Ks[cam_idx][1, 1])
                H = CLASS_HEIGHT_M.get(sem_id, 1.7)
                if h < args.min_box_h:
                    n_drop += 1
                    continue
                d_hat = fy * H / float(h)
                dmin, dmax = d_hat * args.size_lo, d_hat * args.size_hi
            else:  # lidar
                u, v, depth, valid = proj_cache[cam_idx]
                inb = valid & (u >= x) & (u < x + w) & (v >= y) & (v < y + h)
                if inb.sum() >= args.min_pts:
                    dd = depth[inb]
                    dmin = float(np.percentile(dd, args.pt_lo))
                    dmax = float(np.percentile(dd, args.pt_hi))
                elif args.fallback_size and h >= args.min_box_h:
                    # 框内 LiDAR 点太少（远处行人等小目标，正是罕见类）：
                    # 退回单目高度先验估深以保召回，否则这些目标会被整体丢弃
                    fy = float(Ks[cam_idx][1, 1])
                    d_hat = fy * CLASS_HEIGHT_M.get(sem_id, 1.7) / float(h)
                    dmin, dmax = d_hat * args.size_lo, d_hat * args.size_hi
                    n_fallback += 1
                else:
                    n_drop += 1
                    continue
            # 保留 score（第 9 列）：融合阶段才能做「2D 分数足够高才覆盖」与 score 扫描。
            # 旧 schema 8 列 [x,y,w,h,sem_id,dmin,dmax,cam_idx]，此处追加 score 到末列；
            # 下游 reader 均按索引读 row[0..7]，9 列完全向后兼容。
            kept.append([x, y, w, h, sem_id, dmin, dmax, cam_idx, float(score)])

        if kept:
            out_by_tok[tok] = kept

    toks = sorted(out_by_tok)
    offsets = np.zeros(len(toks) + 1, dtype=np.int64)
    boxes_list = []
    for i, t in enumerate(toks):
        rows = out_by_tok[t]
        offsets[i + 1] = offsets[i] + len(rows)
        boxes_list.extend(rows)

    boxes = np.array(boxes_list, dtype=np.float32) if boxes_list else np.zeros((0, 9), np.float32)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(args.out, tokens=np.array(toks), offsets=offsets,
                        boxes=boxes,
                        category_name=np.array([''] * len(boxes)),
                        cam_names=np.array(CAM_CHANNELS))
    print(f'\n[voting] 保存 -> {args.out}', flush=True)
    print(f'[voting] 帧数 {len(toks)}，门控后保留框 {len(boxes)}，'
          f'LiDAR 点不足回退单目 {n_fallback}，丢弃 {n_drop}', flush=True)
    if len(boxes):
        from collections import Counter
        cnt = Counter(boxes[:, 4].astype(int))
        for sid in sorted(cnt):
            print(f'    sem_id {sid:>2}: {cnt[sid]:,}', flush=True)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['det', 'voting'], required=True)

    # 公共
    ap.add_argument('--dataroot', default='/mnt/data/nuscenes')
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--proj', default='data/nuscenes_projection_index.npz')

    # 阶段1
    ap.add_argument('--model', default='PekingU/rtdetr_r50vd')
    ap.add_argument('--split-file', default=None,
                    help='分割数据集的 split 目录（如 data/nuscenes_seg_full_16k/val）')
    ap.add_argument('--score-thr', type=float, default=0.35)
    ap.add_argument('--batch-size', type=int, default=8)

    # 阶段2
    ap.add_argument('--det', default=None, help='阶段1 产出的检测缓存')
    ap.add_argument('--gate-mode', choices=['lidar', 'size', 'none'], default='lidar')
    ap.add_argument('--min-score', type=float, default=0.35)
    ap.add_argument('--min-box-h', type=float, default=25.0, help='size 模式：框高下限(px)')
    ap.add_argument('--size-lo', type=float, default=0.65)
    ap.add_argument('--size-hi', type=float, default=1.50)
    ap.add_argument('--min-pts', type=int, default=3, help='lidar 模式：框内最少点数')
    ap.add_argument('--fallback-size', action='store_true', default=True,
                    help='lidar 模式：框内点不足时退回单目高度先验估深（保召回）')
    ap.add_argument('--no-fallback-size', dest='fallback_size', action='store_false')
    ap.add_argument('--pt-lo', type=float, default=5.0)
    ap.add_argument('--pt-hi', type=float, default=90.0)

    # 安全通道用：把车辆类也放进来。默认关闭——COCO_TO_NUSC 原本只映射 5 个类，
    # 是为了给罕见类做门控（车/卡车/巴士本来就多，不需要救）。但对一个「绝不撞人和车」
    # 的安全通道来说，把 car/bus/truck 丢掉是致命的。默认关闭可保证 S2/S3 的既有对照
    # 不被污染，安全指标单独带这个开关跑。
    ap.add_argument('--with-vehicles', action='store_true',
                    help='检测阶段额外输出 car/bus/truck（安全通道用）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if args.with_vehicles:
        global COCO_TO_NUSC, CLASS_HEIGHT_M
        COCO_TO_NUSC = dict(COCO_TO_NUSC, car=17, bus=16, truck=23, trailer=22)
        CLASS_HEIGHT_M = {**CLASS_HEIGHT_M, 17: 1.60, 16: 3.20, 23: 3.40, 22: 3.50}
        print('[det] 已开启车辆类：car/bus/truck/trailer', flush=True)

    if args.stage == 'det':
        stage_det(args)
    else:
        if not args.det:
            ap.error('--stage voting 需要 --det')
        stage_voting(args)


if __name__ == '__main__':
    main()
