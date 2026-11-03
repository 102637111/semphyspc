#!/bin/bash
# §4.15 融合规则改进实验（一次 GPU 前向，同时出三条规则）。
#
# 背景：§4.15 的结论是「融合规则对所有 conf_3d < θ 的点无条件覆盖、缺少逐点可信度判断」
# 是残余瓶颈，并提出两点改进：(1) 2D 分数足够高才覆盖；(2) θ 按类自适应。
# 本脚本在最优配置（size 门控、score 阈值 0.7）上直接验证这两点。
#
# 三条规则（eval_gated_voting.py 单次推理内同时计算）：
#   rule 1 无条件覆盖：override = in_bbox & (conf3d < θ)，扫全局 θ —— 复现基线 0.1731
#   rule 2 score 门控：override = in_bbox & (score2d >= s) & (conf3d < θ)，扫 (s × θ)
#   rule 3 按类 θ：  override = in_bbox & (conf3d < θ_class[point2d])
#
# rule 3 的按类 θ 取自 size thr0.7 的逐类最优（见 §4.15 表 30 的 size 版）：
#   adult 0.95 / barrier 0.7 / trafficcone 0.95 / bicycle 0.95 / motorcycle 1.0
#   （其余罕见类 2D 从不输出，默认 1.0 不影响）。
#
# 依赖：data/bbox_s3_size_t0.7_score.npz（带 score 第 9 列，已与旧 8 列缓存逐位核对）。
# 注意：本脚本吃 GPU，必须等 P3 训练结束、GPU 空出后再跑（Xid 8 看门狗约束）。
set -u
cd /home/saw/ros2_ws/src/semphyspc

CFG=experiments/seg_nuscenes_full_16k_g256/cfgs/g256_v1/20260907-200401/config.yaml
CKPT=experiments/seg_nuscenes_full_16k_g256/cfgs/g256_v1/20260907-200401/ckpt-best.pth
BB=data/bbox_s3_size_t0.7_score.npz
OUT=results/gated_voting_s3_size_t0.7_fuserule.json

echo "########## 融合规则改进（size thr0.7 + score）  $(date '+%F %T') ##########"
python3 -u scripts/eval_gated_voting.py \
  --config "$CFG" --ckpts "$CKPT" \
  --proj data/nuscenes_projection_index.npz --bbox "$BB" \
  --thetas 0.5,0.6,0.7,0.8,0.9,0.95,1.0 \
  --score-thrs 0.7,0.75,0.8,0.85,0.9 \
  --perclass-thetas '{"2":0.95,"9":0.7,"12":0.95,"14":0.95,"21":1.0}' \
  --group-refine \
  --tag s3_rtdetr_size_t0.7_fuserule \
  --out-json "$OUT" 2>&1 | grep -aE 'mIoU\(罕见类\)|score 门控|按类自适应|θ=|IoU=|token 缺失|Traceback|Error|out of memory'

echo "FUSION_RULE_DONE  $(date '+%F %T')"
echo "结果：$OUT"
