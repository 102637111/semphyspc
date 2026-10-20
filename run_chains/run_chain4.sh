#!/bin/bash
# chain4：接在 chain3 后面。安全通道的两件事——指标与延迟。
#
# 相对第一次统计有两处修正：
#   1. det 阶段必须带 --with-vehicles，否则 run_2d_detector.py:151 的
#      `if name not in COCO_TO_NUSC: continue` 会把 car/bus/truck 全丢掉
#      （GT 里 car 有 75,176 个框，检测端一个都不吐）。
#   2. det 阶段必须用低阈值（0.05），否则 0.35 的存盘阈值把 FPPI 曲线的低段截断，
#      算出来的 recall@FPPI 只是下界。
set -u
cd /home/saw/ros2_ws/src/semphyspc
export HF_ENDPOINT=https://hf-mirror.com

# 注意：本脚本吃 GPU，必须在 P3 训练结束后再跑（Xid 8 看门狗）。
# 时序由 run_post_p3.sh 编排保证，本脚本自身不等待任何 /tmp 标记。

GT=data/s3_det_labels_val.npz
SPLIT=data/nuscenes_seg_full/val
S3OUT=experiments/s3_rtdetr_finetune

if [ ! -f "$GT" ]; then
  echo "!!! 缺少 val 2D 真值 $GT，安全指标无法计算，只做延迟"; HAVE_GT=0
else
  HAVE_GT=1
fi

# ---------- 1) S2 零样本，带车辆 + 低阈值 ----------
echo "########## S2 零样本（车辆 + 低阈值）  $(date '+%F %T') ##########"
python3 scripts/run_2d_detector.py --stage det --model PekingU/rtdetr_r50vd \
  --split-file "$SPLIT" --score-thr 0.05 --with-vehicles \
  --out data/det_rtdetr_val_veh.npz
if [ "$HAVE_GT" = 1 ]; then
  python3 scripts/safety_metrics.py --det data/det_rtdetr_val_veh.npz --gt "$GT" \
    --n_frames 6019 --tag s2_zeroshot_veh --out results/safety_s2_zeroshot_veh.json
fi

# ---------- 2) S3 微调模型，带车辆 + 低阈值 ----------
if [ -d "$S3OUT/hf_best" ]; then
  echo "########## S3 微调模型（车辆 + 低阈值）  $(date '+%F %T') ##########"
  python3 scripts/run_2d_detector.py --stage det --model "$S3OUT/hf_best" \
    --split-file "$SPLIT" --score-thr 0.05 --with-vehicles \
    --out data/det_rtdetr_s3_val_veh.npz
  if [ "$HAVE_GT" = 1 ]; then
    python3 scripts/safety_metrics.py --det data/det_rtdetr_s3_val_veh.npz --gt "$GT" \
      --n_frames 6019 --tag s3_finetuned_veh --out results/safety_s3_veh.json
  fi
  LATMODEL="$S3OUT/hf_best"; LATTAG=s3
else
  echo "!!! 没有 $S3OUT/hf_best，延迟只测零样本模型"
  LATMODEL=PekingU/rtdetr_r50vd; LATTAG=zeroshot
fi

# ---------- 3) 6 相机检测延迟实测 ----------
echo "########## 延迟实测  $(date '+%F %T') ##########"
python3 scripts/measure_det_latency.py --model "$LATMODEL" \
  --out results/det_latency_${LATTAG}.json

echo "ALL_CHAIN4_DONE  $(date '+%F %T')"
