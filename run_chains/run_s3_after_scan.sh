#!/bin/bash
# S3：在 num_group/grid_size 扫描结束后，用 nuScenes train 标注微调 RT-DETR（82 类），
# 再跑完整的 S2 流水线（检测 -> 深度门控投票 -> 门控投票评测），与零样本 S2 基线对照。
set -u
cd /home/saw/ros2_ws/src/semphyspc
export HF_ENDPOINT=https://hf-mirror.com

OUT=experiments/s3_rtdetr_finetune
CFG=experiments/seg_nuscenes_full_16k_g256/cfgs/g256_v1/20260907-200401/config.yaml
CKPT=experiments/seg_nuscenes_full_16k_g256/cfgs/g256_v1/20260907-200401/ckpt-best.pth
DET=data/det_rtdetr_s3_val.npz

echo "########## 等待扫描链结束  $(date '+%F %T') ##########"
for i in $(seq 1 4320); do
  grep -q ALL_SCAN_DONE /tmp/scan.log 2>/dev/null && break
  sleep 60
done
if ! grep -q ALL_SCAN_DONE /tmp/scan.log 2>/dev/null; then
  echo "!!! 等满 72 h 仍未见到 ALL_SCAN_DONE，放弃 S3"; exit 1
fi
echo "########## 扫描结束，开始 S3  $(date '+%F %T') ##########"

python3 scripts/finetune_rtdetr_nuscenes.py \
  --labels data/s3_det_labels_train.npz \
  --dataroot /mnt/data/nuscenes \
  --out_dir "$OUT" \
  --epochs 16 --bs 8 --samples_per_epoch 12000 \
  --lr 2e-5 --target_per_class 4000 --class_boost 4.0 \
  --dev_frac 0.05 --dev_max_items 600 --workers 8
echo "########## 微调结束  $(date '+%F %T')  ##########"

if [ ! -d "$OUT/hf_best" ]; then
  echo "!!! 没有产出 $OUT/hf_best，跳过 S2 流水线"; exit 1
fi

python3 scripts/run_2d_detector.py --stage det --model "$OUT/hf_best" \
  --split-file data/nuscenes_seg_full/val --out "$DET"
echo "-- det 结束 $(date '+%F %T') --"

for gm in lidar none size; do
  BB=data/nuscenes_2d_bbox_cache_rtdetr_s3_val_$gm.npz
  python3 scripts/run_2d_detector.py --stage voting --det "$DET" --gate-mode $gm --out "$BB"
  python3 scripts/eval_gated_voting.py --config "$CFG" --ckpts "$CKPT" \
    --proj data/nuscenes_projection_index.npz --bbox "$BB" \
    --thetas 0.5,0.6,0.7,0.8,0.9,0.95,1.0 --group-refine \
    --tag s3_rtdetr_$gm --out-json results/gated_voting_s3_$gm.json
  echo "-- voting/eval $gm 结束 $(date '+%F %T') --"
done

echo "ALL_S3_DONE"
