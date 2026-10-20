#!/bin/bash
# S3（RT-DETR 82 类微调）→ 检测/门控投票/评测；然后 P3（类别平衡损失 + 罕见帧过采样）。
#
# 与 /tmp/run_chain2b.sh 的 S3 段的差别（2026-09-15 诊断后改写）：
#
#   崩塌根因 = transformers 5.12.1 的 RTDetrForObjectDetectionLoss 主分支**没有切掉
#   去噪 query**（loss_rt_detr.py:450 用未切分的 logits，而 :455-456 给 aux 分支切了）。
#   去噪 query 的参考点就是加噪后的 GT 框，匈牙利匹配时几乎必胜——实测主分支 90~100%
#   的正样本被配给去噪 query，于是推理时真正使用的 300 个真实 query 被大面积当背景压，
#   分类分数整体塌掉而框回归照常变好。scripts/diag_s3_collapse.py 的 600 步对照：
#
#     arm12 原版（不切分，去噪开）        8/170   thr0.05 110   ← 复现崩塌
#     arm9  关掉去噪                     77/170   thr0.05 125
#     arm11 修正切分 + 去噪开            81/170   thr0.05 134   ← 生产配置
#     （基线 66/170，thr0.05 92）
#
#   修正在 finetune_rtdetr_nuscenes.py:build_loss 的 split_dn（默认 True）里，所以本脚本
#   **不再传 --no_denoising**：去噪本身有益（arm11 > arm9），坏的只是那处未切分。
#   同时 --freeze none：崩塌与几何无关（arm7 把框损失权重归零、几何全程冻结照样崩）。
#
# 旧 last.pt 已移到 experiments/s3_rtdetr_finetune_collapsed_20260915/，
# 目录里只剩 token_to_images.json 缓存，所以本次是干净重跑（不会误续崩塌那次）。
#
# 训练期间不要并行跑任何吃 GPU 的东西（显示器 Xorg/gnome-shell 占着同一块 5070 Ti，
# 并行会触发 Xid 8）。
set -u
cd /home/saw/ros2_ws/src/semphyspc
export HF_HUB_OFFLINE=1

S3OUT=experiments/s3_rtdetr_finetune
CFG=experiments/seg_nuscenes_full_16k_g256/cfgs/g256_v1/20260907-200401/config.yaml
CKPT=experiments/seg_nuscenes_full_16k_g256/cfgs/g256_v1/20260907-200401/ckpt-best.pth
DET=data/det_rtdetr_s3_val.npz

# 读 last.pt 的「下一个 epoch」；读不到吐 -1
ep_s3() {
  python3 - "$1" <<'PY'
import sys, os, torch
if not os.path.exists(sys.argv[1]): print(-1); raise SystemExit
try: print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('epoch', -1)) + 1)
except Exception: print(-1)
PY
}
ep_seg() {  # $1 = glob
  python3 - "$1" <<'PY'
import sys, glob, torch
cks = sorted(glob.glob(sys.argv[1]))
if not cks: print(-1); raise SystemExit
try: print(int(torch.load(cks[-1], map_location='cpu', weights_only=False).get('epoch', -1)) + 1)
except Exception: print(-1)
PY
}

echo "########## 开始 S3  $(date '+%F %T') ##########"
for k in 1 2 3; do
  RS=""
  [ -f "$S3OUT/last.pt" ] && RS="--resume $S3OUT/last.pt"
  echo "-- S3 第 $k 次尝试 ${RS:-（从头）}  $(date '+%F %T')"
  python3 -u scripts/finetune_rtdetr_nuscenes.py \
    --labels data/s3_det_labels_train.npz \
    --dataroot /mnt/data/nuscenes \
    --out_dir "$S3OUT" \
    --epochs 16 --bs 8 --samples_per_epoch 12000 \
    --lr 2e-5 --target_per_class 4000 --class_boost 4.0 \
    --dev_frac 0.05 --dev_max_items 600 --workers 8 \
    --freeze none $RS \
    2>&1 | grep -E '\[dn\]|\[freeze\]|\[dev\]|epoch |dev |Error|error|Traceback|out of memory|CUDA'
  echo "-- S3 第 $k 次结束（退出码 ${PIPESTATUS[0]}）  $(date '+%F %T')"
  e=$(ep_s3 "$S3OUT/last.pt")
  echo "-- S3 完成到 epoch $e / 16"
  [ "$e" -ge 16 ] && break
done

if [ ! -d "$S3OUT/hf_best" ]; then
  echo "!!! 没有 $S3OUT/hf_best，跳过下游"
else
  echo "########## S3 下游：检测 → 门控投票 → 评测  $(date '+%F %T') ##########"
  python3 scripts/run_2d_detector.py --stage det --model "$S3OUT/hf_best" \
    --split-file data/nuscenes_seg_full/val --out "$DET"
  echo "-- det 结束  $(date '+%F %T') --"
  for gm in lidar none size; do
    BB=data/nuscenes_2d_bbox_cache_rtdetr_s3_val_$gm.npz
    python3 scripts/run_2d_detector.py --stage voting --det "$DET" --gate-mode $gm --out "$BB"
    python3 scripts/eval_gated_voting.py --config "$CFG" --ckpts "$CKPT" \
      --proj data/nuscenes_projection_index.npz --bbox "$BB" \
      --thetas 0.5,0.6,0.7,0.8,0.9,0.95,1.0 --group-refine \
      --tag s3_rtdetr_$gm --out-json results/gated_voting_s3_$gm.json
    echo "-- voting/eval $gm 结束  $(date '+%F %T') --"
  done
fi
echo "ALL_S3_DONE  $(date '+%F %T')"

# ------------------------------- P3 -------------------------------
P3CFG=cfgs/seg_16k_ng1024_g050_p3.yaml
P3EXP=experiments/seg_16k_ng1024_g050_p3

echo "########## 开始 P3（从头）  $(date '+%F %T') ##########"
for k in 1 2 3; do
  RS=""
  # 必须真的有 ckpt-last.pth 才敢 --resume（目录可能是上次启动即失败的残留）
  [ "$(ep_seg "$P3EXP/cfgs/p3_ng1024_g050/*/ckpt-last.pth")" -ge 0 ] && RS="--resume"
  echo "-- P3 第 $k 次尝试 ${RS:-（从头）}  $(date '+%F %T')"
  python3 -u main_seg.py --config "$P3CFG" --exp_name p3_ng1024_g050 $RS \
      --val_freq 1 --num_workers 8 \
      2>&1 | grep -E 'RareAwareSampler|\[Validation\]|Save checkpoint|Error|Traceback|out of memory'
  e=$(ep_seg "$P3EXP/cfgs/p3_ng1024_g050/*/ckpt-last.pth")
  echo "-- P3 第 $k 次结束，完成到 epoch $e / 40  $(date '+%F %T')"
  [ "$e" -ge 40 ] && break
done

CK=$(ls -t $P3EXP/cfgs/*/*/ckpt-best.pth 2>/dev/null | head -1)
if [ -z "$CK" ]; then echo "!!! P3 找不到 ckpt-best.pth"; else
  echo "P3 使用权重 $CK"
  python3 scripts/eval_seg_full.py --config "$P3CFG" --ckpts "$CK" \
    --exp_tag p3_ng1024_g050 --out_json results/seg_16k_g1024_g050_p3.json
fi

echo "ALL_CHAIN_S3P3_DONE  $(date '+%F %T')"
