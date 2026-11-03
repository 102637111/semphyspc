#!/bin/bash
# 恢复 + 收尾链（取代原先分开的 run_s3_after_scan.sh / run_p3_after_s3.sh）。
#
# 背景：2026-09-13 15:58 ng256_g050 在第 15 轮被 Xid 8（cudaErrorLaunchTimeout）打断，
# 扫描链自动跳到下一组，但崩掉的那组没人补。这条链按顺序做四件事：
#   1. 等扫描链跑完（/tmp/scan.log 里的 ALL_SCAN_DONE）
#   2. 恢复阶段：逐个检查 5 个配置的 ckpt-last.pth，没到 40 轮的用 --resume 续上（每组最多 3 次）
#   3. 评测阶段：五组各出一份逐类 IoU（scripts/eval_seg_full.py）
#   4. S3（RT-DETR 82 类微调 + 检测/门控投票/评测）→ P3（类别平衡损失 + 罕见帧过采样）
#
# 训练期间不要再并行跑任何吃 GPU 的东西（显示器 Xorg/gnome-shell 已经占着同一块卡）。
set -u
cd /home/saw/ros2_ws/src/semphyspc
export HF_ENDPOINT=https://hf-mirror.com

MAX_EP=40
TAGS="ng256_g050 ng1024_g050 ng2048_g050 ng1024_g100 ng1024_g025"

# 读出某个实验目录下最新 ckpt-last.pth 的「下一个 epoch」；读不到就吐 -1
ep_of() {
  python3 - "$1" <<'PY'
import sys, glob, torch
cks = sorted(glob.glob(sys.argv[1]))
if not cks:
    print(-1); raise SystemExit
try:
    sd = torch.load(cks[-1], map_location='cpu', weights_only=False)
    print(int(sd.get('epoch', -1)) + 1)
except Exception as e:
    print(f'[warn] 读 ckpt 失败 {e}', file=sys.stderr)
    print(-1)
PY
}

echo "########## 等待扫描链  $(date '+%F %T') ##########"
for i in $(seq 1 4320); do
  grep -q ALL_SCAN_DONE /tmp/scan.log 2>/dev/null && break
  sleep 60
done
if ! grep -q ALL_SCAN_DONE /tmp/scan.log 2>/dev/null; then
  echo "!!! 等满 72 h 仍未见到 ALL_SCAN_DONE，按扫描中途失败处理，继续收尾"
fi

echo "########## 恢复阶段  $(date '+%F %T') ##########"
for t in $TAGS; do
  for k in 1 2 3; do
    e=$(ep_of "experiments/seg_16k_${t}/cfgs/${t}/*/ckpt-last.pth")
    echo "-- $t 第 $k 次检查：已完成到 epoch $e / $MAX_EP  $(date '+%F %T')"
    if [ "$e" -ge "$MAX_EP" ]; then break; fi
    python3 main_seg.py --config cfgs/seg_16k_$t.yaml --resume --exp_name $t \
        --val_freq 1 --num_workers 8 \
        2>&1 | grep -E '\[Validation\]|Save checkpoint|resume|Error|error|Traceback|out of memory'
    echo "-- $t 第 $k 次续训结束（退出码 ${PIPESTATUS[0]}）  $(date '+%F %T')"
  done
  e=$(ep_of "experiments/seg_16k_${t}/cfgs/${t}/*/ckpt-last.pth")
  [ "$e" -lt "$MAX_EP" ] && echo "!!! $t 三次续训后仍只到 epoch $e"
done

echo "########## 评测五组  $(date '+%F %T') ##########"
for t in $TAGS; do
  CK=$(ls -t experiments/seg_16k_${t}/cfgs/${t}/*/ckpt-best.pth 2>/dev/null | head -1)
  [ -z "$CK" ] && CK=$(ls -t experiments/seg_16k_${t}/cfgs/${t}/*/ckpt-last.pth 2>/dev/null | head -1)
  if [ -z "$CK" ]; then echo "!!! $t 找不到权重，跳过"; continue; fi
  echo "-- 评测 $t 用 $CK"
  python3 scripts/eval_seg_full.py --config cfgs/seg_16k_$t.yaml --ckpts "$CK" \
      --exp_tag "$t" --out_json results/seg_16k_$t.json
done
echo "ALL_SCAN_EVAL_DONE  $(date '+%F %T')"

# ------------------------------- S3 -------------------------------
S3OUT=experiments/s3_rtdetr_finetune
CFG=experiments/seg_nuscenes_full_16k_g256/cfgs/g256_v1/20260907-200401/config.yaml
CKPT=experiments/seg_nuscenes_full_16k_g256/cfgs/g256_v1/20260907-200401/ckpt-best.pth
DET=data/det_rtdetr_s3_val.npz

echo "########## 开始 S3  $(date '+%F %T') ##########"
for k in 1 2 3; do
  RS=""
  [ -f "$S3OUT/last.pt" ] && RS="--resume $S3OUT/last.pt"
  echo "-- S3 微调第 $k 次尝试 ${RS:-（从头）}  $(date '+%F %T')"
  python3 scripts/finetune_rtdetr_nuscenes.py \
    --labels data/s3_det_labels_train.npz \
    --dataroot /mnt/data/nuscenes \
    --out_dir "$S3OUT" \
    --epochs 16 --bs 8 --samples_per_epoch 12000 \
    --lr 2e-5 --target_per_class 4000 --class_boost 4.0 \
    --dev_frac 0.05 --dev_max_items 600 --workers 8 $RS
  rc=$?
  echo "-- S3 微调第 $k 次结束（退出码 $rc）  $(date '+%F %T')"
  # 跑到最后一个 epoch 才算完（续训会从 last.pt 的 epoch 接着走）
  e=$(python3 - "$S3OUT/last.pt" <<'PY'
import sys, os, torch
if not os.path.exists(sys.argv[1]): print(-1); raise SystemExit
try: print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('epoch', -1)) + 1)
except Exception: print(-1)
PY
)
  echo "-- S3 目前完成到 epoch $e / 16"
  [ "$e" -ge 16 ] && break
done

if [ ! -d "$S3OUT/hf_best" ]; then
  echo "!!! 没有产出 $S3OUT/hf_best，跳过 S3 的检测/投票流水线"
else
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

echo "########## 开始 P3  $(date '+%F %T') ##########"
for k in 1 2 3; do
  RS=""
  # 必须真的有 ckpt-last.pth 才敢 --resume（目录可能是上次启动即失败的残留）
  [ "$(ep_of "$P3EXP/cfgs/p3_ng1024_g050/*/ckpt-last.pth")" -ge 0 ] && RS="--resume"
  echo "-- P3 第 $k 次尝试 ${RS:-（从头）}  $(date '+%F %T')"
  python3 main_seg.py --config "$P3CFG" --exp_name p3_ng1024_g050 $RS \
      --val_freq 1 --num_workers 8 \
      2>&1 | grep -E 'RareAwareSampler|\[Validation\]|Save checkpoint|Error|Traceback|out of memory'
  e=$(ep_of "$P3EXP/cfgs/p3_ng1024_g050/*/ckpt-last.pth")
  echo "-- P3 第 $k 次结束，完成到 epoch $e / 40  $(date '+%F %T')"
  [ "$e" -ge 40 ] && break
done

CK=$(ls -t $P3EXP/cfgs/*/*/ckpt-best.pth 2>/dev/null | head -1)
if [ -z "$CK" ]; then echo "!!! P3 找不到 ckpt-best.pth"; else
  echo "P3 使用权重 $CK"
  python3 scripts/eval_seg_full.py --config "$P3CFG" --ckpts "$CK" \
    --exp_tag p3_ng1024_g050 --out_json results/seg_16k_g1024_g050_p3.json
fi

echo "ALL_CHAIN2_DONE  $(date '+%F %T')"
