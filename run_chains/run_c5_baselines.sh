#!/bin/bash
# C⑤ 公平基线：PointNet++ 与 PointTransformer 在「同一数据协议」下训练（40 epoch）并评测，
# 与 PointMamba（merged_pretrain_ft，12 类 mIoU 0.6132）对比。
# 协议与 merged 基线完全一致：16k 点 / 6 通道 / 12 类 / AdamW lr0.001 wd0.05 /
# CosLR 40ep(5 warmup) / total_bs 2 / step_per_update 4 / grad_norm_clip 10。
# 串行跑（先 PointNet++ 后 PointTransformer），GPU 独占，可断点续训。
set -u
cd /home/saw/ros2_ws/src/semphyspc
export HF_ENDPOINT=https://hf-mirror.com

LOG="run_chains/run_c5_baselines_$(date '+%Y%m%d_%H%M%S').log"
exec >> "$LOG" 2>&1

MAX_EP=40

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

run_one() {
  local CFG="$1" EXP="$2" EBS="$3"
  echo "########## $EXP 训练开始  $(date '+%F %T') ##########"
  for k in 1 2 3 4 5; do
    RS=""
    [ "$(ep_of "experiments/$(basename "$CFG" .yaml)/cfgs/$EXP/*/ckpt-last.pth")" -ge 0 ] && RS="--resume"
    echo "-- $EXP 第 $k 次尝试 ${RS:-（从头）}  $(date '+%F %T')"
    python3 main_seg.py --config "$CFG" --exp_name "$EXP" $RS \
        --val_freq 1 --num_workers 8 \
        2>&1 | grep -E '\[Validation\]|Save checkpoint|resume|Error|Traceback|out of memory'
    e=$(ep_of "experiments/$(basename "$CFG" .yaml)/cfgs/$EXP/*/ckpt-last.pth")
    echo "-- $EXP 第 $k 次结束，完成到 epoch $e / $MAX_EP  $(date '+%F %T')"
    [ "$e" -ge "$MAX_EP" ] && break
  done

  e=$(ep_of "experiments/$(basename "$CFG" .yaml)/cfgs/$EXP/*/ckpt-last.pth")
  if [ "$e" -lt "$MAX_EP" ]; then
    echo "!!! $EXP 五次续训后仍只到 epoch $e，跳过评测"
    return
  fi

  CK=$(ls -t experiments/$(basename "$CFG" .yaml)/cfgs/$EXP/*/ckpt-best.pth 2>/dev/null | head -1)
  [ -z "$CK" ] && CK=$(ls -t experiments/$(basename "$CFG" .yaml)/cfgs/$EXP/*/ckpt-last.pth 2>/dev/null | head -1)
  echo "-- $EXP 评测用 $CK  $(date '+%F %T')"
  python3 scripts/eval_seg_full.py --config "$CFG" --ckpts "$CK" --bs "$EBS" \
      --exp_tag "$EXP" --out_json "results/seg_16k_${EXP}.json"
}

echo "########## C⑤ 公平基线开始  $(date '+%F %T') ##########"

run_one cfgs/seg_16k_pointnet2.yaml seg_16k_pointnet2 8
run_one cfgs/seg_16k_pointtransformer.yaml seg_16k_pointtransformer 4

echo "ALL_C5_BASELINES_DONE  $(date '+%F %T')"
