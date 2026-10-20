#!/bin/bash
# 3 种子统计（GPT 审稿意见⑤）：seg_16k_ng1024_g050（mIoU 0.3923 基线）补跑 seed 1/2，
# seed 0 已有（ng1024_g050，2026-09-13）。用于报告均值±方差。
# 串行跑，GPU 独占（训练期间不要再并行跑任何吃 GPU 的东西）。
set -u
cd /home/saw/ros2_ws/src/semphyspc
export HF_ENDPOINT=https://hf-mirror.com

LOG="run_chains/run_3seeds_$(date '+%Y%m%d_%H%M%S').log"
exec >> "$LOG" 2>&1

MAX_EP=40
CFG=cfgs/seg_16k_ng1024_g050.yaml

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

echo "########## 3 种子统计开始  $(date '+%F %T') ##########"
for s in 1 2; do
  EXP="seed$s"
  echo "########## seed=$s  $(date '+%F %T') ##########"
  for k in 1 2 3; do
    RS=""
    [ "$(ep_of "experiments/seg_16k_ng1024_g050/cfgs/$EXP/*/ckpt-last.pth")" -ge 0 ] && RS="--resume"
    echo "-- seed$s 第 $k 次尝试 ${RS:-（从头）}  $(date '+%F %T')"
    python3 main_seg.py --config "$CFG" --exp_name "$EXP" --seed "$s" $RS \
        --val_freq 1 --num_workers 8 \
        2>&1 | grep -E '\[Validation\]|Save checkpoint|resume|Error|Traceback|out of memory'
    e=$(ep_of "experiments/seg_16k_ng1024_g050/cfgs/$EXP/*/ckpt-last.pth")
    echo "-- seed$s 第 $k 次结束，完成到 epoch $e / $MAX_EP  $(date '+%F %T')"
    [ "$e" -ge "$MAX_EP" ] && break
  done
  e=$(ep_of "experiments/seg_16k_ng1024_g050/cfgs/$EXP/*/ckpt-last.pth")
  if [ "$e" -lt "$MAX_EP" ]; then echo "!!! seed$s 三次续训后仍只到 epoch $e"; continue; fi
  CK=$(ls -t experiments/seg_16k_ng1024_g050/cfgs/$EXP/*/ckpt-best.pth 2>/dev/null | head -1)
  [ -z "$CK" ] && CK=$(ls -t experiments/seg_16k_ng1024_g050/cfgs/$EXP/*/ckpt-last.pth 2>/dev/null | head -1)
  echo "-- 评测 seed$s 用 $CK"
  python3 scripts/eval_seg_full.py --config "$CFG" --ckpts "$CK" \
      --exp_tag "$EXP" --out_json results/seg_16k_ng1024_g050_$EXP.json
done
echo "ALL_3SEEDS_DONE  $(date '+%F %T')"
