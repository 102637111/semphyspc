#!/bin/bash
# 只续跑 P3（类别平衡损失 + 罕见帧过采样），S3 段已完成不再重跑。
#
# 背景：原编排脚本 /tmp/run_s3p3.sh 整条链包含「S3 微调 → 检测 → 门控投票 → 评测 → P3」，
# 2026-09-15 18:19 在 P3 的 epoch 5 结束后因重启中断。S3 段的产物（det npz、三个门控的
# bbox 缓存、results/gated_voting_s3_*.json）均已完成落盘，无需重跑，故本脚本只做 P3。
#
# 与原脚本的两点差别：
#   1. 日志写仓库内 experiments/.../resume_logs/，不写 /tmp（/tmp 重启会被清空）。
#   2. 不走 `| grep -E ...` 管道——grep 的块缓冲会让日志滞后数分钟且末行撕裂；程序自身
#      已在实验目录写完整 .log，需要过滤时事后 grep 即可。
#
# 续训安全前提（重启前已逐项核对）：
#   - ckpt-last.pth 完好（epoch=5，含 optimizer 与 best_metrics={'miou':0.2661,'oa':0.8330}）
#   - tools/runner_seg.py:60-66 会恢复 best_metrics，不会用更差的 epoch 冲掉 ckpt-best.pth
#   - main_seg.py:55-63 的 --resume 会解析到最新含 ckpt-last.pth 的目录；
#     找不到时直接 FileNotFoundError，不会静默从头训练
#
# 训练期间不要并行跑任何吃 GPU 的东西（显示器 Xorg/gnome-shell 占着同一块 5070 Ti，
# 并行会触发 Xid 8）。
set -u
cd /home/saw/ros2_ws/src/semphyspc

P3CFG=cfgs/seg_16k_ng1024_g050_p3.yaml
P3EXP=experiments/seg_16k_ng1024_g050_p3
TOTAL_EPOCHS=40

ep_seg() {  # $1 = glob；读 ckpt-last 的「下一个 epoch」，读不到吐 -1
  python3 - "$1" <<'PY'
import sys, glob, torch
cks = sorted(glob.glob(sys.argv[1]))
if not cks: print(-1); raise SystemExit
try: print(int(torch.load(cks[-1], map_location='cpu', weights_only=False).get('epoch', -1)) + 1)
except Exception: print(-1)
PY
}

LOGDIR="$P3EXP/resume_logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/resume_$(date '+%Y%m%d_%H%M%S').log"
echo "日志：$LOG"

e0=$(ep_seg "$P3EXP/cfgs/p3_ng1024_g050/*/ckpt-last.pth")
echo "########## 开始续训 P3  已完成 $e0 / $TOTAL_EPOCHS  $(date '+%F %T') ##########"

for k in 1 2 3; do
  RS=""
  [ "$(ep_seg "$P3EXP/cfgs/p3_ng1024_g050/*/ckpt-last.pth")" -ge 0 ] && RS="--resume"
  if [ -z "$RS" ]; then
    echo "!!! 找不到 ckpt-last.pth，拒绝从头训练（会白跑 40 轮）。请先确认实验目录。" | tee -a "$LOG"
    exit 1
  fi
  echo "-- P3 第 $k 次尝试  $(date '+%F %T')" | tee -a "$LOG"
  python3 -u main_seg.py --config "$P3CFG" --exp_name p3_ng1024_g050 $RS \
      --val_freq 1 --num_workers 8 >> "$LOG" 2>&1
  rc=$?
  e=$(ep_seg "$P3EXP/cfgs/p3_ng1024_g050/*/ckpt-last.pth")
  echo "-- P3 第 $k 次结束（退出码 $rc），完成到 epoch $e / $TOTAL_EPOCHS  $(date '+%F %T')" | tee -a "$LOG"
  [ "$e" -ge "$TOTAL_EPOCHS" ] && break
done

CK=$(ls -t $P3EXP/cfgs/*/*/ckpt-best.pth 2>/dev/null | head -1)
if [ -z "$CK" ]; then
  echo "!!! P3 找不到 ckpt-best.pth，跳过评测" | tee -a "$LOG"
else
  echo "P3 使用权重 $CK  $(date '+%F %T')" | tee -a "$LOG"
  python3 scripts/eval_seg_full.py --config "$P3CFG" --ckpts "$CK" \
    --exp_tag p3_ng1024_g050 --out_json results/seg_16k_g1024_g050_p3.json >> "$LOG" 2>&1
  echo "-- 评测结束（退出码 $?）  $(date '+%F %T')" | tee -a "$LOG"
fi

echo "P3_RESUME_DONE  $(date '+%F %T')" | tee -a "$LOG"
