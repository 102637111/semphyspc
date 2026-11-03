#!/bin/bash
# c135（3×RTX 2080 Ti）3-GPU 并行训练启动器：每张卡跑一个实验，互不干扰。
#
# 用法：
#   bash run_chains/c135_launch_3gpu.sh <config> <exp_name_base> "<seed1 seed2 seed3>" [max_epoch]
#
# 例（S0 三种子）：
#   bash run_chains/c135_launch_3gpu.sh cfgs/seg_16k_ng6144_g050_merged_c135.yaml s0_seed "0 1 2" 40
#
# 说明：
#   - 每个 seed 独占一张 GPU（CUDA_VISIBLE_DEVICES=<i>），协议与本地 4090 单卡完全一致
#     （total_bs=1 × step_per_update=8），不引入 DataParallel/DDP 的批量语义差异。
#   - nohup 后台运行，断开 SSH 不受影响；日志写到 <exp_dir>/train_<seed>.log。
#   - 查看状态：bash run_chains/c135_launch_3gpu.sh status
#   - 停止：     bash run_chains/c135_launch_3gpu.sh stop
set -u

PYTHON="$HOME/miniconda3/envs/semphyspc/bin/python"
cd "$HOME/semphyspc"

list_pids() {
  pgrep -f "main_seg.py --config" 2>/dev/null | sort -n
}

case "${1:-launch}" in
  status)
    echo "=== c135 训练进程 ==="
    ps -o pid,etime,cmd -p $(list_pids) 2>/dev/null | grep -v "^\s*PID" | sed 's/  */ /g' | cut -c1-140 || echo "无运行中的训练"
    echo "=== 各 GPU 占用 ==="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    ;;
  stop)
    echo "停止所有训练进程 ..."
    pkill -f "main_seg.py --config" 2>/dev/null && echo "已发送 SIGTERM" || echo "无进程"
    ;;
  *)
    CFG="$1"
    EXP_BASE="$2"
    SEEDS="$3"
    MAX_EP="${4:-40}"
    STAMP="$(date '+%Y%m%d_%H%M%S')"

    i=0
    for s in $SEEDS; do
      GPU=$i
      EXP="${EXP_BASE}${s}"
      LOGDIR="experiments/$(basename "$CFG" .yaml)/cfgs/$EXP"
      mkdir -p "$LOGDIR"
      LOG="$LOGDIR/train_${STAMP}.log"

      echo "启动 seed=$s -> GPU=$GPU | exp=$EXP | 日志=$LOG"
      CUDA_VISIBLE_DEVICES=$GPU \
        nohup "$PYTHON" main_seg.py \
          --config "$CFG" --exp_name "$EXP" --seed "$s" \
          --val_freq 1 --num_workers 4 \
          > "$LOG" 2>&1 &
      echo "  PID=$!"
      i=$((i+1))
    done
    echo "全部启动完成。查看：bash run_chains/c135_launch_3gpu.sh status"
    ;;
esac
