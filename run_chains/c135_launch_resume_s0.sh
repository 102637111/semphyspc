#!/bin/bash
# c135 S0 三种子续训启动器（迁移自本机 5070 Ti 的 checkpoint）。
#
#   seed0 = 本机 epoch10 续训（--resume） -> GPU0
#   seed1 = 本机 epoch6  续训（--resume） -> GPU1
#   seed2 = 从零开始（无 --resume）        -> GPU2
#
# 每卡一个实验，跑满 max_epoch=40 自动退出（main_seg.py 循环到 40 即结束）。
# 用法：
#   bash run_chains/c135_launch_resume_s0.sh          # 启动
#   bash run_chains/c135_launch_resume_s0.sh status   # 查看
#   bash run_chains/c135_launch_resume_s0.sh stop     # 停止
set -u

PYTHON="$HOME/miniconda3/envs/semphyspc/bin/python"
cd "$HOME/semphyspc"
CFG="cfgs/seg_16k_ng6144_g050_merged_c135.yaml"
STEM="seg_16k_ng6144_g050_merged_c135"

list_pids() { pgrep -f "main_seg.py --config" 2>/dev/null | sort -n; }

case "${1:-launch}" in
  status)
    echo "=== c135 训练进程 ==="
    ps -o pid,etime,cmd -p $(list_pids) 2>/dev/null | sed 's/  */ /g' | cut -c1-160 || echo "无运行中的训练"
    echo "=== 各 GPU 占用 ==="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    ;;
  stop)
    echo "停止所有训练进程 ..."
    pkill -f "main_seg.py --config" 2>/dev/null && echo "已发送 SIGTERM" || echo "无进程"
    ;;
  *)
    STAMP="$(date '+%Y%m%d_%H%M%S')"
    launch() {  # $1=seed  $2=gpu  $3=resume(1/0)
      local seed="$1" gpu="$2" resume="$3"
      local exp="s0_seed${seed}"
      local extra=""
      [ "$resume" = "1" ] && extra="--resume"
      local logdir="experiments/$STEM/cfgs/$exp"
      mkdir -p "$logdir"
      local log="$logdir/train_${STAMP}.log"
      echo "启动 seed=$seed -> GPU=$gpu | exp=$exp | resume=$([ "$resume" = 1 ] && echo 是 || echo 否) | 日志=$log"
      CUDA_VISIBLE_DEVICES=$gpu \
        nohup "$PYTHON" main_seg.py \
          --config "$CFG" --exp_name "$exp" --seed "$seed" \
          --val_freq 1 --num_workers 4 $extra \
          > "$log" 2>&1 &
      echo "  PID=$!"
    }
    launch 0 0 1
    launch 1 1 1
    launch 2 2 0
    echo "全部启动完成。查看：bash run_chains/c135_launch_resume_s0.sh status"
    ;;
esac
