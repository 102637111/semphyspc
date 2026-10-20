#!/bin/bash
# 重跑 2026-09-16 05:28 首轮编排里因两个 bug 失败的部分：
#   1. chain3 的 p3 混淆（原路径拼错 experiments/seg_16k_p3_ng1024_g050，应为 _p3 后缀）
#   2. chain4 的 det/safety（run_2d_detector.py `**{整型键}` 触发 TypeError）
# 均已在源码/脚本修复；本脚本按序重跑并落日志。GPU 已空闲，可立即执行。
set -u
cd /home/saw/ros2_ws/src/semphyspc

LOG="run_chains/rerun_fixes_$(date '+%Y%m%d_%H%M%S').log"
exec >> "$LOG" 2>&1

echo "########## 重跑修复项  $(date '+%F %T') ##########"

echo "########## A：罕见类混淆（含 p3，验证路径修复）  $(date '+%F %T') ##########"
bash run_chains/run_chain3.sh
echo "A 结束（退出码 $?）  $(date '+%F %T')"

echo "########## B：安全通道（det 带车辆 + 低阈值 + 延迟）  $(date '+%F %T') ##########"
bash run_chains/run_chain4.sh
echo "B 结束（退出码 $?）  $(date '+%F %T')"

echo "RERUN_FIXES_DONE  $(date '+%F %T')"
