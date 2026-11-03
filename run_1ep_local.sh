#!/bin/bash
# 本机 RTX 5070 Ti：S0/S1/S2 各跑 1 epoch（seed 0，顺序执行，~51min/epoch）
set -u
cd /home/saw/ros2_ws/src/semphyspc
MASTER=/tmp/1ep_master.log
echo "=== 1ep local start $(date) ===" >> "$MASTER"
for stage in s0 s1 s2; do
  echo "launch ${stage} $(date)" >> "$MASTER"
  python3 main_seg.py --config cfgs/_1ep_${stage}.yaml --exp_name ${stage}_1ep --seed 0 --num_workers 4 > /tmp/1ep_${stage}.out 2>&1
  echo "=== ${stage}_1ep done $(date) exit=$? ===" >> "$MASTER"
done
echo "=== ALL_1EP_DONE $(date) ===" >> "$MASTER"
