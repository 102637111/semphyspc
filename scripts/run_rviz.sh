#!/usr/bin/env bash
# 一键启动：semvis 节点（后台）+ RViz2（前台），直接看到彩色点云 + 物体框 + 类别小字。
#
# 用法（在项目根 /home/saw/ros2_ws/src/semphyspc/ 下）：
#   # 真值语义（最快，不需要 GPU）：
#   bash scripts/run_rviz.sh data/nuscenes_seg_full_16k/val/17bd604f6828423680434cd2e522180a.npz
#
#   # 模型预测语义（6 通道推理，需要 GPU）：
#   bash scripts/run_rviz.sh data/nuscenes_seg_full_16k/val/17bd604f6828423680434cd2e522180a.npz \
#       --config experiments/seg_nuscenes_full_16k/cfgs/singletask_sem/<时间戳>/config.yaml \
#       --ckpts  experiments/seg_nuscenes_full_16k/cfgs/singletask_sem/<时间戳>/ckpt-best.pth
#
# 可选参数透传给 rviz_semvis.py：--frame map --eps 1.0 --min_points 10 --rate 1.0
set -e
NPZ="${1:?用法: run_rviz.sh <npz> [--config <cfg> --ckpts <ckpt>] [--frame map] ...}"
shift || true

cd "$(dirname "$0")/.."
source /opt/ros/humble/setup.bash

# 清理旧的 semvis 发布节点，避免多个地图在 RViz2 里打架闪烁
pkill -f "scripts/rviz_semvis.py" 2>/dev/null || true
pkill -f "scripts/map_to_rviz.py" 2>/dev/null || true
sleep 1

/home/saw/ros2_ws/.venv/bin/python scripts/rviz_semvis.py --npz "$NPZ" "$@" &
PID=$!
trap "kill $PID 2>/dev/null" EXIT

sleep 2
exec rviz2 -d scripts/rviz_semvis.rviz
