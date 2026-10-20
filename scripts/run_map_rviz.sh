#!/usr/bin/env bash
# 一键启动：读地图识别结果 PLY -> RViz2（彩色点云 + 物体框 + 类别小字）。
#
# 用法（先跑 infer_map.py 生成 paper/figures/map/<场景>_map_rgb.ply + _map_semantic.ply）：
#   bash scripts/run_map_rviz.sh domfountain_station3
#
# 可选参数透传给 map_to_rviz.py：--npoints 500000 --eps 1.0 --min_points 30
set -e
SCENE="${1:?用法: run_map_rviz.sh <场景名> [额外参数]}"
shift || true

cd "$(dirname "$0")/.."
source /opt/ros/humble/setup.bash

# 清理旧的 semvis 发布节点，避免多个地图在 RViz2 里打架闪烁
pkill -f "scripts/rviz_semvis.py" 2>/dev/null || true
pkill -f "scripts/map_to_rviz.py" 2>/dev/null || true
sleep 1

/home/saw/ros2_ws/.venv/bin/python scripts/map_to_rviz.py \
    --rgb      "paper/figures/map/${SCENE}_map_rgb.ply" \
    --semantic "paper/figures/map/${SCENE}_map_semantic.ply" "$@" &
PID=$!
trap "kill $PID 2>/dev/null" EXIT

sleep 2
exec rviz2 -d scripts/rviz_semvis.rviz
