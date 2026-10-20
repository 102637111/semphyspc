#!/usr/bin/env bash
# 一键启动 Nav2 离线导航仿真：加载 2D 地图 + Nav2 规划栈 + RViz2。
#
# 用法（在项目根 /home/saw/ros2_ws/src/semphyspc/ 下）：
#   bash scripts/run_nav2_offline_sim.sh                            # 用默认 fishbot room.yaml
#   bash scripts/run_nav2_offline_sim.sh /path/to/map.yaml          # 换成自己的地图
#   bash scripts/run_nav2_offline_sim.sh /path/to/map.yaml /path/to/nav2_params.yaml
#                                                                    # 第 2 参指定 Nav2 参数文件
#
# 语义障碍 + 坡度接入（增强参数 params/nav2_params_semantic.yaml）：
#   bash scripts/run_nav2_offline_sim.sh /path/to/map.yaml "$(pwd)/params/nav2_params_semantic.yaml"
#   # 另开终端跑坡度/语义障碍节点（见 docs/nav2_offline_sim.md 第 8 节）
#
# 启动后在 RViz2 里：
#   1. 顶部工具「2D Pose Estimate」在地图可通行处点一下 + 拖方向 → 设起点（AMCL 定位）
#   2. 顶部工具「Nav2 Goal」点目标位置 + 拖方向 → 规划路径
#   3. 观察绿色路径 + 局部/全局 costmap 更新
#
# 后续接真实机器人 / 真实地图，见 docs/nav2_offline_sim.md。
set -e
MAP="${1:-}"
PARAMS="${2:-}"

cd "$(dirname "$0")/.."
source /opt/ros/humble/setup.bash
source /home/saw/ros2_ws/install/setup.bash 2>/dev/null || true

LAUNCH_FILE="$(cd launch && pwd)/nav2_offline_sim.launch.py"
ARGS=()
if [ -n "$MAP" ]; then
    ARGS+=( "map:=$MAP" )
fi
if [ -n "$PARAMS" ]; then
    ARGS+=( "params_file:=$PARAMS" )
fi

echo "启动 Nav2 离线仿真：launch=$LAUNCH_FILE ${ARGS[*]:-（默认地图）}"
exec ros2 launch "$LAUNCH_FILE" "${ARGS[@]}"
