#!/bin/bash
# B④ 材料敏感性：扰动「材料/代价先验」看导航路径鲁棒性（审稿「材料/代价先验的依据和敏感性分析」）。
# 扰动两个连续先验维度，复用方案 a 的 static map（soft_cost.yaml，lethal+hard occupied）+ 固定评估尺子
# （soft_cost_eval.npz），只换 planner 实际发布的语义代价（soft 范围 / lethality 缩放）。
#   维度1 soft 代价范围：soft_low~soft_high ∈ {10~40, 30~80(基准), 50~120}（植被/护栏软代价绝对数值）
#   维度2 lethality 缩放：材质库手工 lethality × {0.5, 1.0(基准), 1.5}（先验估计偏差）
# 指标：成功率、路径长度、语义代价、hard 侵入、soft 侵入、最小 lethal 间距（同一批 5 对起终点）。
# 纯 CPU（不占 GPU，可与训练并行）。
set -u
cd /home/saw/ros2_ws/src/semphyspc
set +u  # ROS setup.bash 在 set -u 下会因未绑定变量报错
source /opt/ros/humble/setup.bash
source /home/saw/ros2_ws/install/setup.bash 2>/dev/null || true
set -u

INPUT=paper/figures/map/domfountain_station3_map_semantic.ply
LAUNCH=launch/nav2_realmap.launch.py
MAP=/tmp/s3_nav/soft_cost.yaml           # 方案 a static map（lethal+hard occupied）
TRUTH=/tmp/s3_nav/soft_cost_eval.npz     # 固定评估尺子（独立于 planner 代价）
PAIRS_JSON=/tmp/s3_nav/pairs.json
OUT=/tmp/s3_nav

# ---- 1. 生成 5 组扰动 cost.npz（--cost-only 复用 static map；评估尺子固定不变）----
echo "########## 生成扰动代价  $(date '+%F %T') ##########"
gen() {
  local name=$1; shift
  python3 scripts/semantic3d_to_nav.py -i "$INPUT" -o "$OUT/ms_$name" \
      --xmin -25 --xmax 25 --ymin -25 --ymax 25 --cell 0.2 \
      --mode soft_cost --cost-only "$@" 2>&1 | grep -E '语义代价分层'
}
gen soft_10_40  --soft-low 10 --soft-high 40
gen soft_30_80  --soft-low 30 --soft-high 80   # 基准（= 方案 a）
gen soft_50_120 --soft-low 50 --soft-high 120
gen ls_0p5      --lethality-scale 0.5
gen ls_1p5      --lethality-scale 1.5

# ---- 清理所有 nav2_realmap 相关进程（用 [.] 技巧避免 pkill 匹配到执行它的 shell）----
cleanup() {
  pkill -f 'nav2_realmap[.]launch[.]py' 2>/dev/null
  pkill -f 'semantic_costmap_publisher[.]py' 2>/dev/null
  pkill -f 'nav2_map_server/map_server' 2>/dev/null
  pkill -f 'nav2_lifecycle_manager/lifecycle_manager' 2>/dev/null
  pkill -f 'nav2_planner/planner_server' 2>/dev/null
  pkill -f 'nav2_controller/controller_server' 2>/dev/null
  pkill -f 'nav2_smoother/smoother_server' 2>/dev/null
  pkill -f 'nav2_behaviors/behavior_server' 2>/dev/null
  pkill -f 'nav2_bt_navigator/bt_navigator' 2>/dev/null
  pkill -f 'nav2_waypoint_follower/waypoint_follower' 2>/dev/null
  pkill -f 'nav2_velocity_smoother/velocity_smoother' 2>/dev/null
  pkill -f 'tf2_ros/static_transform_publisher' 2>/dev/null
  sleep 3
}

# ---- 2. 每组跑同一批 5 对起终点 ----
run_cfg() {
  local name=$1 cost=$2
  echo "########## cfg=$name  $(date '+%F %T') ##########"
  cleanup
  setsid ros2 launch "$LAUNCH" map:="$MAP" cost_npz:="$cost" use_semantic:=true \
      > /tmp/s3_nav/launch_ms_$name.log 2>&1 &
  sleep 10   # 等 map_server + planner_server + TF 就绪
  while read -r i sx sy gx gy; do
    python3 scripts/plan_and_analyze.py --truth-npz "$TRUTH" \
        --start-x "$sx" --start-y "$sy" --goal-x "$gx" --goal-y "$gy" \
        --out "$OUT/ms_${name}_pair${i}_result.json" 2>&1 | grep -E '成功|失败|RESULT' || echo "  pair$i 无结果"
  done < <(python3 -c "
import json
for i,p in enumerate(json.load(open('$PAIRS_JSON'))):
    print(i, p[0][0], p[0][1], p[1][0], p[1][1])
")
  cleanup
}

for name in soft_10_40 soft_30_80 soft_50_120 ls_0p5 ls_1p5; do
  run_cfg "$name" "$OUT/ms_${name}_cost.npz"
done
cleanup

# ---- 3. 汇总 ----
echo "########## 汇总  $(date '+%F %T') ##########"
python3 - <<'PY'
import json
cfgs=['soft_10_40','soft_30_80','soft_50_120','ls_0p5','ls_1p5']
def rd(c,i):
    try: return json.load(open(f'/tmp/s3_nav/ms_{c}_pair{i}_result.json'))
    except: return None
print('cfg | 成功 | 平均长度(m) | 平均语义代价 | 平均hard侵入 | 平均soft侵入 | 平均lethal间距(m)')
for c in cfgs:
    rs=[rd(c,i) for i in range(5)]
    ok=[r for r in rs if r and r.get('success')]
    n=len(rs)
    def avg(k):
        v=[r[k] for r in ok if r.get(k) is not None]
        return round(sum(v)/len(v),3) if v else None
    print(f'{c:11s} | {len(ok)}/{n} | {avg("path_length_m")} | {avg("mean_cost_known")} | '
          f'{avg("n_hard")} | {avg("n_soft")} | {avg("min_dist_to_lethal_m")}')
PY
echo "ALL_MATERIAL_SENSITIVITY_DONE"
