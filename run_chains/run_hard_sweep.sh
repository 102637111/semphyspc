#!/bin/bash
# B③ 权衡曲线：扫描 hard 类代价上界（185~254），画「成功率 / 路径长度 / hard 避让 / lethal 间距」随 hard 权重的变化。
# 复用 B① 的 static map（仅 lethal occupied）+ 固定评估尺子（soft_cost_cost.npz，hard_ceil=220 的语义真值），
# 只换 planner 实际发布的语义代价（不同 hard_ceil），5 对起终点 × 5 个 hard_ceil。
# 纯 CPU，可与 GPU 训练并行。
set -u
cd /home/saw/ros2_ws/src/semphyspc
set +u
source /opt/ros/humble/setup.bash
source /home/saw/ros2_ws/install/setup.bash 2>/dev/null || true
set -u

PLY=paper/figures/map/domfountain_station3_map_semantic.ply
MAP=/tmp/s3_nav/soft_cost.yaml           # 复用 B① 的 static map（仅 lethal occupied）
TRUTH=/tmp/s3_nav/soft_cost_cost.npz     # 固定评估尺子（hard_ceil=220 语义真值）
PAIRS=/tmp/s3_nav/pairs.json
OUT=/tmp/s3_nav
HC_LIST="220 235 245 250 254"

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

# ---- 生成各 hard_ceil 的 cost.npz（复用同一 static map）----
for hc in $HC_LIST; do
  echo "生成 hard_ceil=$hc 的 cost.npz ..."
  python3 scripts/semantic3d_to_nav.py -i "$PLY" -o "$OUT/hard${hc}" \
      --xmin -25 --xmax 25 --ymin -25 --ymax 25 --cell 0.2 \
      --mode soft_cost --hard-ceil "$hc" --cost-only 2>&1 | grep -E '语义代价|栅格' || true
done

# ---- 每个 hard_ceil 跑 5 对起终点规划 ----
for hc in $HC_LIST; do
  echo "########## hard_ceil=$hc  $(date '+%F %T') ##########"
  cleanup
  setsid ros2 launch launch/nav2_realmap.launch.py \
      map:="$MAP" cost_npz:="$OUT/hard${hc}_cost.npz" use_semantic:=true \
      > /tmp/s3_nav/launch_hard${hc}.log 2>&1 &
  sleep 10
  while read -r i sx sy gx gy; do
    python3 scripts/plan_and_analyze.py --truth-npz "$TRUTH" \
        --start-x "$sx" --start-y "$sy" --goal-x "$gx" --goal-y "$gy" \
        --out "$OUT/hard${hc}_pair${i}_result.json" 2>&1 | grep -E '成功|失败|RESULT' || echo "  pair$i 无结果"
  done < <(python3 -c "
import json
for i,p in enumerate(json.load(open('$PAIRS'))):
    print(i, p[0][0], p[0][1], p[1][0], p[1][1])
")
  cleanup
done
cleanup

# ---- 汇总权衡曲线 ----
echo "########## 权衡曲线汇总  $(date '+%F %T') ##########"
python3 - <<'PY'
import json
hc_list = [220, 235, 245, 250, 254]
def rd(hc, i):
    try: return json.load(open(f'/tmp/s3_nav/hard{hc}_pair{i}_result.json'))
    except: return None
print(f'{"hard_ceil":>9} | {"成功率":>6} | {"平均长度(m)":>10} | {"平均语义代价":>10} | {"平均hard侵入":>10} | {"平均lethal间距(m)":>14}')
rows = []
for hc in hc_list:
    rs = [rd(hc, i) for i in range(5)]
    ok = [r for r in rs if r and r.get('success')]
    n = len(rs)
    def avg(k):
        v = [r[k] for r in ok if r.get(k) is not None]
        return round(sum(v)/len(v), 3) if v else None
    row = dict(hc=hc, succ=len(ok), total=n, plen=avg('path_length_m'),
               cost=avg('mean_cost_known'), hard=avg('n_hard'), lethal=avg('min_dist_to_lethal_m'))
    rows.append(row)
    print(f'{hc:>9} | {len(ok)}/{n} | {row["plen"]} | {row["cost"]} | {row["hard"]} | {row["lethal"]}')
import json as _j
_j.dump(rows, open('/tmp/s3_nav/hard_sweep_summary.json', 'w'), indent=2)
PY
echo "ALL_HARD_SWEEP_DONE"
