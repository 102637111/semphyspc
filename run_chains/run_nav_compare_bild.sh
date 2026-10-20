#!/bin/bash
# B① 导航三组对照（植被密集场景 bildstein_station1）：soft_cost（本文）/ geometric / semantic_ban
# 三种静态地图 + 同一 Nav2 planner，统一用语义真值代价网格（soft_cost_eval.npz）评估路径。
# 与 run_nav_compare.sh 的差异：换植被密集场景 bildstein_station1，子区域 X[-25,25] Y[-28,22]，
# origin=(-25,-28)、res=0.2、cell=0.2。纯 CPU（不占 GPU）。
set -u
cd /home/saw/ros2_ws/src/semphyspc
set +u  # ROS setup.bash 在 set -u 下会因未绑定变量报错
source /opt/ros/humble/setup.bash
source /home/saw/ros2_ws/install/setup.bash 2>/dev/null || true
set -u

LAUNCH=launch/nav2_realmap.launch.py
TRUTH=/tmp/s3_nav_bild/soft_cost_eval.npz   # 评估尺子（lethal=254/hard=200/soft=46/free=0，独立于 planner 代价）
OUT=/tmp/s3_nav_bild
PAIRS_JSON=/tmp/s3_nav_bild/pairs.json

# ---- 选 5 对起终点：都在 soft_cost 最大 free 连通块内，间距 15~25m ----
python3 - <<'PY'
import numpy as np, json, random
from scipy import ndimage
def load_pgm(path):
    with open(path,'rb') as f:
        magic=f.readline().decode().strip()
        w,h=map(int,f.readline().decode().split())
        mx=f.readline().decode().strip()
        return np.frombuffer(f.read(),dtype=np.uint8).reshape(h,w)
soft = load_pgm('/tmp/s3_nav_bild/soft_cost.pgm')[::-1]   # y-major row0=y最小
free = soft == 254
lbl, nl = ndimage.label(free)
sizes = ndimage.sum(free, lbl, range(1, nl+1))
main = lbl == sizes.argmax()+1
ys, xs = np.where(main)
res, ox, oy = 0.2, -25.0, -28.0
def world(gx, gy): return (round(ox+(gx+0.5)*res,3), round(oy+(gy+0.5)*res,3))
random.seed(42)
pairs=[]
idx=list(range(len(xs)))
for _ in range(500):
    i,j = random.sample(idx,2)
    x1,y1 = world(xs[i],ys[i]); x2,y2 = world(xs[j],ys[j])
    if 15 <= np.hypot(x2-x1,y2-y1) <= 25:
        pairs.append([(x1,y1),(x2,y2)])
    if len(pairs)>=5: break
print(f'选定 {len(pairs)} 对起终点（soft_cost 最大 free 块内，间距 15~25m）')
json.dump(pairs, open('/tmp/s3_nav_bild/pairs.json','w'), indent=2)
for p in pairs:
    print(' ', p)
PY

# 清理所有 nav2_realmap 相关进程（用 [.] 技巧避免 pkill 匹配到执行它的 shell）
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

run_mode() {
  local mode=$1 map=$2 cost=$3 use_sem=$4
  echo "########## mode=$mode  $(date '+%F %T') ##########"
  cleanup
  setsid ros2 launch "$LAUNCH" map:="$map" cost_npz:="$cost" use_semantic:="$use_sem" \
      > /tmp/s3_nav_bild/launch_$mode.log 2>&1 &
  sleep 10   # 等 map_server + planner_server + TF 就绪
  while read -r i sx sy gx gy; do
    echo "  --- $mode pair$i: ($sx,$sy) -> ($gx,$gy) ---"
    python3 scripts/plan_and_analyze.py --truth-npz "$TRUTH" \
        --start-x "$sx" --start-y "$sy" --goal-x "$gx" --goal-y "$gy" \
        --out "$OUT/${mode}_pair${i}_result.json" 2>&1 | grep -E '成功|失败|RESULT' || echo "  pair$i 无结果"
  done < <(python3 -c "
import json
for i,p in enumerate(json.load(open('$PAIRS_JSON'))):
    print(i, p[0][0], p[0][1], p[1][0], p[1][1])
")
  cleanup
}

run_mode soft_cost    /tmp/s3_nav_bild/soft_cost.yaml    /tmp/s3_nav_bild/soft_cost_cost.npz true
run_mode geometric    /tmp/s3_nav_bild/geometric.yaml    /tmp/s3_nav_bild/soft_cost_cost.npz false
run_mode semantic_ban /tmp/s3_nav_bild/semantic_ban.yaml /tmp/s3_nav_bild/soft_cost_cost.npz false
cleanup

echo "########## 汇总  $(date '+%F %T') ##########"
python3 - <<'PY'
import json, glob
modes=['soft_cost','geometric','semantic_ban']
def rd(m,i):
    try: return json.load(open(f'/tmp/s3_nav_bild/{m}_pair{i}_result.json'))
    except: return None
print('mode | 成功 | 平均长度(m) | 平均语义代价 | 平均最小lethal间距(m) | 平均hard侵入')
for m in modes:
    rs=[rd(m,i) for i in range(5)]
    ok=[r for r in rs if r and r.get('success')]
    n=len(rs)
    def avg(k):
        v=[r[k] for r in ok if r.get(k) is not None]
        return round(sum(v)/len(v),3) if v else None
    print(f'{m:12s} | {len(ok)}/{n} | {avg("path_length_m")} | {avg("mean_cost_known")} | {avg("min_dist_to_lethal_m")} | {avg("n_hard")}')
PY
echo "ALL_NAV_COMPARE_BILD_DONE"
