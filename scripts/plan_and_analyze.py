#!/usr/bin/env python3
"""导航规划对比分析：调 planner_server 的 ComputePathToPose，统一用「语义真值代价网格」评估路径。

三种障碍表达方式（soft_cost / geometric / semantic_ban）生成的静态地图不同，但都跑同一个
Nav2 planner。本脚本用同一把尺子——语义真值代价网格（soft_cost 生成的 _cost.npz，逐 cell
完整语义分层 0/30~80/150~220/254）——去量每一条路径，保证三组可比。

指标：
  - 规划成功/失败（error_code==0 且 path 非空）
  - 路径长度（逐段欧氏距离和，米）、路径点数
  - 危险区侵入：路径 cell 落在 lethal(254) / hard(150~253) / soft(30~149) 的个数与占比
  - 平均语义代价（路径 cell 代价均值，unknown=-1 跳过）
  - 最小间距：路径点到最近 lethal cell 中心的最小欧氏距离（米）
  - 规划耗时（发请求→收到结果，秒）

用法（需先由 nav2_realmap.launch.py 起 planner_server + map_server + TF）：
  python3 scripts/plan_and_analyze.py \
      --truth-npz /tmp/s3_nav/soft_cost_cost.npz \
      --start-x -13.9 --start-y 3.7 --goal-x 12.5 --goal-y -3.1 \
      --out /tmp/s3_nav/soft_cost_result.json
"""
import argparse
import json
import time

import numpy as np


def load_truth(npz_path):
    z = np.load(npz_path)
    cost = z['cost'].astype(np.int32)          # (ny, nx)，-1=unknown，0~254 语义代价
    origin = z['origin'].astype(np.float64)     # [x0, y0]
    res = float(z['resolution'])
    return cost, origin, res


def path_cells(poses, origin, res, ny, nx):
    """路径点世界坐标 → cell 代价序列（越界/unknown 记为 -1）。"""
    costs = []
    for p in poses:
        x = p.pose.position.x
        y = p.pose.position.y
        gx = int(np.floor((x - origin[0]) / res))
        gy = int(np.floor((y - origin[1]) / res))
        if 0 <= gx < nx and 0 <= gy < ny:
            costs.append(int(cost_grid_cache[gy, gx]))
        else:
            costs.append(-1)
    return costs


def lethal_cell_centers(cost, origin, res):
    """cost>=254 的 cell 世界坐标中心（用于最小间距）。"""
    ys, xs = np.where(cost >= 254)
    cx = (xs + 0.5) * res + origin[0]
    cy = (ys + 0.5) * res + origin[1]
    return np.column_stack([cx, cy])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth-npz', required=True, help='语义真值代价网格（soft_cost _cost.npz）')
    ap.add_argument('--start-x', type=float, default=-13.9)
    ap.add_argument('--start-y', type=float, default=3.7)
    ap.add_argument('--goal-x', type=float, default=12.5)
    ap.add_argument('--goal-y', type=float, default=-3.1)
    ap.add_argument('--planner-id', default='GridBased')
    ap.add_argument('--out', required=True, help='结果 JSON 输出路径')
    ap.add_argument('--timeout', type=float, default=120.0, help='等 action/结果超时（秒）')
    args = ap.parse_args()

    global cost_grid_cache
    cost, origin, res = load_truth(args.truth_npz)
    ny, nx = cost.shape
    cost_grid_cache = cost

    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from nav2_msgs.action import ComputePathToPose
    from geometry_msgs.msg import PoseStamped

    rclpy.init()
    node = Node('plan_and_analyze')
    client = ActionClient(node, ComputePathToPose, '/compute_path_to_pose')

    node.get_logger().info('等待 /compute_path_to_pose ...')
    if not client.wait_for_server(timeout_sec=args.timeout):
        node.get_logger().error('planner_server 未就绪')
        rclpy.shutdown()
        return 1

    def make_pose(x, y):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = float(x)
        p.pose.position.y = float(y)
        p.pose.orientation.w = 1.0
        return p

    goal = ComputePathToPose.Goal()
    goal.start = make_pose(args.start_x, args.start_y)
    goal.goal = make_pose(args.goal_x, args.goal_y)
    goal.planner_id = args.planner_id
    goal.use_start = True

    t0 = time.time()
    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future, timeout_sec=args.timeout)
    gh = send_future.result()
    if gh is None or not gh.accepted:
        node.get_logger().error('compute path goal 被拒/超时')
        rclpy.shutdown()
        return 1

    result_future = gh.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=args.timeout)
    res_msg = result_future.result()
    dt = time.time() - t0

    out = {
        'start': [args.start_x, args.start_y],
        'goal': [args.goal_x, args.goal_y],
        'planner_id': args.planner_id,
        'planning_time_s': round(dt, 3),
    }

    # ComputePathToPose.result 只有 path + planning_time，无 error_code；
    # 规划失败 = 结果超时 / goal 被 abort / path 为空。
    poses = res_msg.result.path.poses if res_msg else []
    out['success'] = bool(poses)
    out['n_points'] = len(poses)

    if not out['success']:
        node.get_logger().warn('规划失败：path 为空（planner 无法连通起点→终点）')
    else:
        # 路径长度
        xy = np.array([[p.pose.position.x, p.pose.position.y] for p in poses])
        seg = np.diff(xy, axis=0)
        out['path_length_m'] = round(float(np.sqrt((seg ** 2).sum(axis=1)).sum()), 3)

        # 路径 cell 代价分层（统一语义真值尺子）
        c = np.array(path_cells(poses, origin, res, ny, nx), dtype=np.int32)
        known = c[c >= 0]
        out['n_lethal'] = int((c >= 254).sum())
        out['n_hard'] = int(((c >= 150) & (c < 254)).sum())
        out['n_soft'] = int(((c >= 30) & (c < 150)).sum())
        out['n_free'] = int(((c >= 0) & (c < 30)).sum())
        out['n_unknown'] = int((c < 0).sum())
        out['mean_cost_known'] = round(float(known.mean()), 2) if known.size else None

        # 最小间距到 lethal cell
        lethal = lethal_cell_centers(cost, origin, res)
        if lethal.size:
            d = np.sqrt(((xy[:, None, :] - lethal[None, :, :]) ** 2).sum(axis=2))
            out['min_dist_to_lethal_m'] = round(float(d.min()), 3)
        else:
            out['min_dist_to_lethal_m'] = None

        node.get_logger().info(
            f'成功：长度 {out["path_length_m"]}m，点数 {out["n_points"]}，'
            f'lethal={out["n_lethal"]} hard={out["n_hard"]} soft={out["n_soft"]}，'
            f'最小 lethal 间距 {out["min_dist_to_lethal_m"]}m')

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('RESULT ' + json.dumps(out, ensure_ascii=False))

    node.destroy_node()
    rclpy.shutdown()
    return 0 if out.get('success') else 2


if __name__ == '__main__':
    raise SystemExit(main())
