#!/usr/bin/env python3
"""发 Nav2 goal 并等待结果（阶段3 截图闭环：A(0,0) -> B(8,0) 绕 person 穿 vegetation）。

用法：
  python3 scripts/send_nav_goal.py --x 8.0 --y 0.0
退出码 0=SUCCEEDED(4)，非 0=失败。
"""
import argparse

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose


class GoalSender(Node):
    def __init__(self):
        super().__init__('send_nav_goal')
        self.client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

    def run(self, x, y):
        self.get_logger().info('等待 /navigate_to_pose action server ...')
        if not self.client.wait_for_server(timeout_sec=120.0):
            self.get_logger().error('action server 超时未就绪')
            return 1
        self.get_logger().info('server 就绪，发 goal (%.1f, %.1f) ...' % (x, y))
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.w = 1.0
        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=30.0)
        gh = send_future.result()
        if gh is None or not gh.accepted:
            self.get_logger().error('goal 被拒')
            return 1
        self.get_logger().info('goal 已接受，等待结果 ...')
        result_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=300.0)
        res = result_future.result()
        if res is None:
            self.get_logger().error('未收到结果（超时）')
            return 1
        self.get_logger().info('导航结束 status=%d (4=SUCCEEDED)' % res.status)
        return 0 if res.status == 4 else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x', type=float, default=8.0)
    ap.add_argument('--y', type=float, default=0.0)
    args = ap.parse_args()
    rclpy.init()
    node = GoalSender()
    code = node.run(args.x, args.y)
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
