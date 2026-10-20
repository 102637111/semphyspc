#!/usr/bin/env python3
"""语义软代价发布器：逐点 lethality → 2D 网格连续代价 → /semantic_cost（OccupancyGrid）。

把 nuScenes 32 类语义点云映射为 Nav2 costmap 的「连续软代价」（0~254），替代「非地面即障碍」的
二元语义障碍：lethal(行人/动物)=254 绝对绕开、hard(车/墙)=150~220 强避让、
frangible/deformable(植被/护栏/交通锥)=30~80 可穿越软目标、passable(地面)=0 自由。
规划器据此「绕开 person、可穿越 vegetation」——这是论文「语义物理软代价层」的落点。

输入：
  离线 --ply <语义PLY>   读 Semantic3D 语义 PLY（semantic_id 即 nuScenes 32 类 id）
  在线（默认）           订阅 --cloud-topic（带 label 字段的 PointCloud2）

输出：
  /semantic_cost  nav_msgs/OccupancyGrid（frame=map，data ∈ [-1,100]，-1 未知、100 致命）

配套 C++ 插件 semantic_costmap_layer 订阅此话题，把连续代价搬进 Nav2 costmap。

用法：
  # 离线 dry-run（不起 ROS，打印代价分层统计）
  python3 scripts/semantic_costmap_publisher.py --ply paper/figures/map/xxx_semantic.ply --dry-run
  # 离线周期发布
  python3 scripts/semantic_costmap_publisher.py --ply xxx.ply --frame-id map
  # 在线订阅带 label 点云
  python3 scripts/semantic_costmap_publisher.py --cloud-topic /semantic_cloud --label-field label
"""
import argparse
import os
import sys

import numpy as np

# 复用 semphyspc 材质库（nuScenes 32 类 lethality，脚本同目录 ../utils）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.materials import class_physics  # noqa: E402

_DT = {'float': '<f4', 'uchar': 'u1', 'int': '<i4', 'uint': '<u4',
       'short': '<i2', 'ushort': '<u2'}


def class_to_cost(cid):
    """nuScenes 32 类 id → 代价 0~254（Nav2 costmap 连续代价）。

    按 safety 分档（比按 lethality 数值更语义化，且避开 car 0.90 恰好落在 L>=0.90 边界的坑）：
    lethal(行人/动物)=254 绝对绕开、hard(车/墙)=150~220 强避让非绝对、
    frangible/deformable(植被/护栏/锥桶)=30~80 可穿越软目标、passable(地面)=0 自由。
    """
    cid = int(cid)
    ph = class_physics(cid)
    safety = ph['safety']
    if safety == 'passable':
        return 0
    if safety == 'lethal':
        return 254
    L = float(ph['lethality'])
    if safety == 'hard':
        # 方案(a)：车/墙/建筑 物理上不可穿越，标绝对障碍 254（不再软代价 185~220）
        return 254
    return int(30 + (L - 0.05) / 0.45 * 50)        # frangible/deformable 0.10~0.30 → 35~57 可穿越


def read_ply(path):
    """读二进制 PLY（Semantic3D 导出格式：xyz + rgb + semantic_id），返回结构化数组。"""
    with open(path, 'rb') as f:
        n_vertex = 0
        props = []
        while True:
            line = f.readline().decode('ascii', 'ignore').strip()
            if line.startswith('element vertex'):
                n_vertex = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                props.append((parts[-1], parts[1]))
            elif line == 'end_header':
                break
        dtype = np.dtype([(name, _DT[t]) for name, t in props])
        return np.fromfile(f, dtype=dtype, count=n_vertex)


def build_cost_grid(xyz, sem, cell_size):
    """逐点 lethality → 逐 cell 取 max → OccupancyGrid 数据。

    返回 (data, origin_x, origin_y, resolution, width, height)，data ∈ [-1,100]（-1 未知）。
    data 布局为 y-major（data[y*width+x]），origin 为左下角。
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    sem = np.asarray(sem).reshape(-1)
    if xyz.shape[0] == 0:
        return None
    x, y = xyz[:, 0], xyz[:, 1]
    xmin, ymin = float(x.min()), float(y.min())
    xmax, ymax = float(x.max()), float(y.max())
    gx = np.floor((x - xmin) / cell_size).astype(np.int64)
    gy = np.floor((y - ymin) / cell_size).astype(np.int64)
    nx = int(np.floor((xmax - xmin) / cell_size)) + 1
    ny = int(np.floor((ymax - ymin) / cell_size)) + 1
    inside = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    gx, gy, sem = gx[inside], gy[inside], sem[inside]

    # 逐点代价（0~254），逐 cell 取 max（一个 cell 里有 person + 地面 → 取 person 的 254）
    pt_cost = np.fromiter((class_to_cost(c) for c in sem), dtype=np.int32, count=sem.size)
    cell_cost = np.full(nx * ny, -1, dtype=np.int32)
    np.maximum.at(cell_cost, gy * nx + gx, pt_cost)

    # 0~254 → OccupancyGrid 0~100（-1 未知），C++ 插件读回后 ×254/100 还原
    data = np.where(cell_cost >= 0,
                    np.round(cell_cost * 100.0 / 254.0).astype(np.int8),
                    np.int8(-1))
    return data, xmin, ymin, cell_size, nx, ny


def main():
    ap = argparse.ArgumentParser(description='语义软代价发布器')
    ap.add_argument('--ply', default='', help='离线读语义 PLY（Semantic3D 格式，semantic_id=nuScenes 32 类）')
    ap.add_argument('--cost-npz', default='', help='离线读预计算语义代价 npz（cost/origin/resolution，直接发布）')
    ap.add_argument('--cloud-topic', default='/semantic_cloud', help='在线带 label 点云话题')
    ap.add_argument('--label-field', default='label', help='在线点云的语义标签字段名')
    ap.add_argument('--frame-id', default='map', help='输出 OccupancyGrid frame')
    ap.add_argument('--cell-size', type=float, default=0.5, help='软代价栅格 cell 大小（米）')
    ap.add_argument('--publish-rate', type=float, default=1.0, help='发布频率（Hz）')
    ap.add_argument('--use-sim-time', action='store_true', help='使用 /clock 仿真时间（Gazebo）')
    ap.add_argument('--dry-run', action='store_true', help='只算代价打印分层统计，不启动 ROS')
    args = ap.parse_args()

    if args.dry_run:
        if not args.ply:
            ap.error('--dry-run 需要 --ply')
        data = read_ply(args.ply)
        xyz = np.column_stack([data['x'], data['y'], data['z']]).astype(np.float32)
        sem = data['semantic_id'].astype(np.int64)
        grid = build_cost_grid(xyz, sem, args.cell_size)
        if grid is None:
            print('无有效点云')
            return
        d, ox, oy, res, nx, ny = grid
        flat = d.astype(np.int32).reshape(-1)
        known = flat[flat >= 0]
        lethal = int((known >= 100).sum())
        hard = int(((known >= 55) & (known < 100)).sum())
        soft = int(((known >= 8) & (known < 55)).sum())
        free = int((known < 8).sum())
        print(f'栅格 {nx}×{ny} @ {res}m，origin=({ox:.2f},{oy:.2f})')
        print(f'cell 代价分层：lethal(>=100)={lethal}  hard(55-99)={hard}  '
              f'soft(8-54)={soft}  free(<8)={free}  unknown={int((flat < 0).sum())}')
        # 抽查映射表
        for cid in (1, 2, 17, 9, 12, 30, 24):
            print(f'  cid={cid:2d} cost_254={class_to_cost(cid):3d} cost_100={round(class_to_cost(cid)*100/254):3d}')
        return

    # ---- ROS 模式 ----
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from nav_msgs.msg import OccupancyGrid
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header

    def transient_local_qos():
        return QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL,
                          history=HistoryPolicy.KEEP_LAST)

    class SemanticCostmapPublisher(Node):
        def __init__(self):
            super().__init__('semantic_costmap_publisher')
            self.cell_size = args.cell_size
            self.frame_id = args.frame_id
            self.label_field = args.label_field
            self._grid = None
            self.pub = self.create_publisher(OccupancyGrid, 'semantic_cost', transient_local_qos())
            self.timer = self.create_timer(1.0 / args.publish_rate, self.publish)

            if args.ply:
                self.load_ply(args.ply)
            elif args.cost_npz:
                self.load_cost_npz(args.cost_npz)
            else:
                self.sub = self.create_subscription(
                    PointCloud2, args.cloud_topic, self.on_cloud, transient_local_qos())
                self.get_logger().info(f'订阅带 label 点云 {args.cloud_topic}（在线模式）')

        def load_cost_npz(self, path):
            """读 semantic3d_to_nav.py 预计算的语义代价 npz（cost 0~254、-1 未知），转 0~100 发布。"""
            self.get_logger().info(f'读预计算语义代价：{path}')
            z = np.load(path)
            cost = z['cost']                    # (ny, nx) int16，y-major（row0=y 最小=origin）
            origin = z['origin']
            res = float(z['resolution'])
            ny, nx = cost.shape
            # 0~254 → OccupancyGrid 0~100（-1 保持未知），布局与 build_cost_grid 一致（y-major）
            data = np.where(cost >= 0,
                            np.round(cost.astype(np.float32) * 100.0 / 254.0).astype(np.int8),
                            np.int8(-1))
            self._grid = (data.reshape(-1), float(origin[0]), float(origin[1]), res, nx, ny)
            self.get_logger().info(f'软代价栅格 {nx}×{ny} @ {res}m，origin=({origin[0]:.2f},{origin[1]:.2f})')

        def load_ply(self, path):
            self.get_logger().info(f'读语义 PLY：{path}')
            data = read_ply(path)
            xyz = np.column_stack([data['x'], data['y'], data['z']]).astype(np.float32)
            sem = data['semantic_id'].astype(np.int64)
            self._grid = build_cost_grid(xyz, sem, self.cell_size)
            if self._grid:
                _, ox, oy, res, nx, ny = self._grid
                self.get_logger().info(f'软代价栅格 {nx}×{ny} @ {res}m，origin=({ox:.2f},{oy:.2f})')

        def on_cloud(self, msg):
            pts = list(point_cloud2.read_points(
                msg, field_names=['x', 'y', 'z', self.label_field], skip_nans=True))
            if not pts:
                return
            arr = np.asarray(pts, dtype=np.float32)
            xyz = arr[:, :3]
            sem = arr[:, 3].astype(np.int64)
            self._grid = build_cost_grid(xyz, sem, self.cell_size)

        def publish(self):
            if self._grid is None:
                return
            d, ox, oy, res, nx, ny = self._grid
            msg = OccupancyGrid()
            msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame_id)
            msg.info.resolution = float(res)
            msg.info.width = nx
            msg.info.height = ny
            msg.info.origin.position.x = float(ox)
            msg.info.origin.position.y = float(oy)
            msg.info.origin.position.z = 0.0
            msg.info.origin.orientation.w = 1.0
            msg.data = d.reshape(-1).tolist()
            self.pub.publish(msg)

    if args.use_sim_time:
        rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    else:
        rclpy.init()
    node = SemanticCostmapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
