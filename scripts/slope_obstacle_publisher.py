#!/usr/bin/env python3
"""坡度障碍发布器：从 3D 点云估计地面坡度，把「陡坡区域」发布为 Nav2 障碍点云。

用途：让地面机器人 / 无人机在规划时规避陡坡，选择坡度可通过的路线——坡度是
导航安全的重要几何约束，与语义障碍物（/pointmamba/semantic_obstacles）互补：
语义障碍挡住「是什么」，坡度挡住「能不能上」。

输入（二选一）：
  离线  --ply <语义PLY>   读 Semantic3D 语义 PLY（地面类 semantic_id ∈ {24,25,26,27}）
  在线  默认订阅 --cloud-topic（默认 /rtabmap/cloud_map），用「z 低分位」粗提地面候选

处理：地面点 → XY 栅格化 → 每 cell 协方差 PCA 估计坡度角 → 坡度角 > 阈值的 cell
      内地面点 → 发布 /slope_obstacles（PointCloud2，frame=map）。

Nav2 端：costmap 加一个 VoxelLayer/ObstacleLayer，observation_sources 含 slope_obstacles，
         clearing=False（只标记、不清除）。见 params/nav2_params_semantic.yaml。

QoS：发布用 sensor QoS（volatile）并周期重发（默认 1Hz），Nav2 观测源默认 sensor QoS
     可稳定收到；在线订阅点云用 transient_local（匹配 RTAB-Map 地图点云）。

用法：
  # 离线验证（不启动 ROS，只算坡度并打印统计）
  python3 scripts/slope_obstacle_publisher.py --ply paper/figures/map/domfountain_station3_map_semantic.ply --dry-run
  # 离线发布（周期性发布 /slope_obstacles）
  python3 scripts/slope_obstacle_publisher.py --ply paper/figures/map/domfountain_station3_map_semantic.ply
  # 在线（订阅 RTAB-Map 地图点云）
  python3 scripts/slope_obstacle_publisher.py
"""
import argparse

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

# nuScenes 32 类：可通行地面类（driveable_surface / flat.other / sidewalk / terrain）
GROUND_CLASSES = {24, 25, 26, 27}

_DT = {'float': '<f4', 'uchar': 'u1', 'int': '<i4', 'uint': '<u4',
       'short': '<i2', 'ushort': '<u2'}


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


def sensor_qos():
    return QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST)


def transient_local_qos():
    return QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST)


def estimate_slope(x, y, z, cell_size, min_points):
    """对地面点集做 XY 栅格化 + PCA 坡度估计。

    返回 (slope_deg, cid)：slope_deg 长度 = 栅格 cell 数（每 cell 坡度角），
    cid 长度 = 输入点数（每个点的 cell 编号）。坡度角 = 该 cell 点云协方差
    最小特征值方向（局部平面法向）与竖直轴 z 的夹角。
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    xmin, ymin = float(x.min()), float(y.min())
    xmax, ymax = float(x.max()), float(y.max())
    gx = np.floor((x - xmin) / cell_size).astype(np.int64)
    gy = np.floor((y - ymin) / cell_size).astype(np.int64)
    nx = int(np.floor((xmax - xmin) / cell_size)) + 1
    ny = int(np.floor((ymax - ymin) / cell_size)) + 1
    n_cells = nx * ny
    cid = gx * ny + gy
    inside = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    cid = cid[inside]
    x, y, z = x[inside], y[inside], z[inside]

    count = np.bincount(cid, minlength=n_cells)
    n = np.maximum(count, 1).astype(np.float32)
    sx = np.bincount(cid, weights=x, minlength=n_cells)
    sy = np.bincount(cid, weights=y, minlength=n_cells)
    sz = np.bincount(cid, weights=z, minlength=n_cells)
    sxx = np.bincount(cid, weights=x * x, minlength=n_cells)
    syy = np.bincount(cid, weights=y * y, minlength=n_cells)
    szz = np.bincount(cid, weights=z * z, minlength=n_cells)
    sxy = np.bincount(cid, weights=x * y, minlength=n_cells)
    sxz = np.bincount(cid, weights=x * z, minlength=n_cells)
    syz = np.bincount(cid, weights=y * z, minlength=n_cells)

    mx, my, mz = sx / n, sy / n, sz / n
    cov = np.zeros((n_cells, 3, 3), dtype=np.float32)
    cov[:, 0, 0] = sxx / n - mx * mx
    cov[:, 1, 1] = syy / n - my * my
    cov[:, 2, 2] = szz / n - mz * mz
    cov[:, 0, 1] = cov[:, 1, 0] = sxy / n - mx * my
    cov[:, 0, 2] = cov[:, 2, 0] = sxz / n - mx * mz
    cov[:, 1, 2] = cov[:, 2, 1] = syz / n - my * mz

    valid = count >= min_points
    slope_deg = np.zeros(n_cells, dtype=np.float32)
    for i in np.where(valid)[0]:
        w, v = np.linalg.eigh(cov[i])
        nz = abs(v[2, 0])  # 最小特征值对应法向量的 z 分量
        nz = min(max(nz, 0.0), 1.0)
        slope_deg[i] = np.degrees(np.arccos(nz))
    return slope_deg, cid


class SlopeObstaclePublisher(Node):
    def __init__(self, args):
        super().__init__('slope_obstacle_publisher')
        self.cell_size = args.cell_size
        self.slope_threshold = args.slope_threshold
        self.min_points = args.min_points
        self.ground_range = args.ground_range
        self.max_points = args.max_points
        self.max_obstacle_points = args.max_obstacle_points
        self.frame_id = args.frame_id
        self._steep_xyz = None  # (N,3) float32 陡坡障碍点
        self._semantic_xyz = None  # (N,3) float32 语义障碍点（离线 --publish-semantic）

        self.pub = self.create_publisher(PointCloud2, 'slope_obstacles', sensor_qos())
        self._publish_semantic = args.publish_semantic
        if self._publish_semantic:
            self.sem_pub = self.create_publisher(PointCloud2, args.semantic_topic, sensor_qos())
        self.timer = self.create_timer(1.0 / args.publish_rate, self.publish)

        if args.ply:
            self.load_ply(args.ply)
        else:
            self.sub = self.create_subscription(
                PointCloud2, args.cloud_topic, self.on_cloud, transient_local_qos())
            self.get_logger().info(f'订阅点云 {args.cloud_topic}（在线模式）')

    def load_ply(self, path):
        self.get_logger().info(f'读语义 PLY：{path}')
        data = read_ply(path)
        x = data['x'].astype(np.float32)
        y = data['y'].astype(np.float32)
        z = data['z'].astype(np.float32)
        sem = data['semantic_id'].astype(np.int64)
        is_ground = np.isin(sem, list(GROUND_CLASSES))
        self.get_logger().info(f'总点 {x.size}，地面点 {int(is_ground.sum())}')
        self.compute_steep(x[is_ground], y[is_ground], z[is_ground])
        if self._publish_semantic:
            is_obstacle = (sem != 0) & ~is_ground  # 非地面且非 noise = 语义障碍
            self.get_logger().info(f'语义障碍点 {int(is_obstacle.sum())}')
            self.extract_semantic(x[is_obstacle], y[is_obstacle], z[is_obstacle])

    def on_cloud(self, msg):
        pts = list(point_cloud2.read_points(msg, field_names=['x', 'y', 'z'], skip_nans=True))
        if not pts:
            return
        xyz = np.asarray(pts, dtype=np.float32)
        # 无语义：地面 = z 最低分位附近（室内地面为 floor，最低 0.5m 内）
        z_min = float(xyz[:, 2].min())
        is_ground = xyz[:, 2] < z_min + self.ground_range
        self.compute_steep(xyz[is_ground, 0], xyz[is_ground, 1], xyz[is_ground, 2])

    def compute_steep(self, x, y, z):
        # 地面点降采样（控制计算量与发布消息大小）
        if x.size > self.max_points:
            idx = np.random.default_rng(0).choice(x.size, self.max_points, replace=False)
            x, y, z = x[idx], y[idx], z[idx]
        slope_deg, cid = estimate_slope(x, y, z, self.cell_size, self.min_points)
        steep = slope_deg[cid] > self.slope_threshold
        xyz = np.column_stack([x[steep], y[steep], z[steep]]).astype(np.float32)
        if xyz.size and xyz.shape[0] > self.max_obstacle_points:
            idx = np.random.default_rng(1).choice(xyz.shape[0], self.max_obstacle_points, replace=False)
            xyz = xyz[idx]
        self._steep_xyz = xyz
        self.get_logger().info(
            f'坡度估计：地面点 {x.size}，陡坡障碍点 {xyz.shape[0]}（>{self.slope_threshold:.0f}°）')

    def extract_semantic(self, x, y, z):
        """离线语义障碍提取：非地面非 noise 点降采样后缓存，供周期发布。"""
        if x.size > self.max_points:
            idx = np.random.default_rng(2).choice(x.size, self.max_points, replace=False)
            x, y, z = x[idx], y[idx], z[idx]
        xyz = np.column_stack([x, y, z]).astype(np.float32)
        if xyz.shape[0] > self.max_obstacle_points:
            idx = np.random.default_rng(3).choice(xyz.shape[0], self.max_obstacle_points, replace=False)
            xyz = xyz[idx]
        self._semantic_xyz = xyz
        self.get_logger().info(f'语义障碍点（降采样后）: {xyz.shape[0]}')

    def publish(self):
        now = self.get_clock().now().to_msg()
        if self._steep_xyz is not None and self._steep_xyz.shape[0] > 0:
            h = Header(stamp=now, frame_id=self.frame_id)
            self.pub.publish(point_cloud2.create_cloud_xyz32(h, self._steep_xyz))
        if self._publish_semantic and self._semantic_xyz is not None and self._semantic_xyz.shape[0] > 0:
            h = Header(stamp=now, frame_id=self.frame_id)
            self.sem_pub.publish(point_cloud2.create_cloud_xyz32(h, self._semantic_xyz))


def main():
    ap = argparse.ArgumentParser(description='坡度障碍发布器')
    ap.add_argument('--ply', default='', help='离线读语义 PLY（Semantic3D 格式）')
    ap.add_argument('--cloud-topic', default='/rtabmap/cloud_map', help='在线点云话题')
    ap.add_argument('--frame-id', default='map', help='输出点云 frame')
    ap.add_argument('--cell-size', type=float, default=0.5, help='坡度栅格 cell 大小（米）')
    ap.add_argument('--slope-threshold', type=float, default=25.0, help='陡坡阈值角（度）')
    ap.add_argument('--min-points', type=int, default=10, help='每 cell 最少地面点数')
    ap.add_argument('--ground-range', type=float, default=0.5, help='在线模式地面候选 z 范围（米）')
    ap.add_argument('--max-points', type=int, default=1500000, help='地面点降采样上限')
    ap.add_argument('--max-obstacle-points', type=int, default=500000, help='发布障碍点上限')
    ap.add_argument('--publish-rate', type=float, default=1.0, help='发布频率（Hz）')
    ap.add_argument('--publish-semantic', action='store_true',
                    help='离线模式同时从 PLY 提取语义障碍（非地面非 noise）发布到 --semantic-topic')
    ap.add_argument('--semantic-topic', default='/pointmamba/semantic_obstacles',
                    help='语义障碍输出话题（默认对齐 pointmamba_realtime 话题，离线可模拟）')
    ap.add_argument('--dry-run', action='store_true', help='只算坡度打印统计，不启动 ROS')
    args = ap.parse_args()

    if args.dry_run:
        # 不启动 ROS：直接复用 compute 逻辑，打印坡度统计
        from collections import Counter
        data = read_ply(args.ply)
        x = data['x'].astype(np.float32)
        y = data['y'].astype(np.float32)
        z = data['z'].astype(np.float32)
        sem = data['semantic_id'].astype(np.int64)
        is_ground = np.isin(sem, list(GROUND_CLASSES))
        x, y, z = x[is_ground], y[is_ground], z[is_ground]
        if x.size > args.max_points:
            idx = np.random.default_rng(0).choice(x.size, args.max_points, replace=False)
            x, y, z = x[idx], y[idx], z[idx]
        slope_deg, cid = estimate_slope(x, y, z, args.cell_size, args.min_points)
        steep = slope_deg[cid] > args.slope_threshold
        n_steep_pts = int(steep.sum())
        n_cells = slope_deg.size
        print(f'地面点（降采样后）: {x.size}')
        print(f'栅格 cell 数: {n_cells}')
        print(f'有效 cell 数（>={args.min_points} 点）: {int((slope_deg > 0).sum())}')
        print(f'陡坡 cell 数（>{args.slope_threshold:.0f}°）: {len(set(cid[steep].tolist()))}')
        print(f'陡坡障碍点: {n_steep_pts}')
        if slope_deg.size:
            print(f'坡度分布: min={slope_deg.min():.1f}° max={slope_deg.max():.1f}° '
                  f'median={np.median(slope_deg):.1f}°')
        return

    rclpy.init()
    node = SlopeObstaclePublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
