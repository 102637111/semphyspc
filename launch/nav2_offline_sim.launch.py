#!/usr/bin/env python3
"""Nav2 导航仿真（离线验证 / 对接真实机器人，两种定位模式）。

模式 A（use_amcl:=true，默认）：离线仿真
    加载 2D 占据栅格地图 → map_server + AMCL 定位 → Nav2 规划栈 → RViz2。
    用假 TF（odom→base_footprint 恒等）与假激光（/scan 全 max_range）替代真实底盘/雷达，
    可在 RViz2 里用「2D Pose Estimate」设起点、「Nav2 Goal」设目标验证规划。

模式 B（use_amcl:=false）：对接真实机器人视觉定位（对齐 orbslam3_rgbd_navigation.launch.py）
    不启动 map_server/AMCL，地图与 TF 由外部提供——真实机器人用 ORB-SLAM3 / RTAB-Map
    视觉 SLAM 定位（publish_tf_map/odom:=true，发布 map→odom→base 的 TF 与 /map 地图），
    Nav2 只跑规划栈。此模式下假 TF / 假激光自动停用。

用法（任意路径均可，本 launch 不依赖 ROS 包）：
  source /opt/ros/humble/setup.bash
  # 离线仿真
  ros2 launch <本文件路径> map:=/path/to/map.yaml
  # 对接真实机器人（TF/地图由 RTAB-Map 或 ORB-SLAM3 发布）
  ros2 launch <本文件路径> use_amcl:=false params_file:=/path/to/nav2_params.yaml

默认地图 = fishbot 示例 room.yaml；真实建图地图由 orbslam3_nav2_bringup/scripts/save_map.py
从 /rtabmap/map 生成（见 docs/nav2_offline_sim.md）。
"""
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    map_yaml = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    use_rviz = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_fake_scan = LaunchConfiguration('use_fake_scan')
    use_amcl = LaunchConfiguration('use_amcl')

    fishbot_share = get_package_share_directory('fishbot_navigation2')
    default_map = os.path.join(fishbot_share, 'maps', 'room.yaml')
    default_params = os.path.join(fishbot_share, 'config', 'nav2_params.yaml')

    # 本 launch 所在目录（semphyspc/launch），向上定位假激光脚本
    this_dir = os.path.dirname(os.path.abspath(__file__))
    fake_scan_script = os.path.join(this_dir, '..', 'scripts', 'offline_scan_publisher.py')

    # 仅当 use_amcl 且 use_fake_scan 都为 true 时启动假激光
    use_amcl_and_fake_scan = PythonExpression(
        ['"', use_amcl, '" == "true" and "', use_fake_scan, '" == "true"'])

    # 1. 定位：map_server + AMCL（仅模式 A；模式 B 由 ORB-SLAM3/RTAB-Map 提供 TF+地图）
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('nav2_bringup'), 'launch', 'localization_launch.py'])),
        condition=IfCondition(use_amcl),
        launch_arguments={
            'map': map_yaml,
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'autostart': 'true',
        }.items(),
    )

    # 2. Nav2 导航栈（planner / controller / smoother / bt_navigator / ...）
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('nav2_bringup'), 'launch', 'navigation_launch.py'])),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'autostart': 'true',
        }.items(),
    )

    # 3. 假机器人 TF：odom → base_footprint（仅模式 A；模式 B 由 robot_state_publisher+odom 提供）
    static_tf = Node(
        condition=IfCondition(use_amcl),
        package='tf2_ros',
        executable='static_transform_publisher',
        name='offline_odom_to_base',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_footprint'],
    )

    # 4. 假激光：/scan 全 max_range（仅模式 A）
    fake_scan = ExecuteProcess(
        condition=IfCondition(use_amcl_and_fake_scan),
        cmd=['python3', fake_scan_script],
        name='offline_scan_publisher',
        output='screen',
    )

    # 5. RViz2（nav2 默认视图：Map / Costmap / Path / RobotModel）
    rviz = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('nav2_bringup'), 'rviz', 'nav2_default_view.rviz'])],
    )

    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=default_map,
                              description='2D 地图 yaml 路径（模式 A）'),
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='Nav2 参数 yaml 路径'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('use_fake_scan', default_value='true'),
        DeclareLaunchArgument('use_amcl', default_value='true',
                              description='true=离线 AMCL 定位；false=外部视觉定位（RTAB-Map/ORB-SLAM3）'),
        localization,
        navigation,
        static_tf,
        fake_scan,
        rviz,
    ])
