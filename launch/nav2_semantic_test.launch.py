#!/usr/bin/env python3
"""阶段 1 验证：语义软代价层端到端（无 AMCL、静态 TF、合成语义点云、全自动激活）。

组件：
  map_server（room.yaml，lifecycle 自动激活）—— 提供 /map 与 map 帧
  静态 TF：map→odom（恒等）、odom→base_footprint（恒等）—— 免 AMCL 的 initialpose 交互
  semantic_costmap_publisher（合成 PLY → /semantic_cost，frame=map）
  navigation 栈（use_amcl:=false，params_file=nav2_params_semantic.yaml）

验证（无 RViz2 交互即可自动化）：
  ros2 topic echo /global_costmap/costmap 看分层：person=254(100)、car=220(87)、
  vegetation=46(18)、ground=0。规划器路径应绕开 person、可穿 vegetation。

用法：
  source /opt/ros/humble/setup.bash
  source install/setup.bash
  ros2 launch <本文件> map:=/home/saw/ros2_ws/src/fishbot_navigation2/maps/room.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    map_yaml = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    use_rviz = LaunchConfiguration('use_rviz')

    this_dir = os.path.dirname(os.path.abspath(__file__))
    default_params = os.path.join(this_dir, '..', 'params', 'nav2_params_semantic.yaml')
    fishbot_share = get_package_share_directory('fishbot_navigation2')
    default_map = os.path.join(fishbot_share, 'maps', 'room.yaml')

    # 合成语义点云生成 + 发布（先生成 /tmp/synthetic_semantic.ply 再发布）
    gen_ply_script = os.path.join(this_dir, '..', 'scripts', 'gen_synthetic_semantic_ply.py')
    pub_script = os.path.join(this_dir, '..', 'scripts', 'semantic_costmap_publisher.py')
    synth_ply = '/tmp/synthetic_semantic.ply'

    # 1. map_server（lifecycle）+ 独立 lifecycle_manager 激活（不含 amcl）
    map_server = Node(
        package='nav2_map_server', executable='map_server', name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_yaml, 'use_sim_time': False}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')])
    map_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_map', output='screen',
        parameters=[{'autostart': True, 'node_names': ['map_server']}])

    # 2. 静态 TF：map→odom→base_footprint 全恒等（免 AMCL）
    tf_map_odom = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='tf_map_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'])
    tf_odom_base = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='tf_odom_base',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_footprint'])

    # 3. 生成合成语义点云（一次性），再发布 /semantic_cost
    gen_ply = ExecuteProcess(
        cmd=['python3', gen_ply_script, '-o', synth_ply], name='gen_synthetic_ply',
        output='screen')
    semantic_pub = ExecuteProcess(
        cmd=['python3', pub_script, '--ply', synth_ply, '--frame-id', 'map',
             '--publish-rate', '2.0'],
        name='semantic_costmap_publisher', output='screen')

    # 4. Nav2 导航栈（planner/controller/costmap/bt_navigator，use_amcl:=false）
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('nav2_bringup'), 'launch', 'navigation_launch.py'])),
        launch_arguments={
            'use_sim_time': 'false',
            'params_file': params_file,
            'autostart': 'true',
        }.items(),
    )

    # 5. RViz2（可选，看 global_costmap 分层）
    rviz = Node(
        condition=IfCondition(use_rviz),
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('nav2_bringup'), 'rviz', 'nav2_default_view.rviz'])])

    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=default_map),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        map_server,
        map_lifecycle,
        tf_map_odom,
        tf_odom_base,
        gen_ply,
        semantic_pub,
        navigation,
        rviz,
    ])
