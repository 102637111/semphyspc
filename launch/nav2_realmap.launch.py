#!/usr/bin/env python3
"""真实地图（Semantic3D）语义导航验证：加载真实点云语义地图 → 语义软代价 → Nav2 路径规划。

链路（对应用户「加载真实地图 → 点云分析 → 路径规划」）：
  map_server（dom.yaml，Semantic3D domfountain_station3 子区域静态地图，50×50m@0.2m）
  静态 TF：map→odom→base_footprint 全恒等（免 AMCL，验证规划无需真机器人）
  semantic_costmap_publisher（--cost-npz dom_cost.npz，发布预计算语义软代价 /semantic_cost）
  navigation 栈（use_amcl:=false，params_file=nav2_params_realmap.yaml，resolution 0.2）

语义软代价分层：person=254(绕开)、车/墙=150~220(强避让)、植被=46(可穿越)、地面=0(自由)。
验证：发 /navigate_to_pose goal 后看 /plan，路径应绕开 person、可穿 vegetation。

用法：
  source /opt/ros/humble/setup.bash && source install/setup.bash
  ros2 launch <本文件>   # 默认 map=/tmp/s3_nav/dom.yaml
  # 另开终端发 goal：ros2 run semphyspc send_nav_goal.py --x -18 --y -15
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    map_yaml = LaunchConfiguration('map')
    cost_npz = LaunchConfiguration('cost_npz')
    params_file = LaunchConfiguration('params_file')
    use_rviz = LaunchConfiguration('use_rviz')
    use_semantic = LaunchConfiguration('use_semantic')
    start_x = LaunchConfiguration('start_x')
    start_y = LaunchConfiguration('start_y')

    # 脚本/参数一律从源码目录取：install 里只装 launch/params 的 data_files，而 scripts 里的
    # 语义发布器依赖 src 的 utils/ 结构（内部 sys.path 上溯 import），必须回 src 跑。
    _SRC = os.path.expanduser('~/ros2_ws/src/semphyspc')
    default_params = os.path.join(_SRC, 'params', 'nav2_params_realmap.yaml')
    default_map = '/tmp/s3_nav/dom.yaml'
    default_cost = '/tmp/s3_nav/dom_cost.npz'
    pub_script = os.path.join(_SRC, 'scripts', 'semantic_costmap_publisher.py')

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

    # 2. 静态 TF：map→odom（恒等）、odom→base_footprint（机器人起点 = start_x/start_y，免 AMCL）
    tf_map_odom = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='tf_map_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'])
    tf_odom_base = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='tf_odom_base',
        arguments=[start_x, start_y, '0', '0', '0', '0', 'odom', 'base_footprint'])

    # 3. 发布预计算语义软代价（cost.npz → /semantic_cost，frame=map）
    #    仅 soft_cost 模式需要；geometric/semantic_ban 是纯二元障碍 baseline，不挂语义层。
    semantic_pub = ExecuteProcess(
        condition=IfCondition(use_semantic),
        cmd=['python3', pub_script, '--cost-npz', cost_npz, '--frame-id', 'map',
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

    # 5. RViz2（可选，看 global_costmap 分层 + /plan 路径）
    rviz = Node(
        condition=IfCondition(use_rviz),
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        arguments=['-d', os.path.join(_SRC, 'rviz', 'semantic_costmap_shot.rviz')])

    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=default_map),
        DeclareLaunchArgument('cost_npz', default_value=default_cost),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        DeclareLaunchArgument('use_semantic', default_value='true',
                              description='soft_cost=true 挂语义代价层；geometric/semantic_ban=false 纯二元障碍'),
        DeclareLaunchArgument('start_x', default_value='-13.9'),
        DeclareLaunchArgument('start_y', default_value='3.7'),
        map_server,
        map_lifecycle,
        tf_map_odom,
        tf_odom_base,
        semantic_pub,
        navigation,
        rviz,
    ])
