#!/bin/bash
# ============================================================
# Color-PointMamba 远程数据集采集工具
#
# 适用于：Orbbec 相机在小车上，本地通过 WiFi 控制的场景
# 原理：通过 ROS2 Service 在机器人端保存点云，再拉取到本地
# ============================================================

set -e

# 配置
ROBOT_IP="${ROBOT_IP:-192.168.1.100}"      # 小车 IP
ROBOT_USER="${ROBOT_USER:-saw}"             # 小车用户名
ROBOT_PCD_DIR="${ROBOT_PCD_DIR:-~/point_cloud}"  # 机器人上保存路径
LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-./my_raw_dataset}"
CAMERA_NAME="${CAMERA_NAME:-camera}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Color-PointMamba 远程数据集采集${NC}"
echo -e "${GREEN}============================================${NC}"
echo "  机器人: ${ROBOT_USER}@${ROBOT_IP}"
echo "  远程保存: ${ROBOT_PCD_DIR}"
echo "  本地数据集: ${LOCAL_DATASET_DIR}"
echo ""

# 功能
case "${1:-help}" in
    capture)
        # 单帧捕获：调用 save_point_cloud 服务
        CLASS_NAME="${2:-unknown}"
        FRAMES="${3:-10}"

        echo -e "${YELLOW}采集类别 '${CLASS_NAME}'，共 ${FRAMES} 帧${NC}"
        echo "按 ENTER 开始，采集期间请缓慢旋转物体..."
        read

        for i in $(seq 1 ${FRAMES}); do
            echo -n "  帧 ${i}/${FRAMES}... "
            # 远程调用 save_point_cloud 服务
            ssh ${ROBOT_USER}@${ROBOT_IP} \
                "source /opt/ros/humble/setup.bash && \
                 ros2 service call /${CAMERA_NAME}/save_point_cloud std_srvs/srv/Empty '{}'" \
                2>/dev/null
            sleep 1.5
            echo "OK"
        done

        # 拉取文件
        echo -e "${YELLOW}拉取文件到本地...${NC}"
        mkdir -p "${LOCAL_DATASET_DIR}/${CLASS_NAME}"
        ssh ${ROBOT_USER}@${ROBOT_IP} \
            "ls -t ${ROBOT_PCD_DIR}/colored_points_*.ply 2>/dev/null | head -${FRAMES}" | \
        while read remote_file; do
            scp -q "${ROBOT_USER}@${ROBOT_IP}:${remote_file}" "${LOCAL_DATASET_DIR}/${CLASS_NAME}/"
            # 删除远程文件
            ssh ${ROBOT_USER}@${ROBOT_IP} "rm -f ${remote_file}" 2>/dev/null
        done

        count=$(ls ${LOCAL_DATASET_DIR}/${CLASS_NAME}/*.ply 2>/dev/null | wc -l)
        echo -e "${GREEN}✓ '${CLASS_NAME}' 完成: ${count} 帧 → ${LOCAL_DATASET_DIR}/${CLASS_NAME}/${NC}"
        ;;

    pull)
        # 拉取所有已保存的点云
        CLASS_NAME="${2:-unknown}"
        mkdir -p "${LOCAL_DATASET_DIR}/${CLASS_NAME}"
        echo -e "${YELLOW}拉取机器人上所有 .ply 文件...${NC}"
        scp -q "${ROBOT_USER}@${ROBOT_IP}:${ROBOT_PCD_DIR}/colored_points_*.ply" \
            "${LOCAL_DATASET_DIR}/${CLASS_NAME}/" 2>/dev/null && \
        ssh ${ROBOT_USER}@${ROBOT_IP} "rm -f ${ROBOT_PCD_DIR}/colored_points_*.ply" 2>/dev/null
        count=$(ls ${LOCAL_DATASET_DIR}/${CLASS_NAME}/*.ply 2>/dev/null | wc -l)
        echo -e "${GREEN}✓ 拉取完成: ${count} 文件${NC}"
        ;;

    build)
        # 构建训练数据集
        echo -e "${YELLOW}构建训练数据集...${NC}"
        python3 scripts/build_rtabmap_dataset.py \
            --raw_dir "${LOCAL_DATASET_DIR}" \
            --output_dir ./data/my_real_dataset \
            --npoints 1024
        echo -e "${GREEN}✓ 数据集构建完成${NC}"
        echo "  train.npz + test.npz → ./data/my_real_dataset/"
        ;;

    *)
        echo "用法:"
        echo "  $0 capture <class_name> [frames=10]   — 采集一个类别的多帧"
        echo "  $0 pull <class_name>                  — 拉取远程已保存的点云"
        echo "  $0 build                              — 构建训练数据集"
        echo ""
        echo "环境变量:"
        echo "  ROBOT_IP        机器人IP (默认 192.168.1.100)"
        echo "  ROBOT_USER      SSH用户 (默认 saw)"
        echo "  ROBOT_PCD_DIR   机器人上点云保存路径"
        echo "  LOCAL_DATASET_DIR  本地数据集目录"
        echo ""
        echo "示例:"
        echo "  ROBOT_IP=10.0.0.5 $0 capture cup 10"
        echo "  ROBOT_IP=10.0.0.5 $0 build"
        ;;
esac
