#!/bin/bash
# 4090 <-> 本机 双向同步训练 checkpoint（experiments/ 目录）。
#
# 用途：4090 是本阶段全量训练（S0/S1/S2 × 3 seed）的主机；本机 5070 Ti 可随时
# 把 4090 的进度拉到本地续训，或本机训完回传。已实测 torch2.6(4090) 的 ckpt 能
# 被 torch2.12(本机) 完整加载（模型 strict + 优化器 state 均通过）。
#
# 用法:
#   bash sync_ckpts.sh pull    # 4090 -> 本机（本机续训前先拉）
#   bash sync_ckpts.sh push    # 本机 -> 4090（本机训完回传）
#
# 续训命令（拉取后）:
#   python3 main_seg.py --config cfgs/seg_16k_ng6144_g050_merged.yaml \
#       --exp_name s0_seed0 --resume --num_workers 4
#   （--resume 会自动找 experiments/seg_16k_ng6144_g050_merged/cfgs/s0_seed0/*/ckpt-last.pth）
set -euo pipefail

REMOTE="intel4090@172.18.54.227"
PORT=2222
REMOTE_DIR="/home/intel4090/semphyspc/experiments"
LOCAL_DIR="/home/saw/ros2_ws/src/semphyspc/experiments"
# 不删对端多出的文件（默认 rsync 不删）；TFBoard 事件文件大且续训用不到，可按需取消注释
EXCLUDES=(--exclude '/TFBoard/')

case "${1:-pull}" in
  pull)
    echo "[pull] ${REMOTE}:${REMOTE_DIR} -> ${LOCAL_DIR}"
    rsync -av --partial --progress -e "ssh -p ${PORT}" \
      "${EXCLUDES[@]}" "${REMOTE}:${REMOTE_DIR}/" "${LOCAL_DIR}/"
    ;;
  push)
    echo "[push] ${LOCAL_DIR} -> ${REMOTE}:${REMOTE_DIR}"
    rsync -av --partial --progress -e "ssh -p ${PORT}" \
      "${EXCLUDES[@]}" "${LOCAL_DIR}/" "${REMOTE}:${REMOTE_DIR}/"
    ;;
  *)
    echo "usage: $0 {pull|push}" >&2
    exit 1
    ;;
esac
