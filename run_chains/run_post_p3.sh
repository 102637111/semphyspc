#!/bin/bash
# P3 结束后按序跑完剩余 GPU 实验：融合规则 → 罕见类混淆 → 安全通道。
#
# 为什么要编排：这三步都吃 GPU，必须严格串行，且不能与 P3 训练并行（本机显示器
# Xorg/gnome-shell 与训练共用 5070 Ti，并行会触发 Xid 8 看门狗超时）。
#
# 等待条件：resume_p3.sh 及其训练进程（main_seg.py）与末尾评测（eval_seg_full.py）
# 全部退出后才开始，避免在 P3 的末尾评测还在占 GPU 时抢占。若 P3 在 9 h 内没能跑完
# （如三振出局），按部分完成继续——融合规则与安全通道只依赖 g256 基线/S3 微调权重，
# 不依赖 P3；只有 chain3 的 p3 混淆会因缺少 ckpt-best 而跳过。
#
# 日志落仓库内 run_chains/post_p3_*.log，重启不丢。
set -u
cd /home/saw/ros2_ws/src/semphyspc

LOG="run_chains/post_p3_$(date '+%Y%m%d_%H%M%S').log"
exec >> "$LOG" 2>&1

echo "########## post-P3 编排启动  $(date '+%F %T') ##########"

deadline=$(( $(date +%s) + 9*3600 ))   # 最多等 9 h（P3 40 轮约 5 h，留裕量）
while true; do
  if ! pgrep -f 'main_seg[.]py' >/dev/null 2>&1 \
     && ! pgrep -f 'eval_seg_full[.]py' >/dev/null 2>&1 \
     && ! pgrep -f 'resume_p3[.]sh' >/dev/null 2>&1; then
    echo "P3 链已完全退出，开始后续实验  $(date '+%F %T')"
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "!!! 等待 P3 超时（9 h），按部分完成继续（融合规则/安全通道不依赖 P3）  $(date '+%F %T')"
    break
  fi
  sleep 60
done

echo "########## 步骤 1/3：融合规则改进（size thr0.7 + score）  $(date '+%F %T') ##########"
bash run_chains/run_fusion_rule.sh
echo "步骤 1 结束（退出码 $?）  $(date '+%F %T')"

echo "########## 步骤 2/3：罕见类混淆分布  $(date '+%F %T') ##########"
bash run_chains/run_chain3.sh
echo "步骤 2 结束（退出码 $?）  $(date '+%F %T')"

echo "########## 步骤 3/3：安全通道（车辆 + 低阈值 + 延迟）  $(date '+%F %T') ##########"
bash run_chains/run_chain4.sh
echo "步骤 3 结束（退出码 $?）  $(date '+%F %T')"

echo "POST_P3_ALL_DONE  $(date '+%F %T')"
