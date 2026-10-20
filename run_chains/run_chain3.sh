#!/bin/bash
# chain3：接在 chain2 后面。做一件决定性的事——把罕见类的「GT → 预测」分布导出来。
#
# 为什么要做：child/stroller 的 IoU 精确为 0，但有两种截然不同的原因，靠 IoU 分不出来：
#   (a) 模型能定位它们、只是没学会 child/adult 的高度阈值 → 点被预测成 ped.adult
#       ⇒ HAG（离地高度）显式输入通道有用，实例级后处理也走得通
#   (b) 点被预测成 noise / 背景 → 实例簇根本不形成
#       ⇒ 上面两条路都断了，只能靠 RGB 外观
# 看 ng1024_g050（收敛的 P1 基线）和 p3_ng1024_g050（类别平衡+罕见帧过采样）两组的分布，
# 还能顺带回答「P3 到底有没有把 child 从 adult 里拽出来」。
set -u
cd /home/saw/ros2_ws/src/semphyspc

# 注意：本脚本吃 GPU，必须在 P3 训练结束后再跑（Xid 8 看门狗）。
# 依赖的 p3 ckpt-best 由 resume_p3.sh 产出；若尚未产出，run_one 会打印警告并跳过，
# 由 run_post_p3.sh 编排保证时序。

run_one() {
  tag="$1"; cfg="$2"; expdir="$3"
  ck=$(ls -t experiments/seg_16k_${expdir}/cfgs/*/*/ckpt-best.pth 2>/dev/null | head -1)
  if [ -z "$ck" ]; then echo "!!! $tag 找不到 ckpt-best.pth（experiments/seg_16k_${expdir}），跳过混淆统计"; return; fi
  echo "-- 混淆统计 $tag  用 $ck  $(date '+%F %T')"
  python3 scripts/eval_seg_full.py --config "$cfg" --ckpts "$ck" \
    --exp_tag "${tag}_confusion" \
    --dump_confusion --confusion_topk 6 \
    --confusion_out results/confusion_${tag}.json
}

echo "########## 罕见类混淆统计  $(date '+%F %T') ##########"
run_one ng1024_g050     cfgs/seg_16k_ng1024_g050.yaml     ng1024_g050
run_one p3_ng1024_g050  cfgs/seg_16k_ng1024_g050_p3.yaml  ng1024_g050_p3

echo "ALL_CHAIN3_DONE  $(date '+%F %T')"
