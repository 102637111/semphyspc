#!/bin/bash
# P3：类别平衡损失 + 罕见帧过采样（架构与 seg_16k_ng1024_g050 完全一致）。
# 队列位置：等 S3 流水线跑完再上 GPU。
set -u
cd /home/saw/ros2_ws/src/semphyspc

CFG=cfgs/seg_16k_ng1024_g050_p3.yaml
EXP=experiments/seg_16k_ng1024_g050_p3

echo "########## 等待 S3 流水线结束  $(date '+%F %T') ##########"
for i in $(seq 1 2880); do
  grep -q ALL_S3_DONE /tmp/s3_run.log 2>/dev/null && break
  sleep 60
done
if ! grep -q ALL_S3_DONE /tmp/s3_run.log 2>/dev/null; then
  echo "!!! 等满 48 h 仍未见到 ALL_S3_DONE，直接开始 P3"
fi
echo "########## 开始 P3 训练  $(date '+%F %T') ##########"

python3 main_seg.py --config "$CFG" --exp_name p3_ng1024_g050 --val_freq 1 --num_workers 8 \
  2>&1 | grep -E "RareAwareSampler|稀有类|过采样|Training\]|Validation\]|Error|Traceback|out of memory"
echo "########## P3 训练结束  $(date '+%F %T') ##########"

CK=$(ls -t $EXP/cfgs/*/*/ckpt-best.pth 2>/dev/null | head -1)
if [ -z "$CK" ]; then echo "!!! 找不到 ckpt-best.pth"; exit 1; fi
echo "使用权重 $CK"

python3 scripts/eval_seg_full.py --config "$CFG" --ckpts "$CK" \
  --exp_tag p3_ng1024_g050 --out_json results/seg_16k_g1024_g050_p3.json
echo "ALL_P3_DONE"
