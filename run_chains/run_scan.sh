#!/bin/bash
cd /home/saw/ros2_ws/src/semphyspc
for tag in ng256_g050 ng1024_g050 ng2048_g050 ng1024_g100 ng1024_g025; do
  echo "########## 开始 $tag  $(date '+%F %T') ##########"
  python3 main_seg.py --config cfgs/seg_16k_$tag.yaml --exp_name $tag --val_freq 1 --num_workers 8 \
    2>&1 | grep -E '\[Validation\]|Save checkpoint|Error|error|Traceback|out of memory'
  echo "########## 结束 $tag  $(date '+%F %T') ##########"
done
echo "ALL_SCAN_DONE"
