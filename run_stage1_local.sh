#!/bin/bash
# 本机 RTX 5070 Ti：阶段一 S0/S1/S2 × 3 seed 全量队列（顺序 MAXP=1）。
# s0_seed0 / s0_seed1 自动从已拉取的 4090 ckpt（epoch 6）续训；其余 from-scratch。
# 崩溃自动 --resume 重试（最多 3 次）；进度与退出码写入 /tmp/stage1_local_master.log。
set -u
cd /home/saw/ros2_ws/src/semphyspc
MASTER=/tmp/stage1_local_master.log
echo "=== stage1 local start $(date) ===" >> "$MASTER"

declare -A CFG
CFG[s0]=cfgs/seg_16k_ng6144_g050_merged.yaml
CFG[s1]=cfgs/seg_16k_ng6144_g050_merged_s1_geom.yaml
CFG[s2]=cfgs/seg_16k_ng6144_g050_merged_s1_multitask.yaml

# 该 (stage,seed) 是否已有可续 ckpt（含 4090 拉取的 + 上次崩溃残留）
have_ckpt() {
  local stage=$1 seed=$2 cfg=${CFG[$stage]}
  local stem=$(basename "$cfg" .yaml) parent=$(basename "$(dirname "$cfg")")
  [ -n "$(ls experiments/$stem/$parent/${stage}_seed${seed}/*/ckpt-last.pth 2>/dev/null)" ]
}

run_one() {
  local stage=$1 seed=$2 cfg=${CFG[$stage]} name=${stage}_seed${seed}
  local resume_flag=""
  if have_ckpt "$stage" "$seed"; then resume_flag="--resume"; fi
  echo "launch ${name} resume=${resume_flag:-no} $(date)" >> "$MASTER"
  python3 main_seg.py --config "$cfg" --exp_name "$name" --seed "$seed" $resume_flag --num_workers 4 \
    > "/tmp/${name}.out" 2>&1
}

for stage in s0 s1 s2; do
  for seed in 0 1 2; do
    name=${stage}_seed${seed}
    for attempt in 1 2 3; do
      run_one "$stage" "$seed"; rc=$?
      if [ $rc -eq 0 ]; then break; fi
      echo "retry ${name} attempt=${attempt} exit=${rc} $(date)" >> "$MASTER"
    done
    echo "=== ${name} done $(date) ===" >> "$MASTER"
  done
done
echo "=== ALL_STAGE1_LOCAL_DONE $(date) ===" >> "$MASTER"
