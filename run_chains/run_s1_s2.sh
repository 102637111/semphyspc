set -x
CFG=cfgs/seg_nuscenes_full_16k_g256.yaml
CKPT=experiments/seg_nuscenes_full_16k_g256/cfgs/g256_v1/20260907-200401/ckpt-best.pth
PROJ=data/nuscenes_projection_index.npz
TH=0.5,0.6,0.7,0.8,0.9,0.95,1.0

run () {  # $1=tag  $2=bbox  $3=outjson  $4=extra
  echo "########## $1 ##########"
  python3 scripts/eval_gated_voting.py --config $CFG --ckpts $CKPT --proj $PROJ \
      --bbox "$2" --thetas $TH --depth_margin 0.5 $4 \
      --tag "$1" --out-json "$3"
}

run s1_oracle_gt     data/nuscenes_2d_bbox_cache.npz                results/gated_voting_s1_oracle.json ""
run s2_rtdetr_lidar  data/nuscenes_2d_bbox_cache_rtdetr_val_lidar.npz results/gated_voting_s2_lidar.json --group-refine
run s2_rtdetr_size   data/nuscenes_2d_bbox_cache_rtdetr_val_size.npz  results/gated_voting_s2_size.json  --group-refine
run s2_rtdetr_none   data/nuscenes_2d_bbox_cache_rtdetr_val_none.npz  results/gated_voting_s2_none.json  --group-refine
echo "ALL DONE"
