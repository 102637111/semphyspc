#!/usr/bin/env python3
"""生成 P3（类别平衡损失 + 罕见帧过采样）的配置。

做法（§4.14 的后续干预实验）：架构与 `cfgs/seg_16k_ng1024_g050.yaml` **完全一致**，
只改两处监督信号相关的设置，从而把「分组粒度」这一主因固定住，单独看
「把监督搬到稀有类上」能不能把 5 个不可约类从 0 抬起来。

  1. `sem_class_weight`：逆平方根频率加权 w_c = clip((f_ref/f_c)^0.5, floor, cap)，
     f_c 为该类在 **训练集实际点数** 中的占比，f_ref 取所有非零类占比的中位数。
     floor 保证 manmade/vegetation 这类占 90% 点数的 stuff 类不被过度压制。
  2. `RARE_AWARE_SAMPLING`：按类实例数给每帧加权过采样（见 datasets/rare_sampler.py）。

用法：
    python3 scripts/make_p3_config.py \
        --counts results/train_class_point_counts_16k.npy \
        --base cfgs/seg_16k_ng1024_g050.yaml \
        --out cfgs/seg_16k_ng1024_g050_p3.yaml
"""
import argparse
import numpy as np

NAMES = ['noise', 'animal', 'ped.adult', 'ped.child', 'ped.cw', 'ped.pers_mob', 'ped.police',
         'ped.stroller', 'ped.wheelchair', 'barrier', 'debris', 'pushable', 'trafficcone',
         'bicycle_rack', 'bicycle', 'bus.bendy', 'bus.rigid', 'car', 'construction',
         'ambulance', 'emergency.police', 'motorcycle', 'trailer', 'truck', 'driveable',
         'flat.other', 'sidewalk', 'terrain', 'manmade', 'static.other', 'vegetation', 'ego']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--counts', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--exponent', type=float, default=0.5)
    ap.add_argument('--floor', type=float, default=0.25)
    ap.add_argument('--cap', type=float, default=10.0)
    ap.add_argument('--rare_max_rep', type=float, default=25.0)
    ap.add_argument('--rare_exponent', type=float, default=0.5)
    args = ap.parse_args()

    cnt = np.load(args.counts).astype(np.float64)
    total = cnt.sum()
    freq = cnt / total
    nz = np.where(cnt > 0)[0]
    f_ref = float(np.median(freq[nz]))
    w = np.ones(32, dtype=np.float64)
    for c in nz:
        w[c] = float(np.clip((f_ref / freq[c]) ** args.exponent, args.floor, args.cap))
    # 全 0 的类（本数据集中有 ambulance）不参与损失，权重取 1 即可
    print(f'f_ref = {f_ref:.6f}（{len(nz)} 个非零类占比的中位数）')
    print(f"{'id':>3} {'类别':<16}{'训练点数':>13}{'占比':>10}{'损失权重':>10}")
    for c in range(32):
        print(f'{c:>3} {NAMES[c]:<16}{int(cnt[c]):>13,}{freq[c]*100:>9.4f}%{w[c]:>10.3f}')

    wlist = ', '.join(f'{v:.4g}' for v in w)
    with open(args.base) as f:
        base = f.read()

    # 1) model 块里插入 sem_class_weight（build_loss_func 从 self.config 读）
    key = "  rms_norm: False, drop_path: 0.1, drop_out: 0.0,\n"
    assert key in base, '基准配置的 model 块结构变了，请更新锚点'
    base = base.replace(key, key + f"  sem_class_weight: [{wlist}],\n")

    # 2) train 的 others 里打开罕见帧过采样
    tk = "train: { _base_: cfgs/dataset_configs/NuscenesSegFull16k.yaml, others: { subset: 'train' } },"
    assert tk in base, '基准配置的 train 数据集行结构变了，请更新锚点'
    base = base.replace(tk,
        "train: { _base_: cfgs/dataset_configs/NuscenesSegFull16k.yaml, others: { subset: 'train', "
        f"RARE_AWARE_SAMPLING: True, RARE_MAX_REP: {args.rare_max_rep:g}, "
        f"RARE_EXPONENT: {args.rare_exponent:g} }} }},")

    base = base.replace('# 方案B：',
                        '# P3：方案B 的架构（16k 点 / num_group=1024 / GridGroup 0.5m）+ 类别平衡损失 + 罕见帧过采样。\n'
                        '# 生成脚本 scripts/make_p3_config.py。与 seg_16k_ng1024_g050.yaml 只差监督信号那两处。\n'
                        '# 原注释：', 1)

    with open(args.out, 'w') as f:
        f.write(base)
    print(f'\n已写出：{args.out}')


if __name__ == '__main__':
    main()
