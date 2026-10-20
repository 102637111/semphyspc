"""P3：罕见类感知的帧采样器。

动机（§4.14）：nuScenes 的 child / police_officer / stroller / personal_mobility
在**所有**配置下逐类 IoU ≡ 0，而其「点加权可达性」并不低——瓶颈是标注量/场景出现次数，
不是采样密度也不是分组粒度。若这个归因成立，那么把监督信号集中到这几百个实例上
应当能把它们从 0 抬起来；若仍然不动，则归因得到进一步确认。

做法：按帧给权重 w_frame = max_{c ∈ 该帧出现的稀有类} (N_max / N_c)^0.5，
裁剪到 [1, max_rep]，其中 N_c 是类 c 在训练集里的**实例数**。然后按该权重有放回采样，
使每轮的样本构成向含稀有类的帧倾斜。数据集自带 yaw 随机旋转与随机降采样，
所以同一帧被重复抽到也不是完全相同的输入。

用法：见 tools/builder.py，仅当 `dataset.rare_weights` 存在且为训练集时启用。
"""
import numpy as np
import torch
from torch.utils.data import Sampler


class RareAwareSampler(Sampler):
    """按帧权重有放回采样；每个 epoch 换一次随机种子。"""

    def __init__(self, weights, num_samples=None, seed=0):
        w = np.asarray(weights, dtype=np.float64).copy()
        if not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError('rare_weights 非法（含 nan/inf 或全 0）')
        self.weights = w / w.sum()
        self.n = int(num_samples) if num_samples else len(w)
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, e):
        self._epoch = int(e)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1          # runner 不调 set_epoch，靠自增保证每轮不同
        idx = rng.choice(len(self.weights), size=self.n, replace=True, p=self.weights)
        return iter(idx.tolist())

    def __len__(self):
        return self.n
