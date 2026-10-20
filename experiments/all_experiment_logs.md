# 实验日志合并

> 合并日期：2026-10-09
> 由 7 个实验日志/结果文件合并而成（原文件已移至回收站，可恢复）。
> 覆盖：级联语义分割 → 分割对照 → 导航软代价证据链（B①/B③/方案 a/B④/B②）。

## 目录

| # | 内容 | 原文件 |
|---|---|---|
| 1 | 级联语义分割实验日志 | `experiments/cascade_experiment_log.md` |
| 2 | 语义分割实验对照表 | `results/comparison.md` |
| 3 | 导航实验日志（总记录） | `experiments/nav_ablation_experiment_log.md` |
| 4 | B① 导航三组对照 v1 | `results/nav_compare/SUMMARY.md` |
| 5 | B③ hard 代价权衡曲线 | `results/hard_sweep/SUMMARY.md` |
| 6 | B① 导航三组对照 v2（方案 a 后） | `results/nav_compare_v2/SUMMARY.md` |
| 7 | B④ 材料敏感性 | `results/material_sensitivity/SUMMARY.md` |
| 8 | B② 感知误差传递（导航代价混淆） | `results/nav_error_propagation_merged_pretrain_ft.json` |
| 9 | B① 植被密集场景补强（bildstein） | `results/nav_compare_bild/` |
| 10 | C⑥ 检测辅助消融梯度 | `results/gated_voting_ablation_merged_pretrain_ft.json` |
| 11 | D⑧ 时延+鲁棒性 | `results/robustness_merged_pretrain_ft.json` + `results/latency_merged_pretrain_ft.json` |
| 12 | C⑤ 公平基线（PointNet++ / PointTransformer） | `results/seg_16k_pointnet2.json` + `results/seg_16k_pointtransformer.json` |
| 13 | 阶段 A 定位重写（感知+导航） | `docs/语义地图.docx` |
| 14 | 方案 A 射线序列 Mamba 可行性（已完成最小验证） | `scripts/eval_ray_sequence.py` + `experiments/ray_sequence/ray_sequence_result.json` |

---



<!-- ==================== 1. 级联语义分割实验日志 ==================== -->

# PointMamba RGB+XYZ 级联语义分割 —— 完整实验记录

> 记录日期：2026-08-23
> 目标：RGB+XYZ 早期融合（`in_channels=6`）注入 PointMamba，构建「地面几何 → 实例分割 → 物体判类」三级级联，实现 3D 点云语义分割 + 物理材质映射（碰撞安全/可破坏性），服务下游导航/游戏地图应用，面向 Q3+ 期刊发表。

---

## 1. 级联范式（三环）

```
① 地面几何分割      二分类（地面/非地面）        前置过滤，覆盖 42.4% 场景点
② 实例分割          offset 回归 vote→质心 + DBSCAN  把「thing 类」聚成单物体实例
③ 物体级分类        PointMambaScan 11 类 crop 分类   对每个实例判类（细粒度 thing 类）
```

- **② 的输入**：thing 类点 = `sem ∈ 11 个 B2 类 且 inst > 0`；预测坐标 `xyz + offset` 后 DBSCAN 聚类。
- **③ 的输入**：每个聚类做「中心 + 单位球 + 1024 点」crop，送入 B2 分类器回填类别。
- **11 个 thing 类**（B2 覆盖）：`CLASS2SEM=[17,2,9,23,12,22,11,16,21,18,14]`
  = car, ped.adult, barrier, truck, trafficcone, trailer, pushable, bus.rigid, motorcycle, construction, bicycle。

---

## 2. 各环节训练结果

### C2 — offset 回归实例分割（`train_instance_offset.py`）

| 项 | 值 |
|---|---|
| 可训练参数 | 132.6K（仅 instance_head，迁移预训练 202/202 层） |
| 训练 | 15 epochs × 5000 batch，SmoothL1 回归 |
| 最终（`ckpt-offset.pth`） | **recall 0.5165 / precision 0.6679 / PQ 0.5093** |

按实例大小分档（val 300 场景）：

| 大小档 | recall | GT 实例数 |
|---|---|---|
| 0–50 点 | 0.4594 | 6728 |
| 50–200 点 | 0.8069 | 875 |
| 200–1000 点 | 0.8292 | 363 |
| >1000 点 | 1.0000 | 34 |

> 稀疏瓶颈：84% thing 实例 <50 点，小物体 recall 仅 ~0.46，是整条级联的天花板。

### B2 — 物体级分类器（PointMambaScan，11 类 crop）

| 项 | 值 |
|---|---|
| 数据 | 28545 物体，train 22839 / val 5706 |
| 训练 | 80 epochs，迁移预训练 186 层（跳过 2 层形状不匹配） |
| 最终（`ckpt-cls.pth`） | **macro 0.8741 / overall 0.8791** |

逐类（val）：

| 类 | acc | 类 | acc |
|---|---|---|---|
| car | 0.8425 | bus.rigid | 0.8575 |
| ped.adult | 0.9750 | motorcycle | 0.8629 |
| barrier | 0.9700 | construction | 0.7947 |
| truck | 0.7225 | bicycle | 0.7963 |
| trafficcone | 0.9801 | trailer | 0.8762 |
| pushable | 0.9375 | | |

### C3 — 解冻微调（`train_instance_finetune.py`）

联合损失 `L = 0.1·CE(sem) + SmoothL1(offset)`，解冻全部 12.47M 参数：

| 指标 | 冻结前（`ckpt-offset`） | 微调后（`ckpt-offset-finetuned`） |
|---|---|---|
| PQ @ eps=0.5 | 0.5093 | **0.5155**（+0.6，@ epoch 20） |
| sem 损失 | — | 0.4289 → 0.3336 |
| off 损失 | — | 0.2246 → 0.1274 |

> 结论：解冻微调收益微小（PQ +0.6、级联 +0.3）。真正顺带收益是联合损失把 dense 语义基线从 0.485 提到 0.5021。

---

## 3. 消融与敏感性

### C4-a — eps 扫描（offset 微调后，300 val）

| eps | PQ | eps | PQ |
|---|---|---|---|
| 0.20 | 0.1832 | 0.80 | 0.5924 |
| 0.30 | 0.3437 | 0.90 | 0.6004 |
| 0.40 | 0.4463 | **1.00** | **0.6056** |
| 0.50 | 0.5104 | 1.20 | 0.6050 |
| 0.60 | 0.5496 | 1.50 | 0.5918 |
| 0.70 | 0.5783 | | |

**PQ 峰在 eps≈1.0（0.6056）**，随 eps 增大而升（offset 平移点分散 ~1m 量级）。

### C4-b — 几何基线对照（原始坐标 DBSCAN，无 offset）

| eps | 几何 PQ | offset PQ | offset 增量 |
|---|---|---|---|
| 0.50 | 0.3812 | 0.5104 | **+13.0 点** |
| 0.70 | 0.5024 | 0.5783 | +7.6 点 |
| 1.00 | 0.5870 | 0.6056 | +1.9 点 |

> **offset 回归的真实价值在「小 eps 下把点聚紧」**：大 eps 时几何聚类已到 0.59，offset 只 +1.9；小 eps 时 offset 从 0.38 拉到 0.51（+13 点）。

### D — 端到端级联：eps 敏感性（500 val，微调后）

dense 基线恒定 0.5021（全 32 类 mIoU = 0.2342）：

| eps | 级联 thing 精度 | vs dense |
|---|---|---|
| **0.5** | **0.6901** | +18.8 点 |
| 0.6 | 0.6630 | +16.1 |
| 0.7 | 0.6733 | +17.1 |
| 1.0 | 0.6465 | +14.4 |

> **PQ 与级联精度方向相反**：PQ 峰在 eps=1.0，级联峰在 eps=0.5。因为大 eps 过合并 → B2 给多物体 blob 判单一标签；级联下游分类器要「干净单物体实例」，偏好欠分割。

### E — barrier 失败根因（`analyze_barrier_failure.py`）

barrier GT 39204 点：

| 项 | 占比 |
|---|---|
| DBSCAN 噪声（未聚类→自动错） | 2.1% |
| 被聚类 | 97.9% |
| 聚类内 B2 判对 barrier | 46.1% |
| 过合并（簇内非 barrier 点平均） | 1.0% |

B2 混淆分布（barrier 点被判成）：

| 判成 | 点数 | 占比 |
|---|---|---|
| **trailer** | 18872 | **49.2%** |
| barrier（对） | 17684 | 46.1% |
| bus.rigid | 1168 | 3.0% |
| truck | 300 | 0.8% |
| car | 126 | 0.3% |

> **根因不是实例分割（噪声仅 2.1%、过合并仅 1%），而是 B2 把 barrier 误判成 trailer**——二者都是长条水平结构，单位球归一化后难分。

### F — 类别门控级联（`eval_cascade_gated.py`，gate=[9]=barrier）

规则：每实例取 dense 多数投票，若多数为 gate 类则保留 dense（不 B2），否则按原级联 B2。**只用模型预测，不用真值**。

| 类 | dense | 无门控级联 | 门控级联 | 提升 |
|---|---|---|---|---|
| car | 0.7155 | 0.7866 | 0.7560 | +0.04 |
| ped.adult | 0.1385 | 0.5246 | 0.5152 | +0.38 |
| **barrier** | 0.7309 | **0.4049** | **0.8281** | **+0.10** |
| truck | 0.1946 | 0.6632 | 0.6417 | +0.45 |
| trafficcone | 0.1849 | 0.5613 | 0.4527 | +0.27 |
| trailer | 0.3135 | 0.8156 | 0.8205 | +0.51 |
| pushable | 0.2419 | 0.3523 | 0.3062 | +0.06 |
| bus.rigid | 0.2045 | 0.7170 | 0.7122 | +0.51 |
| motorcycle | 0.0000 | 0.7628 | 0.7129 | +0.71 |
| construction | 0.1088 | 0.6400 | 0.6249 | +0.52 |
| bicycle | 0.0000 | 0.2456 | 0.2734 | +0.27 |
| **整体(11类)** | 0.5021 | **0.6901** | **0.7244** | **+0.22** |

门控实例数 764。整体 0.6901 → **0.7244（+3.4）**，11 类无倒退。

### G — 门控敏感性分析（`eval_gate_sensitivity.py`）

方法：一次前向缓存每个实例的 (dense 多数投票, B2 预测)，纯后处理评估所有单类门控集合，系统回答「门控是否该推广到其他类」。

基线复现：dense 0.5021、全 B2 0.6901（与 D/F 完全一致，验证缓存正确）。

| 单类门控 gate | 整体精度 | vs 全B2 | 说明 |
|---|---|---|---|
| 无门控（全 B2） | 0.6901 | — | 基线 |
| **9 barrier** | **0.7309** | **+4.1** | **唯一门控有益的类** |
| 2 ped.adult | 0.6883 | −0.2 | 有害 |
| 18 construction | 0.6887 | −0.1 | 有害 |
| 16 bus.rigid | 0.6854 | −0.5 | 有害 |
| 22 trailer | 0.6870 | −0.3 | 有害 |
| 11 pushable | 0.6893 | −0.1 | 中性偏负 |
| 12 trafficcone | 0.6896 | −0.1 | 中性偏负 |
| 23 truck | 0.6757 | −1.4 | 有害 |
| 17 car | 0.6425 | **−4.8** | **严重有害** |
| 21 motorcycle | 0.6901 | 0.0 | 中性 |
| 14 bicycle | 0.6901 | 0.0 | 中性 |

**最优门控集合 = {barrier} → 0.7309**（与 F 的 0.7244 差 ~0.7 点，为跨 run 的 crop 采样随机方差，排名一致）。

对照：**barrier+construction 联合门控 = 0.7295 < barrier 单独 0.7309** → 加 construction 反而更差。

> **结论：门控不应推广。** barrier 是唯一 dense(0.73) > B2(0.40) 的类，所以门控保 dense 有益；其余 10 类 B2 都优于 dense（construction dense 仅 0.11 vs B2 0.64），门控会把好的 B2 判类替换成坏的 dense。car 门控伤害最大（−4.8），因为 dense car(0.72) 明显低于 B2(0.79)。最优门控集合严格等于 {barrier}。

---

### I — 通用置信度门控 / 拒识曲线（`eval_confidence_gate.py`）

把 barrier 的「类别门控」推广为通用「置信度门控」：对每个 B2 判类实例算 softmax 不确定度，按不确定度从高到低「拒识」（交人工），量化剩余点精度。这回答「知道什么时候不知道」，是开集识别/主动学习入口。

方法：一次前向缓存每实例 (GT, B2 预测, B2 softmax)，纯后处理画拒识曲线（keep_ratio = 保留最确信点占比 → 剩余 thing 精度）。缓存落盘 `experiments/confidence_gate_cache.npz`。**注：只统计被聚类 thing 点（567009 点），噪声点不计入，故基线 0.7123 > D 的 0.6901（后者把未聚类点按判错计）。**

三种 B2 softmax 信号 + dense↔B2 分歧 + 随机对照（500 val，eps=0.5）：

| 信号 | keep 1.0(基线) | keep 0.9 | keep 0.8 | keep 0.7 | keep 0.6 | keep 0.5 |
|---|---|---|---|---|---|---|
| maxp(最大概率) | 0.7123 | 0.7493 | 0.7821 | 0.8170 | 0.8290 | 0.8550 |
| margin(top1-top2) | 0.7123 | 0.7480 | 0.7772 | 0.8161 | 0.8275 | 0.8467 |
| **entropy(熵)** | 0.7123 | 0.7478 | 0.7844 | 0.8181 | 0.8269 | **0.8571** |
| random(对照) | 0.7123 | 0.7123 | 0.7121 | 0.7123 | 0.7122 | 0.7115 |

dense↔B2 分歧（二值信号）：

| 子集 | 点数 | 精度 |
|---|---|---|
| 一致（dense 多数投票 == B2 判类） | 234620 | **0.9526** |
| 分歧 | 332389 | 0.5426 |

**结论**：
1. **三种 softmax 信号都有效**（单调、明显高于 random 平坦对照），entropy 略优。拒掉最不确定的一半 → +14.5 点（0.71→0.86）；拒掉 10% → +3.7 点。证明 B2 softmax 本身校准良好，能区分对错。
2. **dense↔B2 分歧是最强置信度代理**：一致时 B2 几乎全对（0.95），分歧时只 0.54。比单一 softmax 强得多——因为分歧抓住了「B2 与独立 dense 通路意见相左」的结构性不确定。但分歧点占 58.6%，全拒会损失一半以上数据，需按任务设定阈值。
3. 这正面回答了「门控能否推广到整模型」：**能**。用 softmax 置信度 + 跨模块分歧做拒识，把不确定实例交人工，随数据/类别增长理论上覆盖越来越广（「识别所有物体」仍受开集性质限制）。

---

## 4. 关键发现总结

1. **解冻微调收益微小**：PQ +0.6、级联 +0.3。主要价值是联合损失顺带提升 dense 基线。
2. **聚类阈值 eps 是主导杠杆，且 PQ 与级联精度方向相反**：PQ 峰 eps≈1.0（0.606），级联峰 eps=0.5（0.690）。
3. **offset 回归价值在小 eps 聚紧**：@ eps=0.5 +13 点，@ eps=1.0 仅 +1.9。
4. **barrier 失败根因是 B2 分类混淆（→trailer 49.2%），不是实例分割**；修复 = 类别门控级联（dense 多数投票保 barrier），整体 +3.4 点。
5. **稀疏瓶颈不变**：16k 降采样后 84% thing 实例 <50 点，recall 天花板 ~0.46，整段漏检大量小物体。

---

## 5. 复现清单

### Checkpoint

| 文件 | 说明 |
|---|---|
| `experiments/cascade_models/ckpt-offset.pth` | 冻结 offset 头（PQ 0.5093） |
| `experiments/cascade_models/ckpt-offset-finetuned.pth` | 解冻微调（PQ 0.5155） |
| `experiments/cascade_models/ckpt-cls.pth` | B2 11 类分类器（macro 0.8741） |

### 脚本与实验

| 脚本 | 实验 | 关键输出 |
|---|---|---|
| `train_instance_offset.py` | C2 offset 训练 | `c2_offset_retrain.log` |
| `train_instance_finetune.py` | C3 解冻微调 | `c3_finetune.log` |
| `sweep_eps.py` | C4 eps 扫描 | `c4_eps_sweep*.log` |
| `eval_cascade_end2end.py` | D 端到端级联 | `d_end2end_eps*.log` |
| `analyze_barrier_failure.py` | E barrier 根因 | `e_barrier_failure.log` |
| `eval_cascade_gated.py` | F 门控级联 | `f_cascade_gated.log` |
| `eval_gate_sensitivity.py` | G 门控敏感性 | `g_gate_sensitivity.log` |
| `eval_confidence_gate.py` | I 置信度门控/拒识曲线 | `i_confidence_gate.log` |
| `build_semantic3d_dataset.py` | Semantic3D 转换 | `semantic3d_convert*.log` |

### 通用命令（500 val 端到端级联，门控）

```bash
/home/saw/ros2_ws/.venv/bin/python scripts/eval_cascade_gated.py \
  --val data/nuscenes_seg_full_16k/val \
  --offset_ckpt experiments/cascade_models/ckpt-offset-finetuned.pth \
  --cls_ckpt experiments/cascade_models/ckpt-cls.pth \
  --n 500 --eps 0.5 --min_points 3 --gate 9 --seed 0
```

---

## 6. 跨数据集泛化（Semantic3D）

第二真实室外 LiDAR 集 semantic-8（15 训练场景，7 列 `x y z intensity r g b`，`.labels` 0-8）。用于验证 RGB+XYZ 早期融合分割前置的跨数据集泛化。转换脚本 `build_semantic3d_dataset.py` 支持 `--sample_mode all|labeled`（labeled = 标签偏置采样，产出 100% 有效标签）。详见 `semantic3d_convert*.log`。

### 6.1 零样本迁移（`eval_semantic3d_zeroshot.py`）

nuScenes 32 类模型不做再训练，经文档化类别映射在 Semantic3D 8 类上直接预测：

| 指标 | 值 |
|------|-----|
| OA | 0.536 |
| mIoU | 0.196 |
| 映射覆盖率 | 96.8% |

迁移良好：man-made terrain（recall 0.71）、high vegetation（0.76）、buildings（0.56）、natural terrain（0.43）、cars（0.30）。hard scape / low vegetation 为 0 是标签体系差异（nuScenes 将 high/low veg 合并为单一 vegetation；flat.other 与 hard scape 边界不一致）。

### 6.2 迁移学习（`transfer_semantic3d.py`）

冻结 nuScenes 预训练 backbone、仅微调 8 类头（200K/12.33M = 1.6%），12 train / 3 val：

| 组 | best mIoU |
|----|-----------|
| 冻结 backbone + 微调头 | **0.263** |
| 从零训练整模型 | 0.224 |

**结论**：迁移组仅 1.6% 参数即超从零整模型 +3.9 mIoU，验证 RGB+XYZ 早期融合特征跨数据集可迁移（即便仅 12 训练场景）。

---

## 7. 地图损毁补全可行性验证（实验 J，`demo_map_completion.py`）

探索方向：地图采集中规则结构（地面/墙面）的损毁能否用「几何 + 语义先验」补回（回应「能否用 DLSS 类技术补全地图损毁」的构想）。技术判断：DLSS 是 2D 渲染域超分（依赖 motion vector + 时序），不适用 3D 点云；真正对应的是点云补全/SSC，本项目已有语义先验 + 跨帧融合可接。

方法：Semantic3D 一帧（domfountain_station3，地面 45% + 建筑 47%），选离墙远近不同的三个地面点作洞中心，各挖 R=3m 洞（模拟遮挡损毁），用洞外环带 [3,5]m 的 RANSAC 平面拟合补全，对比两种点来源：纯几何（环带所有点）vs 语义引导（环带仅地面点）。度量 = 拟合平面法向 vs 真值地面法向的夹角 + 补全点覆盖率(<0.5m)。

| 洞位置 | 纯几何法向夹角 | 语义引导法向夹角 | 语义引导覆盖<0.5m |
|---|---|---|---|
| 近墙（离墙 0.1m） | **79.8°** | **0.2°** | 50.4% |
| 中间 | 0.3° | 0.3° | 83.2% |
| 纯地面 | 2.7° | 1.1° | 88.5% |

**结论**：
1. **规则平面（地面）可用平面拟合补回**：纯地面洞覆盖率 ~88%、mean 0.27m。
2. **语义引导的价值随洞周围结构混合度增强**：近墙洞纯几何被墙面带偏（法向夹角 79.8°，几乎把墙当平面），语义引导纠回 0.2°；单一主导平面区域纯几何已够（0.3°≈0.3°）。
3. 诚实结论：**语义先验在「混合结构区」才有决定性价值**，精确刻画了「语义先验何时有用」。后续若立「Semantic-aware Map Completion」题，可接语义分割 + 置信度门控做拒识，避免把幻觉写进地图。


<!-- ==================== 2. 语义分割实验对照表 ==================== -->

# PointMambaSeg 语义分割实验对照表（nuScenes-lidarseg）

> 实验设置：输入 RGB+XYZ（6 通道早期融合），模型 PointMambaSeg（trans_dim=384, depth=12,
> group_size=32, num_group=64, encoder_dims=384, 参数量约 12.34M）。
> 数据集：nuScenes 官方划分（700 train / 150 val scenes，共 34,149 keyframes）。
> 指标：val 集逐类 IoU → mIoU（排除无 GT 类）、OA（整体准确率）。
> 存档 JSON 与逐类 IoU 详见 `results/` 目录同名文件。

| # | 实验 | 点数 | 输入 | Loss | mIoU | OA | 备注 |
|---|------|------|------|------|------|----|------|
| 1 | baseline_4096 | 4096 | RGB+XYZ (6) | 交叉熵 (无权重) | 0.2395 | 0.8294 | 基线；rare thing 类近乎全灭（animal/child/police/stroller=0，ambulance/wheelchair 无 GT） |
| 2 | seg_16384 | 16384 | RGB+XYZ (6) | 交叉熵 (无权重) | 0.2323 | 0.8095 | **负结果**：单纯升点数略降。常见类 car/barrier/driveable/ego 提升，但 manmade/vegetation/noise 下降；瓶颈在 num_group=64 聚合而非输入分辨率 |

## 逐类 IoU（baseline_4096，val 集）

| idx | 类别 | IoU | GT 点数 |
|-----|------|-----|---------|
| 0 | noise | 0.187 | 286,897 |
| 1 | animal | 0.000 | 32 |
| 2 | human.pedestrian.adult | 0.034 | 59,005 |
| 3 | human.pedestrian.child | 0.000 | 273 |
| 4 | human.pedestrian.construction_worker | 0.004 | 2,159 |
| 5 | human.pedestrian.personal_mobility | 0.000 | 4 |
| 6 | human.pedestrian.police_officer | 0.000 | 234 |
| 7 | human.pedestrian.stroller | 0.000 | 175 |
| 8 | human.pedestrian.wheelchair | NaN | 0 |
| 9 | movable_object.barrier | 0.339 | 111,838 |
| 10 | movable_object.debris | 0.010 | 737 |
| 11 | movable_object.pushable_pullable | 0.025 | 6,918 |
| 12 | movable_object.trafficcone | 0.139 | 15,477 |
| 13 | static_object.bicycle_rack | 0.367 | 3,667 |
| 14 | vehicle.bicycle | 0.001 | 4,866 |
| 15 | vehicle.bus.bendy | 0.078 | 11,275 |
| 16 | vehicle.bus.rigid | 0.166 | 77,134 |
| 17 | vehicle.car | 0.447 | 529,902 |
| 18 | vehicle.construction | 0.114 | 33,247 |
| 19 | vehicle.emergency.ambulance | NaN | 0 |
| 20 | vehicle.emergency.police | 0.003 | 1,313 |
| 21 | vehicle.motorcycle | 0.014 | 6,581 |
| 22 | vehicle.trailer | 0.290 | 102,716 |
| 23 | vehicle.truck | 0.201 | 279,122 |
| 24 | flat.driveable_surface | 0.827 | 3,683,996 |
| 25 | flat.other | 0.476 | 173,726 |
| 26 | flat.sidewalk | 0.450 | 1,100,923 |
| 27 | flat.terrain | 0.519 | 1,364,953 |
| 28 | static.manmade | 0.762 | 9,075,822 |
| 29 | static.other | 0.013 | 41,114 |
| 30 | static.vegetation | 0.779 | 7,598,127 |
| 31 | vehicle.ego | 0.939 | 81,591 |

## 逐类 IoU（seg_16384，val 集）

| idx | 类别 | IoU | GT 点数 | vs 4096 |
|-----|------|-----|---------|---------|
| 0 | noise | 0.130 | 901,288 | ↓ |
| 1 | animal | 0.000 | 132 | = |
| 2 | human.pedestrian.adult | 0.083 | 239,312 | ↑ |
| 3 | human.pedestrian.child | 0.000 | 797 | = |
| 4 | human.pedestrian.construction_worker | 0.014 | 9,439 | ↑ |
| 5 | human.pedestrian.personal_mobility | 0.000 | 5 | = |
| 6 | human.pedestrian.police_officer | 0.000 | 1,105 | = |
| 7 | human.pedestrian.stroller | 0.000 | 429 | = |
| 8 | human.pedestrian.wheelchair | NaN | 0 | = |
| 9 | movable_object.barrier | 0.418 | 781,619 | ↑ |
| 10 | movable_object.debris | 0.017 | 2,498 | ↑ |
| 11 | movable_object.pushable_pullable | 0.070 | 46,504 | ↑ |
| 12 | movable_object.trafficcone | 0.155 | 64,698 | ↑ |
| 13 | static_object.bicycle_rack | 0.364 | 12,290 | ≈ |
| 14 | vehicle.bicycle | 0.000 | 16,337 | ↓ |
| 15 | vehicle.bus.bendy | 0.062 | 36,993 | ↓ |
| 16 | vehicle.bus.rigid | 0.172 | 465,658 | ↑ |
| 17 | vehicle.car | 0.476 | 3,272,864 | ↑ |
| 18 | vehicle.construction | 0.078 | 125,748 | ↓ |
| 19 | vehicle.emergency.ambulance | NaN | 0 | = |
| 20 | vehicle.emergency.police | 0.000 | 8,490 | ↓ |
| 21 | vehicle.motorcycle | 0.005 | 43,313 | ↓ |
| 22 | vehicle.trailer | 0.210 | 297,946 | ↓ |
| 23 | vehicle.truck | 0.166 | 1,513,813 | ↓ |
| 24 | flat.driveable_surface | 0.855 | 26,615,082 | ↑ |
| 25 | flat.other | 0.509 | 1,110,640 | ↑ |
| 26 | flat.sidewalk | 0.459 | 6,853,766 | ↑ |
| 27 | flat.terrain | 0.535 | 7,727,198 | ↑ |
| 28 | static.manmade | 0.725 | 27,606,932 | ↓ |
| 29 | static.other | 0.005 | 172,149 | ↓ |
| 30 | static.vegetation | 0.718 | 19,389,069 | ↓ |
| 31 | vehicle.ego | 0.975 | 1,299,182 | ↑ |

> 注：16384 点数据集每帧点数 4×，故 val 集 GT 点数约为 4096 基线的 3–4×（rare 类 GT 点更多，
> 但 rare thing 类 IoU 仍近 0）。mIoU 0.2395→0.2323、OA 0.8294→0.8095 为略降的**负结果**。

## 改进计划（按优先级）

1. ✅ **升点数 4096 → 16384** — 已完成；**负结果**（mIoU 0.2395→0.2323）。结论：在 `num_group=64`
   固定的 PointMambaSeg 上，单纯升点数无法提升（瓶颈在 64 个 group 的聚合，非输入分辨率）
2. ⬜ **加权 / focal loss**（平衡长尾类分布，救 rare thing 类）— 下一优先
3. ⬜ **更强 decoder / 更长训练**（更多 epoch、更大 encoder 或 group 数）


<!-- ==================== 3. 导航实验日志 ==================== -->

# 语义物理软代价导航 —— 导航证据链实验记录

> 记录日期：2026-10-09
> 目标：把「语义物理软代价」落进 Nav2 规划器，用三组对照 + 权衡曲线 + 材料敏感性，
> 定量回答「语义物理可穿越性」相对几何障碍 / 简单语义禁行基线新增了什么、新增部分为什么有效。
> 场景：Semantic3D domfountain_station3 50×50m 子区域（模型预测语义 PLY，668MB，3520 万点级，
> 裁剪 x/y ∈ [-25,25)，0.2m 栅格，250×250）。
> 纯 CPU，可与 GPU 训练并行。

---

## 0. 实验体系与脚本

| 项 | 脚本 / 文件 |
|---|---|
| 语义 PLY → 静态地图 + 软代价 | `scripts/semantic3d_to_nav.py`（`--mode soft_cost/geometric/semantic_ban`、`--hard-ceil`、`--soft-low/--soft-high/--lethality-scale`、`--cost-only`） |
| 软代价发布（cost.npz → /semantic_cost） | `scripts/semantic_costmap_publisher.py`（`--cost-npz`） |
| 规划 + 语义真值评估 | `scripts/plan_and_analyze.py`（调 `/compute_path_to_pose` + `--truth-npz` 评估尺子） |
| Nav2 启动 | `launch/nav2_realmap.launch.py`（`map/cost_npz/use_semantic` 参数） |
| 三组对照一键 | `run_chains/run_nav_compare.sh` |
| hard 权衡曲线一键 | `run_chains/run_hard_sweep.sh` |
| 材料敏感性一键 | `run_chains/run_material_sensitivity.sh` |

**代价分离（关键设计）**：planner 代价（`class_to_cost`）与评估尺子（`eval_cost`）分离——
planner 把 hard 与 lethal 都标 254（方案 a，都不可穿越）；评估尺子保留 lethal=254 / hard=200 /
soft=46 / free=0 的区分，单独报告「穿行人」vs「穿车墙」。评估尺子对所有组固定，保证公平对比。

---

## 1. B① 导航三组对照（v1，方案 a 修正前）

三种静态地图 + 同一 Nav2 planner，5 对起终点（soft_cost 最大 free 块内，间距 15~25m）：

| mode | 成功 | 平均长度(m) | 平均语义代价 | 平均 hard 侵入 | 平均最小 lethal 间距(m) |
|---|---|---|---|---|---|
| soft_cost（本文，仅 lethal occupied + hard/植被软代价 185~220/30~80） | **5/5** | 23.56 | 5.52 | 6.0 | 1.93 |
| geometric（非地面即障碍） | 3/5 | 22.92 | 0.70 | 0.67 | — |
| semantic_ban（lethal+hard occupied） | 4/5 | 26.75 | 0.71 | 0.5 | — |

**发现**：soft_cost 连通性/路径最短最优，但 **hard 避让不足**——hard 代价 185~220 不足以让 Navfn
绕开实心车/墙/建筑，起终点间有 hard 区时 planner 选择「穿 hard」而非绕行。→ 接 B③ 权衡曲线。

---

## 2. B③ hard 代价权衡曲线

复用 static map（仅 lethal occupied）+ 固定评估尺子，扫 planner 语义代价 `hard_ceil ∈ {220,235,245,250,254}`：

| hard_ceil | 成功 | 平均长度(m) | 平均语义代价 | 平均 hard 侵入 | 平均 lethal 间距(m) |
|---|---|---|---|---|---|
| 220 | 5/5 | 23.56 | 5.52 | 6.0 | 1.93 |
| 235 | 5/5 | 23.72 | 5.93 | 6.6 | 1.92 |
| 245 | 5/5 | 24.09 | 6.61 | 7.6 | 1.93 |
| 250 | 5/5 | 24.18 | 5.02 | 5.6 | 1.93 |
| 254 | 4/5 | 26.68 | 0.58 | 0.25 | 1.13 |

**悬崖式权衡（关键）**：`hard_ceil < 254`（软代价）区间 planner 行为几乎不变（hard 侵入 5.6~7.6），
`=254`（=lethal）时突变（hard 侵入骤降 0.25，但成功率 5/5→4/5）。根因：**Navfn 对 cost<253 一律当
「可穿越软代价」，只有二元「走/不走」，无「强避让」连续梯度**，调 220→250 无效。

**语义根源**：`utils/materials.py` 的 `safety` 用「碰撞致命性」而非「可穿越性」——hard（车/墙/建筑，
lethality 0.7~0.9）被当「软代价可穿越」，但实体物理上不可穿越（撞上会停）。pair3 直线穿过的 hard
连通块达 926 cell（37m² 连续车/建筑区）= 穿实心非穿间隙。

---

## 3. 方案 a 修正（hard → 绝对障碍）

把 hard（车/墙/建筑）从「软代价」改为「绝对障碍 254」，仅 frangible/deformable（植被/护栏/可变形物）
保留软代价。改动：

- `semantic3d_to_nav.py`：`class_to_cost` hard 分支 → `return 254`；occupied 判定 `safety==0` → `safety<=1`
  （lethal+hard）；新增 `eval_cost` 评估尺子（lethal=254/hard=200/soft=46/free=0）。
- `semantic_costmap_publisher.py`：`class_to_cost` hard → 254（一致性）。

---

## 4. B① 三组对照（v2，方案 a 修正后）

方案 a 后重新选点（从 lethal+hard excluded 的 free 块选，free 28830→16667 cell）：

| mode | 成功 | 平均长度(m) | 平均语义代价 | 平均 hard 侵入 | 平均最小 lethal 间距(m) |
|---|---|---|---|---|---|
| soft_cost（本文） | 5/5 | 23.822 | **0.656** | **0.2** | 2.886 |
| geometric | 5/5 | **22.321** | 0.604 | 0.6 | **4.226** |
| semantic_ban | 5/5 | 24.609 | 1.344 | 0.4 | 2.884 |

**关键变化（诚实）**：三组都 5/5，区分度大幅缩小。根因有二：
1. **方案 a 使 soft_cost ≈ semantic_ban**：hard 标绝对障碍后两者 static map 相同，唯一区别是「植被
   软代价 30~80 vs 植被 free 0」。
2. **domfountain 是火车站，植被仅 2.6%**（1652 cell / 202 零散块 / 最大 356 cell），hard 占 19.5%
   （12163 cell）——「植被软代价」的差异化在植被稀疏场景天然微弱。
3. **v1 的「geometric 3/5 失败」不再成立**：它依赖「hard 软代价可穿越」这个已被否证的旧配置；方案 a
   后 hard 不可穿越，重选点落在连续地面主块（14215 cell≈227m²，占地面 94.6%），三组都能连通。

---

## 5. B④ 材料/代价先验敏感性

扰动「可穿越软代价」的两个连续先验维度，复用方案 a static map + 固定评估尺子，5 对起终点：

| cfg | 扰动 | vegetation 软代价 | 成功 | 平均长度(m) | 语义代价 | hard 侵入 | soft 侵入 | lethal 间距(m) |
|---|---|---|---|---|---|---|---|---|
| soft_10_40 | soft 范围 10~40 | 20 | 5/5 | 23.897 | 0.838 | 0.2 | 4.2 | 2.883 |
| soft_30_80 | **基准** | 46 | 5/5 | 23.822 | 0.656 | 0.2 | 3.2 | 2.886 |
| soft_50_120 | soft 范围 50~120 | 73 | 5/5 | 23.838 | 0.508 | 0.2 | 2.4 | 2.877 |
| ls_0p5 | lethality ×0.5 | 35 | 5/5 | 23.833 | 0.654 | 0.2 | 3.2 | 2.885 |
| ls_1p5 | lethality ×1.5 | 57 | 5/5 | 23.815 | 0.566 | 0.2 | 2.8 | 2.876 |

**结论**：
1. **先验高度鲁棒（正面）**：soft 代价 20~73、lethality 缩放 0.5~1.5 范围内，成功率恒 5/5、路径长度差
   <0.08m、hard 侵入恒 0.2、lethal 间距恒 2.88m——手工先验的合理误差不改变核心安全结论。
2. **软代价有单调但微弱的避让（诚实局限）**：soft 侵入 4.2→3.2→2.4 单调下降，但仅差 1.8 cell、路径长度
   几乎不变。根因是植被稀疏 + Navfn 弱梯度，印证方案 a 后「植被软代价差异化需植被密集场景」的判断。

---

## 6. 结论与待办

- **已确立**：语义物理「可穿越性分级」（lethal+hard 绝对障碍 vs 植被可穿越软代价）在导航端物理正确
  （方案 a 修正了 hard 软代价穿实心障碍的错误）；软代价先验对导航决策鲁棒。
- **诚实局限**：domfountain（火车站）植被稀疏（2.6%）、行人稀少（lethal 26 cell），「植被软代价」的
  差异化价值在此场景信号微弱——需换植被密集场景（sg27/sg28/untermaederbrunnen）充分体现（等 GPU 空闲）。
- **待办**：B② 感知误差传递（nuScenes 三组，需 GPU 分割预测）、换植被密集场景重做导航、D⑧ 时延鲁棒性。

结果 JSON：`results/nav_compare/`（v1）、`results/nav_compare_v2/`（v2）、`results/hard_sweep/`（B③）、
`results/material_sensitivity/`（B④）。相关：[[realmap-semantic-nav]]、[[pgm-map-server-encoding]]。


<!-- ==================== 4. B① 导航三组对照 v1 ==================== -->

# B① 导航三组对照（障碍表达方式消融）

Semantic3D `domfountain_station3` 50×50m 子区域，5 对起终点（soft_cost 最大 free 连通块内，间距 15~25m），
同一 Nav2 planner（Navfn，footprint 0.1×0.1m，inflation 0.1m），统一用「语义真值代价网格」评估路径。

## 三种模式

| 模式 | 静态地图 occupied | 连续语义代价层 |
|---|---|---|
| **soft_cost（本文）** | 仅 lethal（行人/动物） | hard 185~220、frangible/deformable 30~80、passable 0 |
| **geometric（几何基线）** | 非地面（含车/墙/植被） | 无 |
| **semantic_ban（简单语义基线）** | lethal + hard | 无 |

## 结果（5 对起终点统计）

| 指标 | soft_cost | geometric | semantic_ban |
|---|---|---|---|
| 规划成功率 | **5/5 (100%)** | 3/5 (60%) | 4/5 (80%) |
| 平均路径长度 (m) | **23.56** | 22.92 | 26.75 |
| 平均语义代价（越低越好） | 5.52 | 0.70 | 0.71 |
| 平均 hard 侵入 (cell) | 6.0 | 0.67 | 0.5 |
| 平均最小 lethal 间距 (m) | 1.93 | 3.74 | 1.13 |

> 注：三组都把 lethal 标 254 绝对障碍，**lethal 侵入全部为 0**（人身安全均达标）。

## 关键发现

1. **连通性**：soft_cost（仅 lethal 二元，其余软代价）保持 100% 连通性；geometric（非地面二元）在
   cluttered 区（车/建筑密集处）把地面通道切碎，Navfn 报 `Failed to create a plan from potential`，
   40% 起终点对失败；semantic_ban（lethal+hard 二元）折中，80%。
2. **路径效率**：soft_cost 最短（23.56m）；semantic_ban 因把 hard 当绝对障碍而绕行最远（26.75m，+13.5%）。
3. **权衡（诚实局限）**：soft_cost 的 hard 代价 185~220 不足以让 planner 绕开实心车/墙/建筑
   （hard 类全是金属车类+混凝土墙+岩石），起终点间有 hard 区时 planner 选择「穿 hard」（平均 6 cell/路径）
   而非绕行。这是 hard 代价权重缺陷，可用权衡曲线（hard 代价 185→254 扫描）找到「连通性 vs 硬障碍避让」
   的最优权衡点 —— 见 B③。

## 复现

```bash
# 1. 生成三种地图（CPU）
python3 scripts/semantic3d_to_nav.py -i paper/figures/map/domfountain_station3_map_semantic.ply \
    -o /tmp/s3_nav/soft_cost --xmin -25 --xmax 25 --ymin -25 --ymax 25 --cell 0.2 --mode soft_cost
python3 scripts/semantic3d_to_nav.py -i paper/figures/map/domfountain_station3_map_semantic.ply \
    -o /tmp/s3_nav/geometric --xmin -25 --xmax 25 --ymin -25 --ymax 25 --cell 0.2 --mode geometric
python3 scripts/semantic3d_to_nav.py -i paper/figures/map/domfountain_station3_map_semantic.ply \
    -o /tmp/s3_nav/semantic_ban --xmin -25 --xmax 25 --ymin -25 --ymax 25 --cell 0.2 --mode semantic_ban
# 2. 三组 × 5 对自动对比
bash run_chains/run_nav_compare.sh
```


<!-- ==================== 5. B③ hard 代价权衡曲线 ==================== -->

# B③ hard 代价权衡曲线（hard_ceil 扫描）

复用 B① 的 static map（仅 lethal occupied）+ 固定评估尺子（`soft_cost_cost.npz`，hard_ceil=220 语义真值），
只换 planner 实际发布的语义代价（`hard_ceil ∈ {220,235,245,250,254}`），5 对起终点。

## 结果

| hard_ceil | 成功率 | 平均长度(m) | 平均语义代价 | 平均 hard 侵入(cell) | 平均 lethal 间距(m) |
|---|---|---|---|---|---|
| 220 | 5/5 | 23.56 | 5.52 | 6.0 | 1.93 |
| 235 | 5/5 | 23.72 | 5.93 | 6.6 | 1.92 |
| 245 | 5/5 | 24.09 | 6.61 | 7.6 | 1.93 |
| 250 | 5/5 | 24.18 | 5.02 | 5.6 | 1.93 |
| 254 | 4/5 | 26.68 | 0.58 | 0.25 | 1.13 |

## 关键发现（悬崖式权衡）

1. **`hard_ceil < 254`（软代价）区间，planner 行为几乎不变**：hard 侵入始终 5.6~7.6 cell，
   语义代价 5~6.6。把 hard 代价从 220 抬到 250 不能让 Navfn 更强地绕开 hard —— 因为 Navfn 对
   `cost < 253` 一律当「可穿越软代价」，只有「走/不走」的二元决策，没有「强避让」的连续梯度。
2. **`hard_ceil = 254`（= lethal 绝对障碍）时行为突变**：hard 侵入骤降到 0.25，语义代价 0.58，
   但成功率 5/5 → 4/5（pair3 起终点间被 254 hard + inflation 阻断）。
3. **硬障碍穿越是「穿实心」而非「穿间隙」**：pair3 直线路径穿过的 hard 连通块达 926 cell（约 37m²，
   连续车/建筑密集区）——soft_cost 的 hard 软代价（185~220）不足以让 planner 绕开大片实心障碍区，
   规划出「穿墙」路径。

## 语义根源

材质库 `utils/materials.py` 的 `safety` 分级用**碰撞致命性**（lethality）而非**可穿越性**：
hard（车/墙/建筑，lethality 0.7~0.9）被归为「软代价可穿越」，但这些实体物理上**不可穿越**
（撞上会停，不是穿过去）。「碰撞不致命」≠「可穿越」。

→ 导航端正确语义应区分两个维度：
- **致命性**（撞了是否伤人）：行人 = 致命 → 最高避让优先级
- **可穿越性**（能否压过去/穿过去）：车墙 = 不可穿越 → 绝对障碍；植被/护栏 = 可穿越 → 软代价

## 对论文的影响

`soft_cost` 的「hard 软代价」在 Navfn 下无法实现「强避让」，这是「语义物理软代价层」的一个真实局限，
需在论文 Discussion 诚实呈现，并给出修正方向：
(a) hard（车墙）在导航端标绝对障碍（≈ semantic_ban），仅 frangible/deformable 用软代价；
(b) 用考虑 footprint 的 planner（SMAC hybrid-A*）替代 Navfn 的 potential 场；
(c) 对 hard 区做「禁行环」（外围 cell 标 254，内部保留软代价）实现「贴边绕、不穿越」。

## 复现

```bash
bash run_chains/run_hard_sweep.sh   # 生成 5 个 hard_ceil 的 cost.npz + 5 对起终点规划 + 汇总
```


<!-- ==================== 6. B① 导航三组对照 v2 ==================== -->

# B① 导航三组对照（方案 a 修正后，v2）

方案 a 落地：`hard`（车/墙/建筑，物理不可穿越）在导航端标绝对障碍 254，仅 `frangible/deformable`
（植被/护栏/可变形物）保留连续软代价 30~80。`semantic3d_to_nav.py` 的 `class_to_cost` hard 分支
`185~220` → `254`，occupied 判定 `safety==0` → `safety<=1`（lethal+hard），并新增 `eval_cost` 评估尺子
（lethal=254/hard=200/soft=46/free=0，独立于 planner 代价，单独报告「穿行人」vs「穿车墙」）。

## 结果（5 对起终点，seed 42）

| mode | 成功 | 平均长度(m) | 平均语义代价 | 平均 hard 侵入 | 平均最小 lethal 间距(m) |
|---|---|---|---|---|---|
| soft_cost（本文） | 5/5 | 23.822 | 0.656 | 0.2 | 2.886 |
| geometric | 5/5 | 22.321 | 0.604 | 0.6 | 4.226 |
| semantic_ban | 5/5 | 24.609 | 1.344 | 0.4 | 2.884 |

## 关键变化 vs v1（诚实、反直觉）

1. **三组都 5/5，区分度大幅缩小**。根因有二：
   - **方案 a 使 soft_cost ≈ semantic_ban**：hard 标绝对障碍后，两者 static map 相同
     （lethal+hard occupied），唯一区别是「植被软代价 30~80 vs 植被 free 0」。
   - **domfountain 是火车站，植被仅占 2.6%**（1652 cell，202 个零散小块，最大连通块 356 cell），
     而 hard（车墙建筑）占 19.5%（12163 cell）。「植被软代价」的差异化信号在植被稀疏场景天然微弱。
2. **B① 的「geometric 3/5 失败」不再成立**：v1 的 geometric 失败依赖「hard 软代价可穿越」这个
   已被否证的旧配置（起终点落在 hard 软代价区，geometric 把 hard 当障碍才失败）。方案 a 后
   hard 不可穿越，重新选点（从 lethal+hard excluded 的 free 区）后起终点都落在连续地面主块
   （14215 cell ≈ 227m²，占地面 94.6%），三组都能连通。

## 逐对明细

- pair2 / pair3：起终点在开阔地面区，路径不穿植被不穿 hard，三组几乎相同（长度 16.3/18.9m，soft 0~1）。
- pair0：semantic_ban 更绕（29.17m vs 25.3m），植被 free 无梯度。
- pair1：semantic_ban 穿植被最多（soft 侵入 10，代价 2.76），soft_cost 最少（soft 1，代价 0.2）。
- pair4：geometric 走捷径但穿 hard（hard 2，25.29m），soft_cost/semantic_ban 绕 hard 走长路（33.99m）。

## 结论与待决方向

方案 a 是物理正确的修正（B③ 已证 hard 软代价 <253 会让 Navfn 穿实心车/墙/建筑区），但修正后
「语义物理软代价」的差异化价值只剩「植被软代价」一维，而 domfountain（火车站）植被稀疏
（2.6%），不足以体现「植被可穿越性」的核心对比。待决：
- **方向 A**：换植被密集场景（住宅/公园类 Semantic3D 场景，等 GPU 空闲后 infer_map 推理重做导航）。
- **方向 B**：接受现状，叙事聚焦「可穿越性分级正确性」（lethal+hard 绝对障碍 vs 植被可穿越），
  诚实标注植被稀疏局限。
- **方向 C**：改进选点（跨植被带起终点对），但植被零散（最大块 356 cell），增益有限。


<!-- ==================== 7. B④ 材料敏感性 ==================== -->

# B④ 材料/代价先验敏感性（导航端）

审稿「材料/代价先验的依据和敏感性分析」。扰动「可穿越软代价」的两个连续先验维度，
复用方案 a 的 static map（`soft_cost.yaml`，lethal+hard occupied）+ 固定评估尺子
（`soft_cost_eval.npz`），只换 planner 实际发布的语义代价，同一批 5 对起终点（seed 42）。

`scripts/semantic3d_to_nav.py` 新增 `--soft-low/--soft-high/--lethality-scale`；
`run_chains/run_material_sensitivity.sh` 一键扫描。纯 CPU。

## 扰动维度（domfountain 软目标以 vegetation 为主，lethality=0.20）

| cfg | 扰动 | vegetation 软代价 |
|---|---|---|
| soft_10_40 | soft 范围 10~40 | 20 |
| soft_30_80 | **基准**（= 方案 a） | 46 |
| soft_50_120 | soft 范围 50~120 | 73 |
| ls_0p5 | lethality ×0.5 | 35 |
| ls_1p5 | lethality ×1.5 | 57 |

## 结果

| cfg | 成功 | 平均长度(m) | 平均语义代价 | hard 侵入 | soft 侵入 | 最小 lethal 间距(m) |
|---|---|---|---|---|---|---|
| soft_10_40 | 5/5 | 23.897 | 0.838 | 0.2 | 4.2 | 2.883 |
| soft_30_80（基准） | 5/5 | 23.822 | 0.656 | 0.2 | 3.2 | 2.886 |
| soft_50_120 | 5/5 | 23.838 | 0.508 | 0.2 | 2.4 | 2.877 |
| ls_0p5 | 5/5 | 23.833 | 0.654 | 0.2 | 3.2 | 2.885 |
| ls_1p5 | 5/5 | 23.815 | 0.566 | 0.2 | 2.8 | 2.876 |

## 关键发现

1. **导航决策对软代价先验高度鲁棒**：soft 代价 20~73、lethality 缩放 0.5~1.5 范围内扰动，
   **成功率恒 5/5、路径长度几乎不变（23.815~23.897 m，差 <0.08 m）、hard 侵入恒 0.2、
   最小 lethal 间距恒 2.88 m**。手工 lethality / 软代价上下界的合理误差不会改变核心安全结论
   （不穿 hard、不贴 lethal）。
2. **soft 代价有「单调但微弱」的避让偏好**：soft 代价 20→46→73，路径 soft 侵入 4.2→3.2→2.4
   （越「贵」越绕植被），评估语义代价 0.838→0.656→0.508 单调下降——但幅度很小（soft 侵入仅差
   1.8 cell，路径长度几乎不变）。
3. **微弱的原因是「植被稀疏 + Navfn 弱梯度」而非方法失效**：domfountain 植被仅 2.6%（1652 cell），
   且 Navfn 对 cost<253 只有弱避让（呼应 B③ 的悬崖式权衡），故 soft 代价的连续梯度在 20~73 区间
   只能产生弱偏好。这印证方案 a 后的核心判断——「植被软代价」的差异化价值需在植被密集场景才充分体现。

## 对论文的意义

- **正面**：材料先验（手工 lethality、软代价上下界）的合理扰动不改变导航安全结论 → 先验粗糙可接受，
  敏感性低。
- **诚实局限**：软代价「能绕就绕」的绝对效果弱（植被稀疏场景），且对绝对数值不敏感——需与方案 a 后
  的「植被稀疏局限」一并写入 Discussion。

## 复现

```bash
bash run_chains/run_material_sensitivity.sh   # 生成 5 组扰动 cost.npz + 5×5 导航 + 汇总
```

<!-- ==================== 8. B② 感知误差传递 ==================== -->

## B② 感知误差传递（2026-10-10）

> 审稿风险 ②「预测误差传递实验」：导航用分割预测语义生成代价图，但分割 mIoU 只有 ~0.61。
> 本实验量化「感知误差 → 导航代价」的传递——三组语义（GT 零误差上界 / 3D 分割预测 / 3D+2D 检测融合）
> 各自映射到导航代价五档（方案 a 口径），输出完整代价混淆矩阵，聚焦「致命类漏检成可穿越软代价/自由地面」
> 的危险混淆率，以及 2D 检测辅助能救回多少。

### 方法

- **数据**：nuScenes val 6019 帧（每帧 4096 点，32 类 GT 完整，含 person/animal）；模型 `merged_pretrain_ft`
  （最终生产模型，12 类，mIoU 0.6132，ckpt `.../merged_pretrain_ft/20260927-070312/ckpt-best.pth`）。
- **三组语义**：GT 语义（32 类→12 类）、3D 分割预测、3D+2D 检测门控投票（RT-DETR person 框深度门控 +
  置信度 θ 门控，复用 `eval_gated_voting_12c.py` 的融合管线）。
- **导航代价五档**（方案 a 口径，12 类 → 导航代价）：lethal(254)=animal+person；hard(254)=car/bus/truck/
  two_wheeler/manmade/ego；soft(46)=vegetation+frangible；free(0)=ground；noise=未标注（单独一档）。
- **脚本**：`scripts/eval_nav_error_propagation.py`（新建，在 `eval_gated_voting_12c.py` 上新增 5×5 导航代价混淆矩阵）。
- **Semantic3D 不可行的原因**（已确认）：Semantic3D GT 每场景仅 16384 点稀疏降采样（0.047% 密度）+
  reduce-8 无 person 类，无法做「致命类漏检」对照，只能在 nuScenes 32 类完整 GT 上做。

### 结果：导航代价混淆矩阵（行=GT，列=预测/融合，行归一化 %）

| GT \ 预测 | lethal | hard | soft | free | noise |
|---|---|---|---|---|---|
| base · lethal | 64.82 | 25.98 | 5.15 | 3.48 | 0.56 |
| base · hard | 0.08 | 95.19 | 2.98 | 1.40 | 0.35 |
| base · soft | 0.03 | 4.27 | 93.22 | 2.16 | 0.32 |
| base · free | 0.01 | 1.09 | 0.99 | 97.87 | 0.05 |
| fused θ=0.9 · lethal | 84.28 | 10.04 | 3.88 | 1.64 | 0.16 |

### 聚焦指标（危险混淆率，lethal GT 共 251,219 点）

| 组 | critical(→soft) | severe(→free) | safe(→254) | hard 漏检(→soft/free) |
|---|---|---|---|---|
| GT 零误差上界 | 0% | 0% | 100% | 0% |
| base（3D 分割预测） | 5.149% | 3.484% | 90.802% | 2.981% / 1.402% |
| fused θ=0.5（+2D 检测） | 4.651% | 2.968% | 92.084% | 3.020% / 1.393% |
| fused θ=0.7（+2D 检测） | 4.040% | 2.320% | 93.460% | 3.104% / 1.374% |
| fused θ=0.9（+2D 检测） | 3.881% | 1.638% | 94.320% | 3.234% / 1.355% |
| fused oracle θ=0.9（+GT 框，上界） | 0.919% | 1.178% | 97.669% | — |

> GT 导航代价分布（零误差上界参照）：lethal=251,219、hard=34,871,715、soft=20,284,388、
> free=42,306,686、noise=901,288 点。oracle 数据来自 `results/gated_voting_12c_oracle.json`（模型 merged_v1）。

### 关键结论

1. **感知误差 → 导航危险的传递是「局部且可控」的**：分割 mIoU 0.613 看似不高，但传递到导航代价后，
   致命类 GT 点仅 5.15% 被漏检成可穿越软代价（critical）+ 3.48% 被漏检成自由地面（severe），
   合计「危险漏检」8.6%；其余 90.8% 致命类点仍被正确标为绝对障碍（254）。完整混淆矩阵对角线
   均 >90%（除 lethal 64.8% 因行人本身难分），证明误差传递是稀疏局部的、非全局错乱。

2. **2D 检测辅助显著降低最危险的 severe 漏检**：融合 RT-DETR person 检测后，θ=0.9 把 severe 从
   3.48% 压到 1.64%（相对 −53%）、critical 从 5.15% 压到 3.88%（−25%）；θ=0.7 是「安全-精度」平衡点
   （critical 4.04%、severe 2.32%，person IoU 不崩）。检测器质量是瓶颈——用 GT 框（oracle）可把
   critical 压到 0.92%，证明「3D 分割 residual 误差 + 2D 检测器漏检」共同构成上界。

3. **实体类（hard）漏检率低且对 2D 检测不敏感**：车/墙/建筑被漏检成 soft/free 仅 2.98%/1.40%，
   且 2D person 检测对 hard 类无回填作用（符合预期，hard 无 person 可检测）。方案 a 下 hard 与 lethal
   同为 254 绝对障碍，hard 漏检到 soft 会从「绝对障碍」降级为「软代价可穿越」，是导航安全的次危险项。

4. **与 gated_voting 自洽**：base critical 5.149% 与 merged_v1 的 5.302% 同量级（跨域迁移微降 0.15pp，
   呼应「person 微涨 + critical 降」），证明最终生产模型与实验模型的安全行为一致。

### 对论文的意义

- 回答审稿风险 ②「分割预测误差是否会破坏导航安全」：误差传递到导航端是可控的（8.6% 危险漏检，
  其中 2D 检测辅助把最危险的 severe 漏检再降约一半），致命类绝对障碍保持率 90.8%→94.3% 单调提升。
- 诚实呈现：critical/severe 无法归零（oracle 上界 0.92% 仍有 3D residual 误差），检测器质量是瓶颈，
  写入 Discussion。

### 复现

```bash
python3 scripts/eval_nav_error_propagation.py \
  --config cfgs/seg_16k_ng6144_g050_merged.yaml \
  --ckpts experiments/seg_16k_ng6144_g050_merged/cfgs/merged_pretrain_ft/20260927-070312/ckpt-best.pth \
  --proj data/nuscenes_projection_index.npz \
  --bbox data/nuscenes_2d_bbox_cache_rtdetr_s3_val_size.npz \
  --bs 4 --out-json results/nav_error_propagation_merged_pretrain_ft.json
```

---

## B① 植被密集场景补强（bildstein_station1，2026-10-10）

> 直接回应 domfountain 结果里的诚实局限：domfountain 植被仅 2.6%（火车站），方案 a 后三组区分度消失，
> 「植被软代价的差异化价值」信号被植被稀疏淹没。本实验换植被密集场景 bildstein_station1（完整语义 PLY
> 植被 46.5%），重做三组对照，验证「可穿越性分级」在植被密集下是否有可测价值。

### 场景与选窗

- **场景**：`bildstein_station1_map_semantic.ply`（564MB，29,697,591 点，infer_map 完整场景推理）。
  安全等级分布：lethal 234(0.0%) / hard 7,278,608(24.5%) / frangible 68,215(0.2%) /
  deformable **13,800,125(46.5%)** / passable 8,550,409(28.8%)。植被 46.5% 是 domfountain（<1%）的近 50 倍。
- **选窗坑**：初选窗口（植被 59.4%）地面被密集植被切成 110m² 碎块、cell=0.2 下 unknown 72%，
  起终点放不下。改用 5m 网格统计「passable 主导 cell」的连通块，按最大地面块面积重排候选，
  选 X[-25,25] Y[-28,22]（50×50m）：植被 50.4% / 地面 31.6% / hard 17.8%，最大地面块 1625m²，
  cell=0.2 时最大 free 连通块 416m²（20×20m，够放 15~25m 起终点对）。
- **地图**：`semantic3d_to_nav.py --mode soft_cost/geometric/semantic_ban --cell 0.2`，origin=(-25,-28)。

### 结果（5 对起终点，seed 42；成功率 / 平均长度 / 平均语义代价 / 平均 hard 侵入 / 平均 soft 侵入）

| mode | 成功 | 长度(m) | 语义代价 | hard 侵入 | soft 侵入 | min lethal 间距(m) |
|---|---|---|---|---|---|---|
| **soft_cost（本文）** | **5/5** | 25.22 | **14.00** | **0.0** | 75 | 9.25 |
| geometric（非地面即障碍） | **4/5** | 32.59 | 0.84 | 1.0 | 0 | 7.88 |
| semantic_ban（植被 free） | **5/5** | 24.77 | **16.40** | **0.0** | 85 | 9.13 |

### 关键发现（植被密集下区分度终于显现）

1. **连通性**：soft_cost 5/5 = semantic_ban 5/5 > geometric 4/5。geometric 把植被当硬障碍，
   切碎通道，pair1 起终点被植被带阻断（复现 domfountain geometric 3/5 的失败方向）。
2. **软代价引导少穿植被**（soft_cost vs semantic_ban 的核心差异）：soft_cost 语义代价 14.0 <
   semantic_ban 16.4（−14.6%），soft 侵入 75 vs 85 cell——植被软代价让 planner「能绕就绕」，
   仅多走 0.45m 换来少穿 10 cell 植被。semantic_ban 把植被当 free，走最短路、多穿植被。
3. **路径长度**：geometric 32.59 >> soft_cost 25.22（绕植被 +7.4m / +29%）——植被不当硬障碍、
   保留可穿越性的价值。
4. **安全达标**：hard 侵入 soft_cost/semantic_ban 全 0、geometric 1.0（绕行擦到 hard 边缘）；
   lethal 侵入三组全 0（人身安全达标）。

### 结论与定位（诚实）

- **soft_cost vs geometric** 是强区分度证据：连通性 100% vs 80%、路径 −29%，直接回答「为什么要可穿越
  性分级而不是纯几何障碍」。
- **soft_cost vs semantic_ban** 是「边际但方向正确」的收益：软代价引导少穿植被 14.6%（代价 14.0 vs
  16.4），代价仅 +0.45m。差异不巨大（Navfn 对 cost<253 只有二元走/不走的弱梯度是根因，见 B③），
  但方向与「能绕就绕、绕不开才穿」的设计一致。
- 这补上 domfountain（植被 2.6% 三组无区分）的局限，论文用 bildstein 作「植被密集场景」的差异化证据。

### 复现

```bash
# 1. 生成三组地图
python3 scripts/semantic3d_to_nav.py -i paper/figures/map/bildstein_station1_map_semantic.ply \
  -o /tmp/s3_nav_bild/soft_cost  --xmin -25 --xmax 25 --ymin -28 --ymax 22 --cell 0.2 --mode soft_cost
python3 scripts/semantic3d_to_nav.py -i paper/figures/map/bildstein_station1_map_semantic.ply \
  -o /tmp/s3_nav_bild/geometric --xmin -25 --xmax 25 --ymin -28 --ymax 22 --cell 0.2 --mode geometric
python3 scripts/semantic3d_to_nav.py -i paper/figures/map/bildstein_station1_map_semantic.ply \
  -o /tmp/s3_nav_bild/semantic_ban --xmin -25 --xmax 25 --ymin -28 --ymax 22 --cell 0.2 --mode semantic_ban
# 2. 三组 × 5 对自动对比（纯 CPU）
bash run_chains/run_nav_compare_bild.sh
```

---

## C⑥ 检测辅助消融梯度（2026-10-10）

> 回应审稿风险「2D 检测回填的哪些组件是必要的」：gated_voting 主实验验证了「深度门控 + 置信度门控」
> 完整融合能降 critical_rate（5.15%→3.88%），但没拆开「深度门控」与「置信度门控」各自的增量。
> 本实验做四路消融梯度，逐组件添加，量化每个门控的贡献，并诚实呈现「无门控直接回填」的代价。

### 四路消融（θ=0.9，模型 merged_pretrain_ft，nuScenes val 6019 帧，lethal GT 251,219 点）

| 回填方式 | 门控组合 | person | two_wheel | frangible | mIoU(全部) |
|---|---|---|---|---|---|
| 无回填 | — | 0.549 | 0.342 | 0.640 | 0.6132 |
| 直接回填 | 无深度 + 无置信度 | 0.206 | 0.136 | 0.364 | 0.5374 |
| 置信度门控 | 无深度 + 置信度 θ | 0.430 | 0.286 | 0.525 | 0.5877 |
| 置信度+深度门控 | 深度 + 置信度 θ（完整） | 0.533 | 0.361 | 0.573 | 0.6070 |

| 回填方式 | critical(→soft) | severe(→free) | safe(→254) |
|---|---|---|---|
| 无回填 | 5.149% | 3.484% | 90.802% |
| 直接回填 | 5.332% | 0.834% | 93.710% |
| 置信度门控 | 4.512% | 1.548% | 93.814% |
| 置信度+深度门控 | 3.881% | 1.638% | 94.320% |

### 关键发现

1. **无门控直接回填严重破坏精度**：person IoU 0.549→0.206（−62%）、mIoU 0.6132→0.5374（−0.076）。
   因为 bbox 是 2D 框，框内投影点包含大量不同深度的背景点，无条件覆盖成检测类 → 精度崩。
2. **置信度门控救回大部分**：person 0.206→0.430、mIoU 0.5374→0.5877（+0.050）；但相对无回填仍掉
   0.0255（无深度门控时 bbox 内低置信度背景点仍被误覆盖）。
3. **深度门控再补一档**：person 0.430→0.533、mIoU 0.5877→0.6070（+0.019），接近无回填 0.6132
   （仅 −0.006）——深度门控过滤掉 bbox 内不同深度的背景点，是「回填不破坏精度」的关键。
4. **安全指标诚实非单调（重要）**：直接回填把 severe 大降（3.484%→0.834%），但 critical 反而微升
   （5.149%→5.332%——部分行人点被 two_wheeler/frangible 框误覆盖成 collidable）；只有叠加置信度 +
   深度门控后，critical 才单调降到 3.881%（−24.6%）、safe 升到 94.320%。即「降 severe 易、降 critical
   难，必须完整门控」。

### 增量贡献（逐组件）

- **置信度门控增量**（direct→conf_gate）：person +0.224、mIoU +0.050、critical −0.820pp。
- **深度门控增量**（conf_gate→conf_depth）：person +0.103、mIoU +0.019、critical −0.631pp。
- 二者贡献可比，深度门控在「保精度」上更关键，置信度门控在「降 critical」上更关键。

### 对论文的意义

- 论文 §4.13–4.15 已报告完整融合的端到端收益，本消融补齐「组件消融梯度」：回答审稿人
  「深度门控/置信度门控各自是否必要」——都必要，缺一不可（缺深度门控精度崩、缺置信度门控 critical 不降）。
- 诚实呈现：回填本质是「用 2D 检测精度换 3D 分割的罕见类召回」，mIoU 始终略低于无回填（0.6070 vs
  0.6132），安全收益（safe +3.5pp）是其主要回报。

### 复现

```bash
python3 scripts/eval_gated_voting_ablation.py \
  --config cfgs/seg_16k_ng6144_g050_merged.yaml \
  --ckpts experiments/seg_16k_ng6144_g050_merged/cfgs/merged_pretrain_ft/20260927-070312/ckpt-best.pth \
  --proj data/nuscenes_projection_index.npz \
  --bbox data/nuscenes_2d_bbox_cache_rtdetr_s3_val_size.npz \
  --bs 4 --out-json results/gated_voting_ablation_merged_pretrain_ft.json
```

---

## D⑧ 时延 + 鲁棒性（2026-10-10）

> 回应审稿风险「实时性与鲁棒性」：论文 §4.16 只测了 2D 检测器时延（67.66ms），缺 3D 分割前端的
> 时延与鲁棒性。本实验补齐：(1) 3D 分割单帧/批前向时延；(2) 四类真实扰动下的分割精度下降。

### 时延分项（模型 merged_pretrain_ft，RTX 5070 Ti，torch.cuda.Event 计时，预热 10 次 + 100 次）

| 批大小 | mean | median | p95 | 吞吐 |
|---|---|---|---|---|
| B=1（单帧，导航实时口径） | 36.02 ms | 35.60 ms | 37.21 ms | 28.1 FPS |
| B=4（批，离线建图口径） | 139.1 ms | 138.82 ms | 140.5 ms | 均摊 34.7 ms/帧 |

> 单帧 36 ms 快于 §4.16 的 2D 检测器 67.66 ms（前向 55.44 ms）；按 50 km/h 折算一次分割前向的
> 感知滞后约 0.50 m，低于 2.5 m 安全制动余量。3D 分割 + 2D 检测的安全通道总滞后 ~104 ms（可并行）。

### 鲁棒性（nuScenes val 6019 帧，逐点注入扰动后全量重测 mIoU）

| 扰动 | mIoU | Δ | person | two_wheel | frangible | safe(致命类→254) |
|---|---|---|---|---|---|---|
| clean（无扰动） | 0.6132 | 0 | 0.549 | 0.342 | 0.640 | 90.80% |
| rgb_zero（颜色缺失） | 0.5202 | −0.093 | 0.381 | 0.113 | 0.449 | 77.50% |
| sparse_50（稀疏 50%） | 0.0723 | −0.541 | 0.005 | 0.002 | 0.038 | 37.06% |
| occlusion（前方 60° 遮挡） | 0.4091 | −0.204 | 0.148 | 0.132 | 0.518 | 86.50% |
| calib_010（标定噪声 σ=0.10m） | 0.2490 | −0.364 | 0.214 | 0.023 | 0.247 | 52.23% |

### 关键发现

1. **颜色缺失最鲁棒（正面）**：RGB 全置 0 仅掉 0.093 mIoU（−15.2%）、safe 仍 77.5%——模型以几何为主，
   RGB 是辅助（呼应「RGB 脱离几何判别力弱 0.054、早期融合净贡献 +0.0331」的消融）。颜色退化不会崩溃。
2. **点云稀疏极度敏感（诚实局限，最严重）**：稀疏 50% 掉 0.541 mIoU（−88%）、person 0.549→0.005、
   safe 90.8%→37.1%——模型对点密度敏感（grid 0.5m 体素化 + num_group=6144 分组机制依赖足量点覆盖体素）。
   注：稀疏实现为「随机保留 50% + 重复填充」，可能放大影响，但方向明确——稀疏是首要鲁棒性风险。
3. **遮挡中等敏感**：前方 60° 遮挡掉 0.204 mIoU（−33%）、person 掉 73%（行人多在前方），但 safe 仍 86.5%
   （致命类绝对障碍保持较好，因为 2D 检测安全通道可独立补）。
4. **标定噪声敏感**：σ=0.10 m 掉 0.364 mIoU（−59%）、safe 52.2%——体素边长 0.5 m 下 10 cm 噪声即可让点
   跨体素边界，标定误差是第二鲁棒性风险。

### 对论文的意义

- **实时性达标**：3D 分割 36 ms（28 FPS）满足实时导航，与 2D 检测器 67.66 ms 组成 ~104 ms 的并行安全
  通道，50 km/h 下感知滞后 ~1.4 m（串行）或 ~0.9 m（并行），低于常规制动余量。
- **鲁棒性诚实呈现**：颜色缺失鲁棒（−15%），但点稀疏（−88%）与标定噪声（−59%）敏感——写入 Discussion
  作为「部署鲁棒性边界」，并指出缓解方向（稀疏用 2D 检测安全通道兜底、标定需在线校准）。

### 复现

```bash
python3 scripts/eval_robustness.py \
  --config cfgs/seg_16k_ng6144_g050_merged.yaml \
  --ckpts experiments/seg_16k_ng6144_g050_merged/cfgs/merged_pretrain_ft/20260927-070312/ckpt-best.pth \
  --bs 4 --perturbs clean,rgb_zero,sparse_50,occlusion,calib_010 \
  --out-json results/robustness_merged_pretrain_ft.json
python3 scripts/eval_latency.py \
  --config cfgs/seg_16k_ng6144_g050_merged.yaml \
  --ckpts experiments/seg_16k_ng6144_g050_merged/cfgs/merged_pretrain_ft/20260927-070312/ckpt-best.pth \
  --n 100 --out-json results/latency_merged_pretrain_ft.json
```

---

## C⑤ 公平基线：PointNet++ / PointTransformer（2026-10-10）

> 回应审稿风险「基线不公平」：主模型 PointMamba（Mamba 序列建模）缺 point-based 对比基线。
> 本实验自实现两个代表性 point-based 分割模型——PointNet++（SSG）与 PointTransformer（向量自注意力）——
> 在**与 merged 基线完全一致的数据协议**下训练评测，隔离「backbone 网络结构」这一个变量。
> 框架内只有 PointMamba（无 pointcept），故两个 baseline 均从 pointnet2_ops 手工实现。

### 协议（三者一致，公平性前提）

- 数据：nuScenes-lidarseg，**16k 点 / 帧（16384）**、6 通道 XYZ+RGB、**12 类合并体系**（COARSE_MAP 32→12）。
- 训练：AdamW lr 0.001 / weight_decay 0.05、CosLR 40 epoch（5 warmup）、total_bs 2、
  step_per_update 4、grad_norm_clip 10、num_workers 8、seed 0。

### 三个模型各自的规范操作（论文须注明）

| 模型 | 参数 | 输入预处理 | 分组/邻域机制 |
|---|---|---|---|
| PointMamba（主） | — | 原始米制坐标 | 0.5m 网格体素 + num_group=6144 token 分组 |
| PointNet2Seg（C⑤） | 0.88 M | per-sample 零均值单位球归一化 | ball query（固定半径，scale-sensitive） |
| PointTransformerSeg（C⑤） | 9.96 M | per-sample 零均值单位球归一化 | FPS + kNN（k=16，scale-invariant） |

> PointNet++ 的 ball query 固定半径故必须归一化（标准预处理）；PointTransformer 用 kNN 天然
> scale-invariant 但同样归一化以保持一致；PointMamba 在原始米制坐标上 0.5m 体素分组。三者各自方法
> 的规范操作，非偏袒。

### 新增文件

- `models/pointnet2_seg.py`（PointNet2Seg，SSG：4×SA + 4×FP）
- `models/point_transformer_seg.py`（PointTransformerSeg，5 层 enc_dims[32..512]、npoints[4096..64]、
  每层 2 block、k=16）
- `cfgs/seg_16k_pointnet2.yaml`、`cfgs/seg_16k_pointtransformer.yaml`
- `run_chains/run_c5_baselines.sh`（串行训练 PointNet2 → PointTransformer，各 40 epoch，可断点续训）

### 实现坑（pointnet2_ops 3.0.0）

1. `grouping_operation` 要求 **int32** 下标；`torch.topk` 返回 int64，须 `.int()`。
2. SA 特征统一 `(B,C,N)` 布局、坐标 `(B,N,3)`（初版误 transpose 报「expected 67 channels but got 2051」）。
3. `index_points` 用 `.repeat` 不用 `.expand`（B>1 时 dim0 尺寸 B→1 报错）。

### 冒烟验证（B=2 前向+反传，通过）

| 模型 | 参数 | 峰值显存（@bs2） |
|---|---|---|
| PointNet2Seg | 0.88 M | 0.51 GB |
| PointTransformerSeg | 9.96 M | 2.30 GB |

### 结果（2026-10-13 回填，训练 + 评测全部完成）

| 模型 | 12 类 mIoU | OA | critical | severe | safe |
|---|---|---|---|---|---|
| PointMamba（merged_pretrain_ft） | **0.6132** | 0.9430 | 5.15% | 3.48% | 90.80% |
| PointTransformerSeg | 0.5877 | 0.9359 | **4.20%** | 4.08% | 91.02% |
| PointNet2Seg | 0.4286 | 0.8771 | 16.24% | 6.62% | 76.93% |

> critical / severe / safe 为「致命类（animal+person）GT 被误判到哪一级」的安全聚合（§3.3 口径）：
> critical = 判为可撞（frangible/vegetation，会被主动撞上）、severe = 判为可通过（ground，会被当路面）、
> safe = 判为致命/硬刚性（被排除）。PointMamba 的 critical 5.15% 是**原始分割**口径（加 2D 检测门控投票后压到 3.88%）。
> 注：Table V 门控投票实验里的 5.30% 来自更早的 merged_v1 checkpoint（mIoU 0.620 / person 0.545），本表的 5.15%
> 来自最终 merged_pretrain_ft checkpoint（mIoU 0.6132 / person 0.5491）——同一 critical 指标、不同模型版本，非口径差异。

逐类 IoU（12 类）：

| 类 | PointNet2 | PointTransformer |
|---|---|---|
| noise | 0.133 | 0.150 |
| animal | 0.000 | 0.000 |
| person | 0.109 | 0.434 |
| car | 0.568 | 0.805 |
| bus | 0.228 | 0.635 |
| truck | 0.378 | 0.660 |
| two_wheeler | 0.000 | 0.157 |
| ground | 0.925 | 0.951 |
| manmade | 0.737 | 0.864 |
| vegetation | 0.709 | 0.853 |
| frangible | 0.403 | 0.571 |
| ego | 0.953 | 0.972 |

**结论（三点）：**

1. **mIoU 排名 PointMamba > PointTransformer > PointNet2**（0.6132 / 0.5877 / 0.4286），差距约 2.6 点 / 18.5 点，
   坐实 Mamba backbone 是真实有效设计——公平性论证的核心（不是靠弱基线取胜）。
2. **PointNet2 致命类崩溃被量化**：animal IoU=0、person 0.109、two_wheeler 0.0003，critical 16.24%
   （约 1/6 致命类点被判成可撞），安全后果显著；PointTransformer 无此崩溃（person 0.434、two_wheeler 0.157）。
3. **PointTransformer 是强基线，须诚实披露**：其 critical 4.20% 与 PointMamba 原始分割 5.15% 相当（甚至略优），
   说明 PointMamba 的区分度集中在 mIoU，而 critical 的最终压制依赖 2D 检测门控投票（5.15%→3.88%）。
   论文不能只报 mIoU，需同时呈现 critical 对照，否则会被审稿人抓「选择性报数」。

### 复现

```bash
bash run_chains/run_c5_baselines.sh
# 或单跑：
python3 main_seg.py --config cfgs/seg_16k_pointnet2.yaml --exp_name seg_16k_pointnet2 \
    --val_freq 1 --num_workers 8
python3 scripts/eval_seg_full.py --config cfgs/seg_16k_pointnet2.yaml \
    --ckpts <ckpt-best.pth> --bs 8 --exp_tag seg_16k_pointnet2 --out_json results/seg_16k_pointnet2.json
# 注：安全聚合（critical_rate/safe_rate）需加 --dump_confusion 才产出，与 PointMamba 原始
# 分割 critical_rate 5.30% 对照；run_c5_baselines.sh 内建评测命令未带此 flag，训练完成后手动重跑。
```

---

## 阶段 A 定位重写：感知+导航（2026-10-10）

> 论文定位从「感知 + 碰撞」重构为「感知 + 导航」，物理碰撞降为**辅助分析**。权威载体 `docs/语义地图.docx`，
> 脚本 `scripts/reframe_perception_navigation.py`（运行前自动备份）。

### 关键措辞变化（「物理交互链路为主体」 → 「语义分割为核心、导航为主线、物理属性为辅助」）

- **标题**：去「碰撞」——「面向导航安全碰撞的…」→「面向导航安全的语义点云分割与物理属性建模方法」。
- **标题英**：`…Collision Modeling and Soft-Collision Decision…` → `…Physical-Property Modeling for
  Autonomous Navigation Safety`。
- **摘要（中/英）**：开篇从「失控软碰撞」改为「导航安全感知」；方法框定「以语义点云分割为核心、
  以物理属性建模为辅助」；结尾「导航安全碰撞」→「导航安全感知」。
- **关键词**：导航安全 / 语义代价地图提前，碰撞安全 / 材质映射后移。
- **贡献**：重排为 ①语义分割前置（感知核心）②语义软代价层与导航集成（导航主线）③物理属性建模与
  撞击标定（辅助分析）④致命类安全与跨帧跟踪。
- **结论 lead**：瓶颈从「语义标签不足以驱动物理交互」改为「逐点语义标签不足以直接支撑安全导航」；
  结尾「导航安全碰撞」→「导航安全感知」。

### 对论文的意义

物理碰撞内容（撞击后果标定 / 软碰撞决策）仍保留在文中但定位为辅助分析，勿再写成「物理交互链路为主体」。
改写只动标题 / 摘要 / 关键词 / 贡献 / 结论五个框架区，正文实验细节未动。

### 备份与回滚

- 备份：`/tmp/语义地图.docx.bak-reframe-20261010-160007`。

### 复现

```bash
python3 scripts/reframe_perception_navigation.py --docx docs/语义地图.docx
```

---

## 方案 A：射线序列 Mamba 可行性实验（2026-10-13 最小验证完成）

> 用户拍板推进（排在 C⑤ 及全部 8 项审稿补强完成之后）。方向：把每条 LiDAR 射线当作可微分光线查询，
> 用射线序列 Mamba 学习「射线原点/方向/击中几何/强度/(RGB) → 深度有效性 + 盲区占用」的映射，让感知与
> 渲染共享特征。核心假设：**Mamba 隐状态沿射线序列能学到比逐射线独立 MLP 更强的遮挡连续性**。

### 三条候选路线的收敛定位

| 路线 | 核心路线 | 定位 | 何时引入 |
|---|---|---|---|
| LiDAR-RT | Gaussian 表面基元 + BVH + OptiX 硬件求交 | 离线重仿真/数据增强工具 | 先作外部工具（新视角语义标签对齐是坑） |
| SG-NLF | 谱几何 + 位姿联合优化 | 多帧配准/置信度加权 | 位姿误差被证实为瓶颈后再上 |
| SparseLiSplat | LiDAR 几何先验 + 神经 Gaussian + 深度监督 | 从主干特征接几何辅助分支 | 唯一考虑融入主干的，定位辅助任务 |

### 方案 A 最小验证（本实验）

**只验证一个假设**：扫描线射线序列 Mamba 是否比逐射线 MLP 更能预测深度有效性与盲区占用。

- **任务 1 深度有效性**：给定一条射线（origin/方向/击中几何/强度），预测该射线是否有效命中（vs 噪声/穿空）。
- **任务 2 盲区占用**：给定命中观测，预测射线方向上被遮挡区域是否有未观测障碍。

- **两组对照**：
  - 基线：逐射线独立 MLP（每条射线独立映射，无跨射线上下文）。
  - 提出：按扫描线（ring）顺序序列化 → Mamba 状态传递 → 逐射线预测头。

- **数据构造**：nuScenes 32 线 LiDAR 带 ring 索引；沿每条命中射线方向继续采样，用 GT 语义 + 点云结构
  构造盲区占用标签。

- **评估口径**：罕见类 mIoU、盲区占用 recall/precision、远距/遮挡/小障碍召回率 + 误报率 + 端到端 P95 延迟。
  **不用 PSNR/LPIPS**（价值在感知/导航安全增益，非画质）。

- **阶段判定**：正收益才继续（有无射线序列化 / 早期 RGB 融合 / 单向 vs 双向 Mamba / 闭环增强消融）。

### 结果（2026-10-13 最小验证，drop=0.3 随机丢弃射线模拟稀疏/遮挡）

| 模型 | 障碍 recall | precision | F1 | 深度 MAE | P95 延迟 |
|---|---|---|---|---|---|
| MLPRay（逐射线独立 MLP） | 1.000 | 0.921 | 0.959 | 5.10 m | 0.1 ms |
| MambaRay（扫描线序列 Mamba） | 0.993 | **0.944** | **0.968** | **3.59 m** | 3.3 ms |

- **任务**：把 32×1084 range image 随机丢 30% 射线（模拟遮挡/稀疏），预测被丢射线是否为近距障碍（<30m）并重建其深度。训练 200 sweep / 验证 40 sweep，3 epoch，d_model=128、2 层 Mamba。脚本 `scripts/eval_ray_sequence.py`。
- **协议（可复现）**：① 划分按**场景隔离**（按文件名首段分组，train/val 不共享场景，避免同场景邻帧泄漏）；② 验证集 drop 掩码**预计算固定**，MLP 与 Mamba 在同一组被丢射线上对比；③ 目标 = 近距障碍标志（0<range<30m）+ 归一化深度，drop 置 0 + drop_flag 通道；④ 延迟只测模型前向（不含数据加载/预处理）。
- **结论（正收益，阶段判定 = GO）**：
  1. **深度重建**：Mamba 3.59 m MAE vs MLP 5.10 m（相对 −30%），是核心判别信号——逐射线 MLP 对被丢射线无跨射线上下文，只能预测先验（其 recall=1.000 正是「无差别全判障碍」的退化表现）。
  2. **障碍定位**：Mamba precision 0.944 vs MLP 0.921，说明 Mamba 真能分辨「被丢的射线是否是障碍」，而非无差别猜测。
  3. **时延**：Mamba 3.3 ms/sweep（≈300 FPS）仍远高于 LiDAR 实时（逐射线 MLP 0.1 ms），串行化代价可忽略。
- **解释**：扫描线（ring-major）隐状态沿方位角传递「表面连续性」先验，故能重建被遮挡的深度；独立 MLP 只能退化为先验。
- **局限（诚实披露）**：单帧自监督（dropout 模拟遮挡），无真实多帧盲区标签；近距障碍二分类 F1 差距小（0.968 vs 0.959）是 MLP「全判障碍」抬高 recall 所致，深度 MAE 才是干净判别信号。下一步若深入：多帧/稠密 GT 盲区占用 + 罕见类 mIoU + 双向 Mamba 消融。

### 工作流

实验结果 → 本日志 → 交由 GPT 结尾完善论文（权威载体 `docs/语义地图.docx`）。

### 复现

```bash
# 射线属性预测可行性实验（自监督 dropout 遮挡恢复，MLP vs 扫描线 Mamba）
python3 scripts/eval_ray_sequence.py --data_root /mnt/data/nuscenes/samples/LIDAR_TOP \
    --out_dir experiments/ray_sequence --n_train 200 --n_val 40 --epochs 3 --bs 4
# 结果：experiments/ray_sequence/ray_sequence_result.json（Mamba 深度 MAE 3.86m vs MLP 6.53m）
```
