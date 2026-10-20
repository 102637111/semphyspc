# 课程实验：用 Transformer 做点云分类（PCT on ModelNet40）

一份**纯 PyTorch** 的点云识别入门实验。不依赖任何稀疏卷积库（torchsparse / spconv /
MinkowskiEngine 都不需要），**Windows / Linux / macOS、有显卡或纯 CPU 都能跑**，适合作为
《三维视觉 / 深度学习》课程的实验作业。

## 1. 这是什么

模型是 **PCT（Point Cloud Transformer）**，思路很直白：

```
输入点云 (1024, 3)  →  把每个点当做一个"词"，用 self-attention 让点之间互相"看"
                   →  全局 max+mean 池化得到整个物体的特征
                   →  线性分类头输出 40 个类别之一
```

和经典的 PointNet++（层级 MLP）相比，PCT 是标准的 **Transformer 范式**（注意力 + 池化），
用来回答一个问题：**去掉卷积、只靠注意力，能不能识别三维物体？**

数据集是 **ModelNet40**（点云分类的「Hello World」，40 类常见物体：飞机/椅子/桌子/杯子…，
12311 个 CAD 模型采样成点云）。

## 2. 环境（Windows 也能跑）

```bash
pip install torch numpy        # 就这两个；有 NVIDIA 卡自动用 GPU，没有则用 CPU
```

- 有显卡（推荐）：`pip install torch`（PyTorch 官方 Windows 版自带 CUDA）。
- 纯 CPU：同样的命令，只是训练慢一些（本实验单卡只占 **0.08 GiB 显存**，笔记本集成显卡/CPU 也能跑）。

## 3. 数据

本仓库已经带好了：`data/ModelNet/modelnet40_normal_resampled/`（40 个类目录，每个 `.txt` 是一个
物体的 1024 点 xyz）。若要从零下载：ModelNet40 官网或 `modelnet40_normal_resampled.zip`，
解压到 `data/ModelNet/` 下即可。

首次运行会把全部点云读进内存并缓存成 `.npz`（加速后续 epoch）。

## 4. 怎么跑

```bash
# 训练（默认 120 轮，约 8 分钟/单卡，CPU 则更久）
python3 scripts/pct_modelnet40.py

# 快速试跑（3 轮，验证环境是否配好）
python3 scripts/pct_modelnet40.py --epochs 3 --batch 16

# 调参
python3 scripts/pct_modelnet40.py --epochs 200 --batch 64 --lr 3e-3
```

每 10 轮在测试集上评一次，`test_acc` 会从 ~50% 一路涨到 **~90%**。训练完保存
`results/pct_modelnet40_best.pth`。

## 5. 预期结果与作业方向

- 基线：约 **90%** 分类准确率（120 轮）。
- 可改的实验变量（都是几行的事）：
  1. **点数**：`NUM_POINTS` 从 1024 改成 256/4096，看精度怎么变（信息量 vs 计算量）；
  2. **注意力层数/头数**：`PCT(depth=…)` 改深改浅，看过拟合与精度；
  3. **增广开关**：把 `augment()` 关掉，看旋转/缩放/抖动带来多少增益；
  4. **对比 PointNet++ / PointNet**：换成只做 max-pool 的 MLP（去掉 attention 块），
     观察「没有注意力」时精度掉多少——这是理解 transformer 贡献的关键对照。

## 6. 关键代码位置

| 文件/函数 | 作用 |
|-----------|------|
| `SA` | 单头→多头 self-attention（transformer 的核心） |
| `Block` | attention + LayerNorm + MLP 的残差块 |
| `PCT` | 整体：embedding → 4 层注意力 → 池化 → 分类头 |
| `augment()` | 绕 z 轴随机旋转 + 缩放 + 抖动（标准点云增广） |
| `load_off_txt()` | 读单个 `.txt` 点云并中心化/归一化 |
