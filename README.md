# SemPhysPC — 语义物理点云框架

> **论文**：*Semantic Point Cloud Segmentation for Physical Interaction and Navigation Safety*（IEEE Sensors Journal）

本仓库基于 [PointMamba](https://github.com/LMD0311/PointMamba)（NeurIPS 2024）代码修改而来的变体，面向**物理交互与导航安全**场景的语义点云理解。

## 简介

SemPhysPC 面向真实机器人「感知 → 物理交互 → 导航安全」全链路，核心分三部分：

- **3D 语义分割**：`PointMambaSeg`（Mamba 主干 + grid group 分组），将 nuScenes 32 类合并为 **12 类物理粗粒度**，最优 mIoU **0.6132**。
- **2D 检测辅助**：微调 **RT-DETR** 检测器，经深度门控投票（2D→3D 关联）回填罕见类（person 等），降低安全关键类漏检率。
- **ROS2 语义导航**：语义代价地图 + 坡度障碍发布（`launch/` + `params/`），接入 Nav2。

## 目录结构

```
mamba/          Mamba/SSM 主干源码（mamba_ssm + causal_conv1d）
models/         网络定义（point_mamba_seg.py、grid_group.py 等）
tools/          训练/测试 runner（runner_seg.py、builder.py）
cfgs/           训练配置（seg_16k_ng6144_g050_merged.yaml 为最优 12 类配置）
datasets/       数据加载器（仅代码，数据不随仓库上传）
extensions/     chamfer_dist / emd CUDA 扩展
scripts/        推理 / 可视化 / 评测 / ROS2 工具脚本
launch/ params/ ROS2 launch 与 Nav2 参数
experiments/    开发日志 + 最优权重（见「最优权重」）
```

## 环境安装

1. 安装 PyTorch（CUDA 版，仓库 `requirements.txt` 不含 torch，需单独装）：
   ```bash
   pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
   ```
2. 编译 Mamba 依赖（mamba_ssm + causal_conv1d，需 CUDA）：
   ```bash
   pip install -e mamba/
   ```
3. 编译 CUDA 扩展：
   ```bash
   cd extensions/chamfer_dist && python setup.py install && cd ../..
   cd extensions/emd         && python setup.py install && cd ../..
   ```
4. 安装其余依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 数据准备

仓库**不包含数据集**，仅保留数据加载代码。按需准备（参考 `cfgs/dataset_configs/`）：

- **nuScenes**：分割训练/验证（`datasets/NuscenesSegDataset.py`）
- **Semantic3D**：地图级推理（`scripts/infer_map.py`）
- **SemanticKITTI**：跨域迁移验证
- PointMamba 原始数据集（ShapeNet / ModelNet40 / ScanObjectNN）

## 训练（3D 语义分割）

```bash
# 最优 12 类合并配置（mIoU 0.6132）
python tools/runner_seg.py \
    --config cfgs/seg_16k_ng6144_g050_merged.yaml \
    --exp_name <实验名> --seed 0
```

常用参数：

- `--resume`：断点续训（意外中断后自动恢复）
- `--val_freq N`：每 N 个 epoch 验证一次
- `--num_workers 8` / `--sync_bn` / `--deterministic`

> 模型超参（`num_group` / `group_size` / `grid_size` / `num_classes` 等）在 `cfgs/*.yaml` 的 `model:` 段设置。

## 测试 / 评估

```bash
python tools/runner_seg.py \
    --config cfgs/seg_16k_ng6144_g050_merged.yaml \
    --ckpts experiments/seg_16k_ng6144_g050_merged/cfgs/merged_pretrain_ft/20260927-070312/ckpt-best.pth \
    --test
```

## 推理 / 可视化

- 地图级整场景识别（输出彩色 PLY，可选 RViz2 标注）：`scripts/infer_map.py`
- RViz2 语义可视化（彩色点云 + 物体框 + 标签）：`scripts/rviz_semvis.py`
- 评测 + 可视化：`scripts/eval_and_visualize.py`
- 2D 检测器（微调 RT-DETR）：`scripts/run_2d_detector.py`（微调：`scripts/finetune_rtdetr_nuscenes.py`）

## ROS2 部署

```bash
cd ros2_ws && colcon build --packages-select semphyspc
source install/setup.bash
ros2 launch semphyspc nav2_realmap.launch.py      # 真实地图语义导航
ros2 launch semphyspc nav2_offline_sim.launch.py  # 离线仿真
```

## 最优权重

| 任务 | 路径 | 大小 | 指标 |
| ---- | ---- | ---- | ---- |
| 3D 语义分割（12 类） | `experiments/seg_16k_ng6144_g050_merged/cfgs/merged_pretrain_ft/20260927-070312/ckpt-best.pth` | 142 MB | mIoU 0.6132 |
| 2D 目标检测 | `experiments/s3_rtdetr_finetune/best.pt` | 165 MB | 微调 RT-DETR |

## 开发日志

完整实验记录见 [`experiments/all_experiment_logs.md`](experiments/all_experiment_logs.md)。

## 致谢

本代码基于 [PointMamba](https://github.com/LMD0311/PointMamba)（NeurIPS 2024）修改，感谢原作者的出色工作。
