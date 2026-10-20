# 论文实测：Unity 场景加载 + 位置相关破坏测试（Windows）

本包用于在 **Windows + Unity** 上复现论文第 3.8 节「游戏物理导出」与第 4.4 节「位置相关破坏验证」。

## 包含内容

```
UnityPaperTest/
├── Assets/
│   ├── GameSceneLoader.cs     # 读取 game_scene.json，生成 BoxCollider + Rigidbody + PhysicMaterial + DestructibleObject
│   ├── DestructibleObject.cs  # 位置相关破坏组件（复刻 utils/damage.py）
│   └── ImpactTester.cs        # 鼠标点击撞击测试（可选）
├── StreamingAssets/
│   └── game_scene.json        # 66 个实例的场景数据（collider/rigidbody/damage 参数）
├── objects.csv                # Unreal DataTable 用（66 物体）
├── parts.csv                  # Unreal DataTable 用（99 部位）
└── README_Windows.md
```

`game_scene.json` 由 `scripts/export_game_scene.py` 从 nuScenes 16k 帧 `000260ba…` 导出，
含 44 建筑 + 11 植被 + 5 轿车 + 3 卡车 + 2 自车 + 1 拖车。

## 步骤

### 1. 安装 Unity（国内账号）

- 打开 Unity 中国官网 `https://unity.cn/`，注册账号（职业选「个人开发者」，别选学生）。
- 下载并安装「团结 Hub」（或国际版 Unity Hub）。
- 在 Hub 里「管理许可证 → 添加许可证 → 获取免费的个人许可证」。
- 「安装编辑器」选一个 **3D** 版本（团结引擎或 Unity 6 / 2022.3 LTS 均可，本包只用标准 C# + JsonUtility + PhysicMaterial，无特殊 API）。

### 2. 新建项目并导入脚本

1. Hub 里新建 **3D（URP 或内置管线皆可）** 项目。
2. 把 `Assets/` 下 3 个 `.cs` 文件复制到项目 `Assets/` 目录。
3. 在 `Assets/` 下新建文件夹 `StreamingAssets`，把 `StreamingAssets/game_scene.json` 放进去。
4. 回到 Unity，等待脚本编译（Console 无报错）。

### 3. 挂载加载器并运行

1. 在 Hierarchy 里新建一个空物体（GameObject → Create Empty）。
2. 给它挂 `GameSceneLoader` 组件（jsonPath 保持默认 `game_scene.json`）。
3. 给 Main Camera（或任意物体）挂 `ImpactTester` 组件，energy 保持默认 270。
4. 点 **Play** —— 自动生成 66 个物体。

### 4. 观察与测试

- **场景**：能看到建筑/植被/车辆/自车，每个物体带 BoxCollider（Scene 视图绿色线框）。
- **坐标约定**：nuScenes(x前,y左,z上) → Unity(x右,y上,z前)，脚本已自动转换。
- **位置相关破坏**：Play 后鼠标左键**点车身的不同位置**，看 Console 输出：
  - 撞**弱部位**（车头/车尾薄板）→ `状态=destroyed, 破碎比例≈0.5`
  - 撞**强部位**（车身厚实处）→ `状态=intact, 破碎比例=0.00`
  - 对应论文 4.4 表：拖车最弱部位 strength 0.225 / 270J → destroyed；最强部位 0.900 / 270J → intact。

> 提示：`DestructibleObject` 的 `OnPartBroken` 事件可挂自定义回调（生成碎块、换材质、播特效），
> 用于把「破碎」可视化出来。

## 扩展

- **重新导出场景**（在 Linux 端，用模型推理而非 GT 标签）：
  ```bash
  python scripts/export_game_scene.py --input cloud.ply \
      --config cfgs/seg_nuscenes_full.yaml --ckpts experiments/seg_nuscenes_full/cfgs/seg_full_v1/20260814-140501/ckpt-best.pth \
      --output_dir out_game --emit_csv
  ```
- **Unreal**：直接把 `objects.csv` / `parts.csv` 导入 DataTable 即可。
