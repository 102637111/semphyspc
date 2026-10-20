#!/usr/bin/env python3
"""把主论文 paper_draft.md（SemPhysPC 物理交互）书写成 docx 文档。

用法（在项目根 /home/saw/ros2_ws/src/semphyspc/ 下）：
    python3 scripts/build_semphyspc_docx.py

输出：docs/paper_draft_semphyspc.docx
相比 paper_draft.md，本脚本额外完成：
  1. 嵌入图 1（fig1_pipeline.png）、图 2（fig2_architecture.png）；
  2. 深化 rare thing 类「类别不平衡/长尾」分析（§4.2 + §5 未来工作）；
  3. 同步正向结果：num_group 64→256 规模化（§4.2）+ 罕见类 2D/3D 深度门控投票融合（§4.13）；
  4. 参考文献 [1]–[8] 归一化为 IEEEtran 格式。
"""
import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG1 = os.path.join(ROOT, "paper", "figures", "fig1_pipeline.png")
FIG2 = os.path.join(ROOT, "paper", "figures", "fig2_architecture.png")
_FX = os.path.join(ROOT, "paper", "figures", "effect")
FIG3_RGB = os.path.join(_FX, "17bd604f6828423680434cd2e522180a_rgb.png")
FIG3_SEM = os.path.join(_FX, "17bd604f6828423680434cd2e522180a_semantic.png")
FIG3_SAF = os.path.join(_FX, "17bd604f6828423680434cd2e522180a_safety.png")
LEG_SEM = os.path.join(_FX, "legend_semantic.png")
LEG_SAF = os.path.join(_FX, "legend_safety.png")
OUT = os.path.join(ROOT, "docs", "paper_draft_semphyspc.docx")

EAST = "宋体"
WEST = "Times New Roman"

doc = Document()

# ---- 页面：A4 + 适中边距 ----
for s in doc.sections:
    s.page_width = Inches(8.27)
    s.page_height = Inches(11.69)
    s.left_margin = Inches(1.0)
    s.right_margin = Inches(1.0)
    s.top_margin = Inches(1.0)
    s.bottom_margin = Inches(1.0)

# ---- 全局默认字体 ----
normal = doc.styles["Normal"]
normal.font.name = WEST
normal.font.size = Pt(10.5)
normal._element.rPr.rFonts.set(qn("w:eastAsia"), EAST)


def _set_run(run, size=10.5, bold=None, italic=None, color=None, east=EAST, west=WEST):
    run.font.name = west
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor(*color)
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:eastAsia"), east)


def body(text, size=10.5, bold=False, align=None, space_after=6, indent=False):
    """正文段落；支持 **加粗** 简单标记。"""
    p = doc.add_paragraph()
    parts = text.split("**")
    for i, part in enumerate(parts):
        if part == "":
            continue
        run = p.add_run(part)
        _set_run(run, size=size, bold=(bold or (i % 2 == 1)))
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    return p


def heading(text, level=1):
    sizes = {1: 14, 2: 12, 3: 11}
    p = doc.add_paragraph()
    run = p.add_run(text)
    _set_run(run, size=sizes.get(level, 11), bold=True, east="黑体")
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(6)
    return p


def caption(text, size=9):
    p = doc.add_paragraph()
    run = p.add_run(text)
    _set_run(run, size=size, bold=True, east="黑体")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    return p


def figure_row(items, width=2.0):
    """无边框表格并排多张图，每列「图 + (a)(b)… 标签」。items: [(img_path, label), ...]"""
    t = doc.add_table(rows=1, cols=len(items))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (img, label) in enumerate(items):
        cell = t.rows[0].cells[i]
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if os.path.exists(img):
            p.add_run().add_picture(img, width=Inches(width))
        else:
            p.add_run("[" + label + " 占位]")
        lp = cell.add_paragraph()
        lp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        lrun = lp.add_run(label)
        _set_run(lrun, size=9)
    return t


def table(headers, rows, col_widths=None, font_size=9, header_size=9):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(h)
        _set_run(run, size=header_size, bold=True, east="黑体")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            _set_run(run, size=font_size)
    if col_widths:
        for i, w in enumerate(col_widths):
            for r in t.rows:
                r.cells[i].width = Inches(w)
    return t


# ======================================================================
# 标题
# ======================================================================
p = doc.add_paragraph()
run = p.add_run("面向物理交互的语义点云分割与碰撞/损毁建模方法")
_set_run(run, size=16, bold=True, east="黑体")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER

p = doc.add_paragraph()
run = p.add_run("Semantic Point-Cloud Segmentation towards Physical Interaction: "
                "Collision and Destruction Modeling for Autonomous Navigation Safety and Game Map Building")
_set_run(run, size=12, bold=True, italic=True)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER

body("初稿（Draft v1）　目标期刊：**IEEE Sensors Journal**（首选）；中文/英文 EI 或 SCI Q3 为导师底线。"
     "注意 Sensors 是传感器导向的期刊，审稿人关注传感模态、多传感器融合与应用验证——因此「失控软碰撞」这类应用主张的可信度"
     "（检测召回、模态覆盖、决策延迟）是重点审查对象。主线：分割为前置，物理交互/软碰撞链路为主体。"
     "文中图 1–3 已嵌入；所有数值均来自已运行的实验与 results/ 存档。", size=9, align="center", space_after=10)

# ======================================================================
# 摘要 / 关键词
# ======================================================================
heading("摘要", 1)
body("自动驾驶车辆在失控等紧急场景下需要选择\"非致命、可变形/可破坏\"的物体（灌木、交通锥、护栏等）"
     "进行主动软碰撞以实现安全减速，而避免撞击行人、动物、车辆与墙体等致命或刚性目标；另一方面，"
     "基于实地扫描的三维点云地图本身不具备任何碰撞与损毁逻辑，无法直接用于游戏场景建模。"
     "二者的共同瓶颈在于：**仅得到逐点语义类别还不够，还必须将语义类别转化为物理交互属性"
     "（碰撞安全等级、质量、摩擦、恢复系数、可破坏性与损毁响应）**。针对这一问题，本文提出一套从语义分割"
     "到物理交互的端到端方法：以基于 RGB 与 XYZ 早期融合的 PointMambaSeg 语义分割为前置，构建 32 类语义→"
     "物理材质映射库；在此基础上实现实例级聚类、**位置相关破坏模型**（材料强度 × 几何结构强度 × 撞击局部化）、"
     "导航软碰撞目标决策、跨帧实例跟踪，以及面向 Unity/Unreal/ROS2 的游戏物理与导航系统导出。"
     "实验在 nuScenes 公开数据集上验证了分割前置：把 Mamba 的 token 数（num_group）由 64 提升到 1024，mIoU 由 0.2323 提升到 0.3923；"
     "受控扫描进一步表明 token 预算在 1024 处进入饱和（1024→2048 的增益仅为 256→1024 的 5.4%，却要 2 倍训练开销）、体素边长 0.5 m "
     "是驻点，据此定位 Mamba 的规模化瓶颈为 **token 数而非输入点数**——在 token 数同为 1024 时把点数翻倍到 32768 反而使 mIoU 下降 "
     "0.0226；严格的 RGB 消融（结构、数据、轮数全部相同，仅输入通道 3→6）表明 RGB 早期融合带来 **+0.0331 mIoU（相对 +9.2%）**，"
     "且增益集中在几何相似、须靠外观区分的类（警车 +0.330、工程车 +0.164、交通锥 +0.118、施工人员 +0.105），说明颜色不编码类别身份"
     "而是在几何歧义处提供消歧；并以受控实验把罕见类失效的成因分解为「分组粒度为主因、采样密度仅为门槛、标注量决定其余恒零类」的三元分工"
     "（16k 点固定、仅换分组方式即可使罕见类 mIoU 提升 3.17 倍），进而用地面参考的几何测量把恒零类进一步区分为"
     "「几何不可分（警员与普通行人几何完全重叠，只能靠外部模态）」与「几何可分但监督不足（儿童/婴儿车，可经增加监督改善）」两种情况；"
     "针对分割对"
     "长尾罕见类（行人/自行车/摩托车等）近乎全灭的缺陷，研究 6 相机深度门控的 2D/3D 投票融合，并严格区分「机制上界」与「可部署"
     "效果」：真值框 oracle 设置下罕见类 mIoU 上界为 0.0995→0.5060（全类 0.3090→0.4457），可部署设置改用 COCO 预训练检测器的零样本"
     "预测则为 0.0995→0.1519（全类 0.3199），仅兑现上界增益的 12.9%——而关联精度分析表明该落差源于检测器召回（0.18 vs oracle "
     "0.861）而非融合机制；进一步用 nuScenes train split 的 3D 真值框投影标签微调检测器以补齐召回（框级关注类平均召回 0.1160→0.6521，"
     "barrier / trafficcone 由严格 0 升至 0.75 / 0.67），发现端到端收益须以**注入标签的点级精度**为前提——默认操作点下罕见类 mIoU 反而"
     "下降（0.1519→0.1427），而在提高操作点并采用单目高度先验门控后可反超至 **0.1731（相对 +13.9%）**（按类自适应 θ 可进一步升至 "
     "0.1815），并指出融合规则对所有低置信点**无条件覆盖、缺少逐点可信度判断**是残余瓶颈；级联实例分割经类别门控把 thing 类语义精度"
     "提升至 0.724，并通过"
     "置信度门控实现「可拒识」的可信决策（拒识 50% 最不确定实例后精度 +14.5 点）；并对撞击后果做了闭合形式标定"
     "（自车 1500 kg、30 m/s），定量指出原软碰撞候选集的减速上限仅 0.824 m/s（初速的 2.75%），据此把决策扩展为"
     "「正碰软目标 + 刮蹭硬面」双模式并纳入锚定类（合成几何下刮蹭减速 5.150 m/s、与手算误差 0.00%）；同时展示了同一撞击能量"
     "作用于物体不同部位得到不同破坏结果的真实性，以及多航向下正确选择软碰撞目标的决策行为。"
     "该方法为\"导航安全碰撞\"与\"游戏实景建图\"提供了一条可落地的技术路径。", align="justify")

body("**关键词**：点云语义分割；Mamba；物理材质映射；碰撞安全；损毁建模；导航软碰撞；游戏物理导出", space_after=10)

heading("Abstract", 1)
body("In loss-of-control scenarios, autonomous vehicles should actively select non-lethal, deformable or frangible "
     "objects (e.g., bushes, traffic cones, guardrails) for soft collision to decelerate safely, while avoiding lethal "
     "or rigid targets such as pedestrians, animals, vehicles and walls. Meanwhile, real-scanned 3D point-cloud maps "
     "carry no collision or destruction logic and cannot be used directly for game scene building. The shared bottleneck "
     "is that **per-point semantic labels alone are insufficient; the semantics must be translated into physical "
     "interaction properties (collision-safety level, mass, friction, restitution, destructibility and damage response)**. "
     "We propose an end-to-end method from semantic segmentation to physical interaction: (1) a PointMambaSeg segmentation "
     "front-end with early RGB+XYZ fusion; (2) a 32-class semantic-to-physical-material mapping library; (3) instance-level "
     "clustering and a **position-dependent destruction model** (material strength × geometric structural strength × impact "
     "localization); (4) a soft-collision target decision module for navigation; (5) cross-frame instance tracking; and "
     "(6) game-physics and navigation system export for Unity/Unreal/ROS2. Experiments on the nuScenes benchmark validate "
     "the segmentation front-end: raising the Mamba token count (num_group) from 64 to 1024 improves mIoU from 0.2323 to "
     "0.3923; a controlled hyper-parameter sweep shows the token budget to saturate at 1024 (1024→2048 adds only 5.4% of "
     "the 256→1024 gain while doubling the training cost) and a voxel size of 0.5 m to be the interior optimum, locating "
     "the scaling bottleneck in token count rather than input resolution — at a fixed token count of 1024, doubling the "
     "input from 16,384 to 32,768 points actually lowers mIoU by 0.0226; a strictly paired ablation (identical "
     "architecture, data and schedule, with only the input channels changing from 3 to 6) shows the early RGB+XYZ fusion "
     "to contribute +0.0331 mIoU (+9.2% relative), with the gain concentrated on classes that are geometrically similar "
     "and separable only by appearance (police vehicle +0.330, construction vehicle +0.164, traffic cone +0.118, "
     "construction worker +0.105) — colour does not encode class identity, but disambiguates where geometry is ambiguous; "
     "a controlled study further decomposes the rare-class failure "
     "into three factors, finding grouping granularity to be the dominant cause (at a fixed 16,384 points, changing only "
     "the grouping scheme raises rare-class mIoU by 3.17x), sampling "
     "density to act merely as a threshold, and annotation volume to govern the remaining zero-IoU classes, which a "
     "ground-referenced geometric measurement then separates into geometrically inseparable (police officers, whose height "
     "and footprint overlap those of ordinary pedestrians item by item, thus requiring an external modality) versus "
     "geometrically separable but under-supervised (children and strollers, improvable by added supervision); to address the "
     "near-total failure on long-tail rare classes (pedestrians, bicycles, motorcycles, etc.), we study a 6-camera "
     "depth-gated 2D/3D voting fusion and separate its mechanism ceiling from its deployable performance: under an oracle "
     "setting with ground-truth 2D boxes the rare-class mIoU goes from 0.0995 to 0.5060 (full mIoU 0.3090 to 0.4457), "
     "while the deployable setting, using zero-shot predictions of a COCO-pretrained detector, reaches 0.1519 (full mIoU "
     "0.3199) and thus realizes only 12.9% of the oracle gain; an association-precision analysis attributes that gap to "
     "detector recall (0.18 vs 0.861 for oracle) rather than to the fusion mechanism; fine-tuning the detector on 2D labels "
     "projected from the nuScenes train split lifts frame-level recall on the classes of interest from 0.1160 to 0.6521 and "
     "raises barrier / trafficcone from an exact zero to 0.75 / 0.67, yet the end-to-end gain materialises only when the "
     "point-level precision of the injected labels rises with it — at the default operating point the rare-class mIoU drops "
     "instead (0.1519 to 0.1427), whereas raising the operating point and using the monocular height-prior gate surpasses "
     "the zero-shot setting at 0.1731 (+13.9% relative), with a per-class adaptive θ further lifting it to 0.1815; we "
     "identify the unconditional overwrite of every low-confidence point, which carries no per-point trust judgement, as "
     "the residual bottleneck; the cascaded "
     "instance segmentation reaches 0.724 thing-class accuracy via class gating, and a confidence-gating module enables "
     "rejectable, trustworthy decisions (+14.5 points after rejecting the 50% most uncertain instances); we further "
     "calibrate the impact consequences in closed form (1500 kg ego at 30 m/s), showing quantitatively that the original "
     "soft-collision candidate set caps deceleration at only 0.824 m/s (2.75% of initial speed); based on this we extend "
     "the decision to a dual-mode scheme (head-on against soft targets + glancing scrape along hard surfaces) that also "
     "admits anchored classes (scrape deceleration 5.150 m/s in synthetic geometry, 0.00% error against the hand "
     "calculation); we further demonstrate position-dependent destruction and heading-dependent soft-collision decisions. "
     "The method provides a practical path for both navigation safety collision and game map building.", align="justify")

body("**Keywords**: point-cloud semantic segmentation; Mamba; physical material mapping; collision safety; destruction "
     "modeling; navigation soft collision; game physics export", space_after=10)

# ======================================================================
# 1 引言
# ======================================================================
heading("1 引言", 1)
body("三维点云语义分割在自动驾驶与机器人领域取得了显著进展，但其价值往往止步于\"识别出是什么\"。"
     "在实际系统中有两类典型的工程需求，恰恰要求分割结果能够**进一步驱动物理交互**：", indent=True)
body("**（1）导航失控安全碰撞。** 车辆在制动失效、路面打滑等失控场景下，需要主动选择前进方向上的实体进行"
     "\"软碰撞\"以减速，例如刮蹭护栏、撞击灌木或交通锥，而必须避免撞击行人、动物、车辆与墙体等致命或刚性目标。"
     "这一决策需要同时回答两个问题：*前方有什么？*（语义）以及 *撞上它会怎样？*（物理代价）。", indent=True)
body("**（2）游戏实景建图。** 基于激光雷达实地扫描得到的高保真点云地图，若直接作为游戏地图使用，既没有碰撞体，"
     "也没有损毁逻辑，无法支撑\"撞碎、刮蹭、穿越\"等真实交互。传统做法依赖美术人员手工为每个物体配置碰撞与材质参数，"
     "费时且难以迁移到新场景。", indent=True)
body("二者的共同瓶颈可概括为：**语义标签 ≠ 物理交互属性**。现有语义分割研究以提升 mIoU 为目标，输出逐点类别标签后"
     "即告完成，缺乏从类别到\"碰撞安全等级 / 质量 / 摩擦 / 恢复系数 / 可破坏性 / 损毁响应\"的映射；而游戏与仿真中的"
     "破坏模型（如 Voronoi 碎裂、有限元）又往往与真实场景的语义脱节，无法从点云直接构建。", indent=True)
body("为此，本文提出一条**以语义分割为前置、以物理交互链路为主体**的完整方法，命名为 **SemPhysPC**"
     "（Semantic Physical Point Cloud，语义物理点云框架），主要贡献如下：", indent=True)
body("1. **语义→物理材质映射库**：将 nuScenes 32 个语义类别映射到 5 级碰撞安全等级与 13 类物理材质，为每个类别定义"
     "密度、摩擦、恢复系数、可破坏性、致命性代价与损毁响应（none/deform/break/shatter）。", indent=True)
body("2. **位置相关破坏模型**：对每个实例进行子部位分解，定义\"材料强度 × 几何结构强度\"的每部位强度，并令撞击只作用"
     "于撞击点附近的部位，从而实现\"撞不同位置、坏不同部位\"的真实损毁逻辑。", indent=True)
body("3. **导航软碰撞决策**：在前向视锥内，按\"致命性代价最低\"的原则从可破坏/可变形类别中自动选择软碰撞减速目标。", indent=True)
body("4. **跨帧实例跟踪与系统导出**：基于匈牙利匹配实现跨帧稳定实例 ID 与静态点云融合，并提供 Unity C#、Unreal DataTable "
     "与 ROS2 节点的工程化导出。", indent=True)
body("实验部分在 nuScenes 公开数据集上验证了分割前置（并如实报告了\"升点数\"消融的负结果），以及物理映射、破坏模型、"
     "软碰撞决策与跟踪的可行性。", indent=True)

# ======================================================================
# 2 相关工作
# ======================================================================
heading("2 相关工作", 1)
body("**2.1 点云语义分割。** PointNet[1] 开创了直接处理点云的范式，PointNet++[2] 引入层级采样与局部聚合；Transformer "
     "架构（Point Transformer[3]）进一步利用注意力机制建模远距离依赖。近期，状态空间模型（Mamba）[4] 凭借线性复杂度在长序列"
     "建模上表现出潜力，PointMamba[5] 通过希尔伯特曲线序列化与选择性状态空间块处理点云。本文采用其语义分割变体（PointMambaSeg），"
     "并在输入端采用 RGB+XYZ 的 6 通道早期融合以引入颜色线索。", indent=True)
body("**2.2 语义→材质/属性映射。** 已有的点云材质估计工作多聚焦于视觉材质（PBR 贴图、反射率）[6]，服务于渲染；而面向物理"
     "交互的类别→属性映射（碰撞安全、质量、摩擦、可破坏性）尚缺乏系统性的定义与工程化实现，这正是本文的核心。", indent=True)
body("**2.3 游戏破坏与损毁建模。** 经典方法包括 Voronoi 预碎裂、有限元（FEM）与网格代理[7]，精度高但依赖精细网格与离线烘焙，"
     "难以从稀疏点云直接构建。本文采用轻量级\"子部位 + 强度状态机\"方案，以可解释的强度公式近似真实破坏行为。", indent=True)
body("**2.4 导航安全与紧急避障。** 现有紧急避障多以几何/动力学约束为主（如 MPC 安全边界、可达集）[8]，将障碍统一视为不可穿越"
     "刚体。本文引入语义级\"致命性代价\"，使规划器在失控时可优先选择低代价的软碰撞目标，与几何方法互补。", indent=True)

# ======================================================================
# 3 方法
# ======================================================================
heading("3 方法", 1)
heading("3.1 整体流水线", 2)
body("流水线分为**前置（分割）**与**主体（物理交互链路）**两阶段：", indent=True)
body("LiDAR + 相机 RGB ──► ① PointMambaSeg（RGB+XYZ 早期融合，32 类）→ ② 语义→物理材质映射 → ③ 实例级聚类（DBSCAN，"
     "thing/stuff 划分）→ ④ 破坏模型 / ⑤ 软碰撞决策 / ⑥ 跨帧跟踪 → 三条下游（游戏物理导出 Unity/Unreal、导航 ROS2 节点、"
     "地图级融合 fused_map.npz）。", indent=True)
if os.path.exists(FIG1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(FIG1, width=Inches(6.0))
    caption("图 1　SemPhysPC 整体流水线：语义分割为前置，语义→物理属性映射为核心跃迁，"
            "驱动破坏模型、软碰撞决策与跨帧跟踪三条下游。")
else:
    body("[图1 占位：整体流水线图]", align="center")

heading("3.2 分割前置：PointMambaSeg（RGB+XYZ 早期融合）", 2)
body("在输入端将点云 XYZ 与相机投影得到的 RGB 颜色在通道维拼接，形成 6 通道输入，经过非层级编码器与 Mamba 块提取特征，"
     "再通过 KNN 上采样解码器恢复逐点语义。关键设计：Mamba 的 token 序列长度由 num_group 决定（FPS 采样 num_group 个中心 + "
     "KNN 32 邻域聚合），与输入点数无关；token 数是 Mamba 序列建模的信息容量上限，是本文定位到的规模化瓶颈（§4.2）。"
     "默认配置 num_group=64，本文主配置 num_group=1024（饱和点，见表 2a）。模型参数量约 12.34M。", indent=True)
body("**为什么把 RGB 放在早期融合而非后处理**：颜色本身几乎不携带类别身份（§4.2.2），它的作用是在**几何相似**处提供消歧——警车与"
     "普通轿车、工程车与货车、施工人员与行人在 LiDAR 几何上是同一类形状，只能靠涂装与反光背心区分。这类消歧信号必须在 token 化"
     "**之前**进入，才能影响哪些点落进同一个 token；若放在分组之后，几何相似的点已经混在一起，颜色再无从区分。§4.2.2 的严格配对"
     "消融显示该设计净贡献 +0.0331 mIoU，且增益恰好集中在上述类别上。", indent=True)
if os.path.exists(FIG2):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(FIG2, width=Inches(6.0))
    caption("图 2　PointMambaSeg 网络结构：(a) 6 通道融合 → Mamba×12 → 上采样 → 语义 + offset 头；"
            "(b) PointMambaScan 物体级判类。")
else:
    body("[图2 占位：PointMambaSeg 网络结构图]", align="center")

heading("3.3 语义→物理材质映射库", 2)
body("这是本文的核心贡献之一。将 32 个语义类别映射到 **5 级碰撞安全等级** 与 **13 类物理材质**：", indent=True)
body("**碰撞安全等级**（导航决策核心）：", indent=True)
table(
    ["等级", "含义", "撞击代价 lethality", "典型类别"],
    [
        ["lethal 致命", "行人/动物，严禁撞击", "1.00", "pedestrian.*, animal"],
        ["hard 硬刚性", "车/墙/建筑，撞击严重受损", "0.75–0.90", "car/truck/bus, manmade, bicycle"],
        ["frangible 可破坏", "护栏/交通锥，可主动撞击减速", "0.10–0.30", "barrier, trafficcone, pushable"],
        ["deformable 可变形", "灌木，可安全穿越减速", "0.20", "vegetation"],
        ["passable 可通过", "路面/人行道/地形，正常行驶", "0.00–0.05", "driveable_surface, sidewalk, terrain"],
    ],
    col_widths=[1.4, 2.2, 1.5, 2.2],
)
body("**物理材质库**（物理引擎参数，表 1）：", indent=True, space_after=3)
caption("表 1　物理材质库全表（密度/摩擦/恢复系数/刚性/可破坏性 → 典型语义）")
table(
    ["物理材质", "密度 kg/m³", "摩擦", "恢复系数", "刚性", "可破坏性", "典型语义"],
    [
        ["metal 金属", "7850", "0.50", "0.10", "rigid", "0.05", "车/公交/拖车"],
        ["glass 玻璃", "2500", "0.40", "0.05", "frangible", "0.95", "玻璃护栏"],
        ["concrete 混凝土", "2400", "0.70", "0.05", "rigid", "0.00", "建筑/人行道"],
        ["rock 岩石", "2600", "0.70", "0.05", "rigid", "0.00", "静态岩石"],
        ["asphalt 沥青", "2300", "0.80", "0.05", "rigid", "0.00", "可行驶路面"],
        ["plastic 塑料", "1000", "0.40", "0.40", "frangible", "0.50", "护栏/推拉物"],
        ["rubber 橡胶", "1100", "0.90", "0.70", "deformable", "0.30", "交通锥/轮胎"],
        ["wood 木材", "700", "0.50", "0.20", "frangible", "0.40", "木质结构"],
        ["vegetation 植被", "300", "0.50", "0.05", "deformable", "0.80", "灌木/树"],
        ["soil 土壤", "1500", "0.60", "0.00", "deformable", "0.00", "地形/泥地"],
        ["gravel 碎石", "1800", "0.60", "0.00", "deformable", "0.00", "碎屑/砂石"],
        ["soft_tissue 软组织", "1000", "0.50", "0.00", "deformable", "0.90", "行人/动物"],
        ["unknown 兜底", "1000", "0.50", "0.10", "rigid", "0.00", "噪声/未知"],
    ],
    col_widths=[1.2, 0.9, 0.7, 0.8, 1.0, 0.9, 1.4],
)
body("刚性字段 rigid/semi_rigid/deformable/frangible/fluid 决定物理引擎的碰撞体类型，可破坏性决定损毁血条阈值。"
     "每类还定义了 **损毁响应 damage_response**（none 刚性不可破坏 / deform 只变形不碎 / break 局部断裂 / shatter "
     "玻璃整体碎裂），以及**每类典型质量先验**（car=1500kg、truck=8000kg、bus=12000kg 等）。", indent=True)

heading("3.4 实例级聚类与级联物体判类", 2)
body("级联的实例分割采用「逐点 offset 回归 + 聚类」的投票式方法（② 环）。对 thing 点（可移动物体）回归一个指向其所属实例"
     "质心的 3D 偏移向量 offset = 质心 − 点坐标（PointMambaSeg 的实例头，3 通道），推理时把每个点平移到预测质心，再对平移后的点"
     "做 DBSCAN 聚类，得到独立实例及其实例级属性（类别、中心、包围盒、物理属性）。", indent=True)
body("由于逐点语义对稀有 thing 类（摩托车、自行车、行人等）几乎全灭，本文在实例分割之上再加一层**物体级判类**（③ 环）："
     "对每个聚出的实例，取其实例点云做居中 + 单位球归一化 + 采样到固定点数（1024），输入一个 11 类（car/pedestrian/barrier/"
     "truck/trafficcone/trailer/pushable/bus/motorcycle/construction/bicycle）的 PointMambaScan 分类器（宏准确率 87.4%）"
     "重新判类，并把该类别回填到实例内所有点。级联完整链路为：**① 地面几何分割 → ② offset 实例分割 → ③ 物体级判类**。", indent=True)

heading("3.5 位置相关破坏模型", 2)
body("更真实的损坏逻辑 = **材料强度 × 几何结构强度 + 撞击局部化**：", indent=True)
body("1. 对每个实例点云进行 KMeans 子部位分解（k 随点数自适应）；", indent=True)
body("2. 每部位强度 = 材料断裂强度（MATERIAL_STRENGTH，金属 0.90、混凝土 0.95、玻璃 0.12、植被 0.20）× 几何结构强度"
     "（用点云协方差奇异值比 σ₃/σ₁ 衡量\"厚实度\"，薄板/细枝更弱）；", indent=True)
body("3. 撞击只作用于撞击点半径范围内的部位（局部化），部位损伤累积超过 hp = strength × 1000J 即破碎；", indent=True)
body("4. 物体按破碎部位比例进入 intact → damaged → destroyed 状态；damage_response 决定是掉碎块（break/shatter）、"
     "只变形（deform）还是不可破坏（none）。", indent=True)
body("由此实现\"**撞同一物体不同位置，得到不同破坏结果**\"。", indent=True)
body("需要说明本模型的可信范围：其中材料强度常数与破坏阈值的标定情况见 §4.4.5——本文验证的是该模型的**相对**性质"
     "（位置相关性），其绝对阈值尚待按应变能密度重标。", indent=True)

heading("3.6 导航软碰撞决策", 2)
body("决策在**两类接触模式**下进行，二者共用同一安全约束，但风险量纲不同。", indent=True)
body("**模式 A（正碰软目标）**：在前向视锥（距离 + 半锥角约束）内保留**可破坏(frangible) + 可变形(deformable)** 类别作为候选，"
     "按 lethality 升序（越安全越靠前）推荐最低致命性的撞击目标；致命（行人/动物）与硬刚性（车/墙）被排除，可通过类"
     "（路面/人行道/地形）是行驶面、不作为\"撞击目标\"。锚定候选（如植被）的减速量按**区间**报告（§4.4.1），排序取区间下端以保守。",
     indent=True)
body("**模式 B（刮蹭硬面）**：在**侧向走廊**（航向坐标系下 u ∈ [3, 60] m、|w| ∈ [1.5, 8] m）内寻找与行进方向大致**并行**的"
     "硬刚性锚定面，以**小接近角**侧向擦拭，靠持续库仑摩擦减速。候选须通过墙面几何判据：转到航向坐标系后按纵向分箱，要求同一侧"
     "存在足够长的连续墙面（相邻箱横向漂移 < 0.35 m，**转角处因此自然断开**）、墙高 ≥ 0.6 m、接近角 ≤ 20°、可用行程 ≥ 10 m。"
     "刮蹭的风险量纲不是 lethality 而是**接近角**（越平缓越安全），因为轻蹭墙面并不是\"撞墙\"。", indent=True)
body("决策在 risk ≤ risk_limit 的约束下最大化 Δv。需要注意模式 B 使用**另一套搜索区域**：正碰看正前方，而可刮蹭的墙必须与路径"
     "并行、位于车侧，前向视锥看不到它。两类模式的减速量与缺口分析见 §4.4。", indent=True)

heading("3.7 跨帧实例跟踪", 2)
body("对连续关键帧做实例聚类后，用**同类 3D 包围盒 IoU + 匈牙利匹配**保持实例跨帧稳定 ID：静态物体（建筑/树/路沿）融合点云"
     "成地图级物体，动态物体（车/行人）记录轨迹，连续多帧未匹配则注销。", indent=True)

heading("3.8 系统导出与集成", 2)
body("**游戏**：导出 game_scene.json（collider/rigidbody/damage），以及 objects.csv/parts.csv（Unreal DataTable 直接导入）；"
     "提供 Unity C# 组件 GameSceneLoader 与 DestructibleObject，自动生成 BoxCollider + Rigidbody + 位置相关破坏脚本。", indent=True)
body("**导航**：封装为 ROS2 节点（soft_contact 包），订阅点云 + 里程计，发布推荐撞击点、候选目标与摘要。此外提供**语义+几何"
     "融合导航**：把语义障碍（PointMamba 实时分割按 obstacle_classes 过滤）与坡度障碍（PCA 地面坡度估计，陡坡 cell）作为"
     "「标记型」观测源叠加进 Nav2 costmap，使规划同时避开「是什么（语义障碍）」与「能不能上（陡坡）」"
     "（slope_obstacle_publisher.py + 增强参数 nav2_params_semantic.yaml）。", indent=True)
body("坐标约定：nuScenes(x 前, y 左, z 上) → Unity(x 右, y 上, z 前)。", indent=True)

# ======================================================================
# 4 实验
# ======================================================================
heading("4 实验", 1)
heading("4.1 数据集与实验设置", 2)
body("采用 nuScenes 公开数据集（32 类 lidarseg 语义标签，700 train / 150 val scenes，共 34,149 关键帧），输入为 LiDAR 点云 + "
     "6 相机投影 RGB（6 通道），每帧采样 16,384 点。分割模型 PointMambaSeg：trans_dim=384、depth=12、group_size=32、in_channels=6、"
     "num_classes=32，AdamW 优化器（lr=1e-3、weight_decay=0.05），余弦退火，200 epoch。num_group 为消融变量（64 / 256）。硬件为 RTX 5070 Ti。", indent=True)

heading("4.2 分割前置结果", 2)
body("逐类 IoU（baseline_4096，val 集，表 2 的逐类明细）：", indent=True, space_after=3)
caption("表 2　32 类逐类 IoU（baseline_4096，val 集）")
table(
    ["idx", "类别", "IoU", "idx", "类别", "IoU"],
    [
        ["0", "noise", "0.187", "16", "vehicle.bus.rigid", "0.166"],
        ["1", "animal", "0.000", "17", "vehicle.car", "0.447"],
        ["2", "human.pedestrian.adult", "0.034", "18", "vehicle.construction", "0.114"],
        ["3", "human.pedestrian.child", "0.000", "19", "vehicle.emergency.ambulance", "—"],
        ["4", "construction_worker", "0.004", "20", "vehicle.emergency.police", "0.003"],
        ["5", "personal_mobility", "0.000", "21", "vehicle.motorcycle", "0.014"],
        ["6", "police_officer", "0.000", "22", "vehicle.trailer", "0.290"],
        ["7", "stroller", "0.000", "23", "vehicle.truck", "0.201"],
        ["8", "wheelchair", "—", "24", "flat.driveable_surface", "0.827"],
        ["9", "movable_object.barrier", "0.339", "25", "flat.other", "0.476"],
        ["10", "movable_object.debris", "0.010", "26", "flat.sidewalk", "0.450"],
        ["11", "pushable_pullable", "0.025", "27", "flat.terrain", "0.519"],
        ["12", "movable_object.trafficcone", "0.139", "28", "static.manmade", "0.762"],
        ["13", "static_object.bicycle_rack", "0.367", "29", "static.other", "0.013"],
        ["14", "vehicle.bicycle", "0.001", "30", "static.vegetation", "0.779"],
        ["15", "vehicle.bus.bendy", "0.078", "31", "vehicle.ego", "0.939"],
    ],
    col_widths=[0.5, 2.6, 0.7, 0.5, 2.6, 0.7],
)
body("（\"—\"表示该类别在 val 集无 GT，不计入 mIoU；完整 32 类与 seg_16384 逐类对照见 results/comparison.md）", size=9)
caption("表 2（整体指标汇总）")
table(
    ["实验", "点数", "num_group", "训练轮数", "mIoU", "OA"],
    [
        ["baseline_4096", "4096", "64", "200", "0.2395", "0.8294"],
        ["seg_16384", "16384", "64", "200", "0.2323", "0.8095"],
        ["seg_16384_g256", "16384", "256", "200", "0.3090", "0.8679"],
        ["seg_32768_g1024", "32768", "1024", "200", "0.3697¹", "0.9266"],
        ["seg_16k_ng1024（主配置）", "16384", "1024", "40", "0.3923", "0.9027"],
        ["seg_16k_ng2048", "16384", "2048", "40", "0.3965", "0.9100"],
    ],
    col_widths=[1.9, 0.8, 0.8, 0.8, 0.8, 0.8],
)
body("前四行是不同配置的历史先导实验（200 轮预算，配置随行变化），后两行来自 §4.2.1 的受控扫描（40 轮预算，每次只变一个变量），"
     "也是本文主实验采用的分割前置。注意末两行用一半的点数超过了 32768 点的历史实验：16k／num_group=1024 的 mIoU 0.3923 高于 "
     "32k／1024 的 0.3697——这一点在 §4.2.1 与 §4.14.2 中以严格配对的方式得到确认。", indent=True)
body("¹ 该行为 eval_seg_full.py 在完整 val 集上的复评值；其训练内 val best 记为 0.3695，两者差 0.0002（该组训练早于统一评测脚本）。"
     "后两行与 §4.14 各表均以完整 val 集口径报告。", size=9)
body("分割前置在常见类别上表现良好（driveable_surface 0.827、manmade 0.762、vegetation 0.779、ego 0.939），但 rare thing 类"
     "（animal/child/police/stroller/motorcycle 等）近乎全灭，是 mIoU 的主要拖累。", indent=True)
body("**rare 类全灭的根因：类别不平衡（长尾分布）。** nuScenes 32 类存在严重的类别频率失衡——driveable_surface、vegetation、"
     "manmade 等高频类占据绝大多数点，而 animal、child、police_officer、motorcycle、bicycle 等稀有类别在训练集中样本占比极低。"
     "在逐点交叉熵损失按全体点求平均的机制下，模型学到的最优解是\"把稀有类别点预测为高频类\"以最小化整体损失，故这些类的 IoU "
     "近乎为 0；而 IoU 对稀有类天然残酷——分母小，一旦预测错少量点即归零。这一现象是逐点语义分割中普遍存在、且与模型结构基本无关"
     "的长尾问题，并非 PointMambaSeg 特有。常见的缓解手段包括按类别频率加权的损失或焦点损失（Focal Loss）、稀有类别重采样、"
     "以及面向稀有物体的复制-粘贴增广（copy-paste / PolarMix 等），本文将其列为未来工作（§5）。", indent=True)
body("**消融：升点数 vs 升 token 数（定位规模化瓶颈）**。三组对照：① 升点数——在 num_group=64 固定下将点数 4096→16384，"
     "mIoU 0.2395→0.2323、OA 0.8294→0.8095 略降（逐类有涨有跌：car 0.447→0.476、barrier 0.339→0.418、driveable 0.827→0.855 "
     "提升；manmade 0.762→0.725、vegetation 0.779→0.718 下降），为负结果；② 升 token 数——保持 16384 点不变，将 num_group 64→256，"
     "mIoU 0.2323→0.3090（+33%）、OA 0.8095→0.8679（+5.8 点），200 epoch 跑满，best 出现在 epoch 133；③ 同时升 token 数与点数——"
     "改用 GridGroup 体素分组（grid_size=0.5，O(N) 复杂度）把 num_group 扩到 1024、并把点数翻倍到 32768，mIoU 进一步提升到 0.3697、"
     "OA 达 0.9266（200 epoch 跑满，best 出现在 epoch 145）。三组消融合起来定位了瓶颈：**Mamba 的规模化瓶颈主要是 token 数"
     "（num_group）**——num_group 决定序列建模的信息容量上限，token 数不变时单纯增加点数几乎无法带来信息增益。这一发现把此前"
     "孤立的\"升点数负结果\"解释为同一瓶颈的两面。", indent=True)
body("上段对照②与③的数值随所属行的训练预算不同而不可直接相比，且③中\"点数翻倍是否真有增益\"在当时被训练轮数混淆（见 §4.14.2）。"
     "§4.2.1 用统一 40 轮预算重做了受控扫描，结论比上述初步判断更强：**点数翻倍不但无增益，反而有害**。", indent=True)

heading("4.2.1 超参扫描：num_group 的饱和边界与 grid_size 的驻点", 3)
body("上一节的消融已把瓶颈定位到 token 数，但那条曲线的三个点配置各异（点数与分组方式同时变化），无法回答两个与部署直接相关的问题："
     "token 预算加到多少为止？体素边长取多少合适？为此本节固定其余全部条件——输入 16384 点、batch size 8、40 epoch 跑满、同一评测"
     "口径——每次只扫一个变量。全部结果用 eval_seg_full.py 在完整 val 集上评测（GT 点数 98 615 296，与训练期 val 完全同口径，评测值"
     "与训练内 best 逐位吻合），罕见类沿用 §4.14 的定义（val 中有 GT 点的 10 个罕见类，wheelchair 因 val 无 GT 不计入）。", indent=True)
body("固定 grid_size = 0.5 m，扫 num_group（表 2a）：", indent=True, space_after=3)
caption("表 2a　num_group 扫描（16384 点、grid_size 0.5 m、40 epoch）")
table(
    ["num_group", "mIoU", "OA", "罕见类 mIoU", "相对上一档 ΔmIoU"],
    [
        ["256", "0.3144", "0.8766", "0.1152", "—"],
        ["1024", "0.3923", "0.9027", "0.2145", "+0.0779"],
        ["2048", "0.3965", "0.9100", "0.2429", "+0.0042"],
    ],
    col_widths=[1.0, 0.9, 0.9, 1.1, 1.6],
)
body("256→1024（token 数 4 倍）带来 +0.0779；1024→2048（token 数再翻倍）只带来 +0.0042，即前一次增益的 5.4%，而单次训练耗时从 7.4 h "
     "增至 13.9 h。**收益已落入与代价不相称的区间——1024 是该配置下的性价比拐点，这也是本文主实验取 num_group=1024 的依据。**"
     "罕见类的响应衰减得更慢：0.1152→0.2145→0.2429，第二段仍有 +0.0284（为第一段的 28.6%）。也就是说，**token 预算对罕见类的边际"
     "收益比全类更持久**，这与 §4.14.2 的机理一致——罕见类的失效本就是\"没有自己的 token\"，多给 token 直接命中其瓶颈；而全类 mIoU "
     "在 1024 之后受限于其他因素。", indent=True)
body("固定 num_group = 1024，扫 grid_size（表 2b）：", indent=True, space_after=3)
caption("表 2b　grid_size 扫描（16384 点、num_group=1024、40 epoch）")
table(
    ["grid_size (m)", "mIoU", "OA", "罕见类 mIoU"],
    [
        ["0.25", "0.3887", "0.9019", "0.2183"],
        ["0.5", "0.3923", "0.9027", "0.2145"],
        ["1.0", "0.3731", "0.8993", "0.1956"],
    ],
    col_widths=[1.3, 1.0, 1.0, 1.3],
)
body("**0.5 m 是全类指标的驻点，且两侧的下降有不同的成因**。粗化到 1.0 m 是\"物体被并掉\"——体素边长超过行人尺度，行人级物体重新被"
     "并入地面 token，全类（−0.0192）与罕见类（−0.0189）同时下降。细化到 0.25 m 则不降不升：全类 −0.0036、罕见类 +0.0038，两者几乎"
     "对消。原因在 §4.14.2 末已出现的体素合并机制：0.25 m 下体素数远超 1024，超出部分会被 FPS 合并回最近的保留体素，分组并未真正细化。"
     "但合并本身不是中性的——它按\"离保留体素最近\"合并，等价于在固定 token 预算下把 token 重新分配给场景中更小的结构，因此小物体略受益、"
     "大物体略受损。**净效果为零，说明在 token 预算固定时，\"更细的体素\"本身不是杠杆**；grid_size 的合理取值应由 token 预算与场景物体"
     "尺度共同决定，而不是越小越好。", indent=True)
body("两条曲线合起来给出的结论是：本文主实验配置（16384 点 / num_group=1024 / grid_size 0.5）在三个维度上都处在\"已越过门槛、尚未"
     "进入过饱和\"的位置——这与 §4.14.1 中\"16k 已越过采样门槛\"的判断在 token 维度上互为对应物。", indent=True)

heading("4.2.2 RGB 消融：颜色在几何相似处提供消歧", 3)
body("§3.2 的网络采用 RGB 与 XYZ 的早期融合（in_channels=6：三维坐标 + 三通道颜色）。这一设计此前缺少直接证据——早先的通道有效性"
     "分析（scripts/rgb_validity_probe.py）测的是\"单看颜色能否分类\"，结论是颜色几乎不携带类别身份（平衡精度 0.054），这曾使早期融合"
     "看起来像是一个未经辩护的选择。本节用严格配对的消融回答该问题：**同一份配置、同一份数据、同样 40 epoch 跑满，只把输入通道从 6 "
     "改成 3**（表 2c）。", indent=True, space_after=3)
caption("表 2c　RGB 消融（其余条件严格相同，仅输入通道 3 vs 6）")
table(
    ["输入通道", "mIoU", "OA", "罕见类 mIoU"],
    [
        ["XYZ（3 通道）", "0.3592", "0.8947", "0.1816"],
        ["XYZ + RGB（6 通道）", "0.3923", "0.9027", "0.2145"],
        ["Δ", "+0.0331（相对 +9.2%）", "+0.0080", "+0.0329"],
    ],
    col_widths=[1.9, 1.7, 1.0, 1.3],
)
body("两组只在输入通道数上有差异：模型结构、分组、token 预算、优化器与调度、数据划分、评测脚本全部相同，因此 +0.0331 可直接归因于"
     "颜色。除最终 best 的差值外，两组在全部 8 个同 epoch 配对点上都是一致的正值（epoch 10 / 15 / 20 / 25 / 28 / 29 / 30 / 31 的 Δ "
     "依次为 +0.0295 / +0.0322 / +0.0261 / +0.0305 / +0.0323 / +0.0292 / +0.0352 / +0.0268，均值 ≈ +0.030），排除了\"best 由单次"
     "波动决定\"的可能。", indent=True)
body("**这与\"颜色单独不携带类别身份\"并不矛盾，反而给出了一致的解释。**逐类看增益的分布极不均匀，且高度集中在几何上难以区分、但"
     "外观有强约定的类别上：", indent=True, space_after=3)
caption("表 2d　RGB 增益的逐类分布（仅列 |Δ| 最大的六个类）")
table(
    ["类别", "XYZ", "+RGB", "Δ"],
    [
        ["vehicle.emergency.police", "0.127", "0.457", "+0.330"],
        ["vehicle.construction", "0.179", "0.344", "+0.164"],
        ["movable_object.trafficcone", "0.292", "0.410", "+0.118"],
        ["human.pedestrian.construction_worker", "0.077", "0.182", "+0.105"],
        ["human.pedestrian.adult", "0.408", "0.490", "+0.082"],
        ["vehicle.car", "0.776", "0.807", "+0.031"],
    ],
    col_widths=[3.0, 0.8, 0.8, 0.8],
)
body("警车（+0.330）、工程车（+0.164）、交通锥（+0.118）、施工人员（+0.105）——这四个类恰好都是\"几何与同类普通物体几乎相同、靠涂装/"
     "颜色/反光背心区分\"的目标。警车与普通轿车、工程车与货车、施工人员与行人，在 LiDAR 几何上分别是同一类形状；颜色是唯一可用的判别"
     "信号。而颜色对几何本身可分的类几乎没有贡献（vehicle.bus.bendy −0.001、vehicle.bicycle −0.008）。唯一明显的负项是 "
     "movable_object.debris（−0.049），可解释为碎屑的颜色分布过杂，注入的颜色通道引入了噪声。", indent=True)
body("据此可以把 §3.2 的设计理由说清楚：**颜色不编码类别身份，而是在几何相似处提供消歧**。这也是本文把 RGB 作为早期融合输入而非"
     "后处理外观特征的原因——消歧信号必须在 token 化之前进入，才能影响哪些点被分到同一个 token；若放在分组之后，同类几何形状的点已经"
     "混在一起，颜色再无从区分。", indent=True)
body("这同时更正了早先分析的一个推论。原先\"颜色判别力弱（平衡精度 0.054）\"被读作\"RGB 融合价值有限\"，但那是脱离几何测颜色的独立"
     "判别力；融合场景下的正确问题是\"给定几何，颜色能否补充信息\"，答案是能，且补充量正好落在几何歧义最大的类别上。两点不冲突：前者低，"
     "后者高。", indent=True)

heading("4.3 物理属性映射结果", 2)
body("32 类语义 → 物理属性完整映射（表 3，节选）：", indent=True, space_after=3)
caption("表 3　32 类语义→物理属性映射（节选）")
table(
    ["类别", "物理材质", "安全等级", "致命性", "质量(kg)", "损毁响应"],
    [
        ["animal / pedestrian.*", "soft_tissue", "lethal", "1.00", "25–75", "deform"],
        ["movable_object.barrier", "plastic", "frangible", "0.30", "30", "deform"],
        ["movable_object.trafficcone", "rubber", "frangible", "0.10", "5", "deform"],
        ["movable_object.pushable_pullable", "plastic", "frangible", "0.25", "20", "break"],
        ["movable_object.debris", "gravel", "deformable", "0.15", "5", "none"],
        ["vehicle.car / bus.* / truck / trailer / ego", "metal", "hard", "0.80–0.90", "1500–12000", "deform"],
        ["vehicle.motorcycle / bicycle", "metal", "hard", "0.75–0.80", "15–200", "deform"],
        ["flat.driveable_surface", "asphalt", "passable", "0.00", "—", "none"],
        ["flat.sidewalk / flat.other", "concrete", "passable", "0.10–0.20", "—", "none"],
        ["flat.terrain", "soil", "passable", "0.05", "—", "deform"],
        ["static.manmade", "concrete", "hard", "0.90", "—", "none"],
        ["static.other", "rock", "hard", "0.70", "500", "none"],
        ["static.vegetation", "vegetation", "deformable", "0.20", "200", "break"],
        ["noise", "unknown", "passable", "0.00", "—", "none"],
    ],
    col_widths=[2.6, 1.1, 1.0, 0.8, 1.0, 1.0],
)
body("映射结果与直觉一致：交通锥/护栏为可破坏（低代价），灌木为可变形（可穿越），路面/人行道/地形为可通过（行驶面，正常行驶），"
     "行人/动物为致命（代价 1.0），建筑/车为硬刚性；完整 32 类映射见 utils/materials.py 的 CLASS_META。", indent=True)

caption("图 3　语义→物理映射可视化：(a) 真实 RGB 点云 → (b) 32 类逐点语义标签 → (c) 5 级碰撞安全等级着色，"
        "直观展示「语义→物理」这一核心跃迁。")
figure_row([
    (FIG3_RGB, "(a) 真实 RGB"),
    (FIG3_SEM, "(b) 语义标签"),
    (FIG3_SAF, "(c) 碰撞安全等级"),
], width=2.0)
figure_row([
    (LEG_SEM, "32 类语义图例"),
    (LEG_SAF, "5 级安全图例"),
], width=2.6)

heading("4.4 撞击后果标定", 2)
body("§3.5 的破坏模型与 §3.6 的决策模块共享同一个输入：**撞击给自车带来的实际减速量 Δv**。该量若不标定，模型中的绝对阈值就只是"
     "约定值，\"撞了什么、会怎样\"便无法对外复核。本节对它做标定，并据此指出决策模块的缺口与改进方向。", indent=True)

body("**4.4.1 标定方法：闭合形式解而非物理仿真器**", indent=True)
body("对自由物体，正碰的速度变化有精确解 Δv = (1+e)·m₂/(m₁+m₂)·v，其中 m₁ = 1500 kg 为自车质量、v = 30 m/s（108 km/h）、"
     "e 为该类恢复系数。锚定物体（建筑、树、固定护栏）在完全刚性极限下 Δv = (1+e)v，即上式 m₂→∞ 的极限；可破坏的锚定物在断裂前"
     "需做功，其点值取决于断裂功模型，故本文**只给区间 [自由体值, (1+e)v]，不给点值**。", indent=True)
body("我们曾以 PyBullet 作为标定基准，结果**否定了这一方案**：实测最大速度相对误差 33.7%；更关键的是**动量偏差在全部算例、全部"
     "参数取值下恒为约 8.0%**——扫时间步（10⁻³ ~ 5×10⁻⁵ s）、扫 ERP（0.0 / 0.2 / 0.8）、扫求解器迭代数（10 ~ 1000），结果"
     "**逐位相同**；把质量比从 1:1 改到 300:1，该偏差几乎不变（7.98% → 7.99%）。一个与所有数值参数和质量比都无关的恒定动量亏损，"
     "说明它不是离散化或收敛精度问题，而是求解器在本问题上的**系统性偏离**：以等质量、e = 0.7 为例，有效恢复系数由设定的 0.7 "
     "退化到约 0.52。以一个有偏的仿真器作为基准，等于把有偏模型当作测量——这正是本文要避免的做法。因此全文改用上述闭合形式解："
     "它满足动量守恒与能量不增，并通过四个极限行为自检（m₂→0、m₂→∞、锚定刚性、等质量 e=0），最差的动量相对误差为 1.6×10⁻¹⁶。",
     indent=True)

body("**4.4.2 每类减速的标定**", indent=True, space_after=3)
caption("表 4　30 m/s、1500 kg 自车下的正碰减速标定（节选）")
table(
    ["类别", "材质", "锚定", "质量先验 (kg)", "Δv (m/s)", "占初速"],
    [
        ["movable_object.debris", "gravel", "否", "5", "0.100", "0.3%"],
        ["movable_object.trafficcone", "rubber", "否", "5", "0.169", "0.6%"],
        ["vehicle.bicycle", "metal", "否", "15", "0.327", "1.1%"],
        ["movable_object.pushable_pullable", "plastic", "否", "20", "0.553", "1.8%"],
        ["movable_object.barrier", "plastic", "否", "30", "0.824", "2.8%"],
        ["vehicle.motorcycle", "metal", "否", "200", "3.882", "12.9%"],
        ["vehicle.car", "metal", "否", "1500", "16.500", "55.0%"],
        ["vehicle.truck", "metal", "否", "8000", "27.789", "92.6%"],
        ["static.vegetation", "vegetation", "是", "200", "[3.71, 31.5]", "—"],
        ["static.other", "rock", "是", "500", "[7.88, 31.5]", "—"],
        ["static.manmade", "concrete", "是", "1200*", "[14.0, 31.5]", "—"],
    ],
    col_widths=[2.4, 1.0, 0.6, 1.1, 1.1, 0.8],
)
body("自由物体为精确值，锚定类给出区间。* 该类无质量先验，按材质密度 × 0.5 m³ 的兜底估计；锚定类的质量只进入区间下端，"
     "上界 (1+e)v 与质量无关。可通过类（路面/人行道/地形）是行驶面而非撞击目标，故不列入（§3.3）。", indent=True)

body("**4.4.3 决策缺口：现有候选集的减速上限仅 2.75%**", indent=True)
body("§3.6 的 select_soft_contact 在「可破坏 + 可变形」集合内按 lethality 升序推荐。由表 4，该集合的自由体成员（碎屑 5 kg、"
     "锥 5 kg、推拉物 20 kg、护栏 30 kg）在 30 m/s 下的减速上限仅 **0.824 m/s，即初速的 2.75%**。val 集 40 帧上的实测与此一致："
     "37 帧有推荐，推荐目标为植被 29 帧、锥 6 帧、护栏 2 帧，Δv 中位 3.71 m/s——但该值来自**锚定**的植被，属区间下端。", indent=True)
body("⇒ 现有规则优化的是「**不伤人**」，并未优化「**减速**」。二者在该候选集上不是同一件事：轻质量自由体可被撞飞、确实不伤人，"
     "但对 1500 kg 自车的减速可忽略。这与应用目标（失控场景下的选择性碰撞减速）存在实质缺口。", indent=True)

body("**4.4.4 改进：纳入锚定类与刮蹭模式**", indent=True)
body("针对上述缺口，§3.6 的决策被扩展为两类接触模式并纳入锚定类：", indent=True)
body("• **模式 A：正碰软目标。** 锚定类按区间报告（表 4），排序取区间下端（保守）。", indent=True)
body("• **模式 B：刮蹭硬面。** 以**小接近角**侧向擦过硬刚性锚定面（墙体/建筑/固定护栏），靠持续库仑摩擦减速。法向力上限来自"
     "轮胎侧向力上限 μ_tire·m·g，故减速度 a = μ_wall·μ_tire·g **与质量无关**；取 μ_tire = 0.90（干沥青）、μ_wall = 0.50"
     "（车身-混凝土），a = 4.415 m/s²，行程 L 后的减速 Δv = min(aL/v, v)。30 m/s 下 35 m 行程可减 **5.150 m/s"
     "（初速的 17.2%）**，约为软目标正碰上限的 6 倍。这正是「撞墙」与「蹭墙」的区别：正碰致命，刮蹭可控。", indent=True)
body("在合成几何（墙面侧向 4.0 m、纵向 2~62 m、高 3.0 m，32 线雷达模型）上，规划器检出墙长 35.0 m、接近角 −0.00°、"
     "侧向 4.00 m，给出 Δv = 5.150 m/s，与手算**误差 0.00%**。", indent=True)
body("但必须给出一个数据集层面的限制：与行进方向**并行**的墙对旋转式激光雷达是**掠射**，入射角随距离趋于平行，采样密度由近处的 "
     "954.3 点/m 降至远处的 2.93 点/m，**衰减 325 倍**。在 nuScenes val 的 200 帧上实测，可刮蹭墙面长度的中位数为 0 m、最大 8 m，"
     "75% 的帧不存在任何长度 ≥1 m 的并行墙面。**因此刮蹭模式在现有数据集上无法获得真实样本验证，其有效性目前仅在合成几何下成立。**"
     "这是采集几何的必然，与算法或采样密度无关——16384 与 65536 两种采样给出完全相同的结果。", indent=True)

body("**4.4.5 强度参数标定：§3.5 中哪些部分可信**", indent=True)
body("§3.5 的每部位强度由 MATERIAL_STRENGTH 与几何因子相乘。把 MATERIAL_STRENGTH 与文献失效强度按应变能密度 σ_f²/(2E) 排序对照，"
     "二者**不相关**：Spearman ρ = 0.105（p = 0.734，13 种材质）。偏差最大的是岩石（模型序 1、物理序 9）与混凝土（2 / 10）——"
     "混凝土的 σ_f²/(2E) = 1.5×10⁻⁴ 在全部材质中接近最低，而模型给了次高强度；橡胶则相反（模型序 6、物理序 1，弹性体应变能容量"
     "最高）。", indent=True)
body("⇒ §3.5 的**绝对强度与绝对破坏阈值未经标定，不能作为定量结论引用**。本文保留并验证的是该模型的**相对性质**：同一能量作用于"
     "同一物体的不同部位会得到不同结果，其差异来自几何结构因子 σ₃/σ₁（薄板 ≈ 0.25、厚实块 ≈ 1.0）而非材料常数。以真实 nuScenes "
     "实例（帧 000260ba…）为例，对 vehicle.trailer 施加 270 J，撞最弱部位进入 destroyed、撞最强部位保持 intact；static.vegetation "
     "在 137 J 下同样呈现弱部位碎裂、强部位完好。**该演示说明的是位置相关性这一机制成立，其绝对判定阈值随本节标定结果而变**；"
     "按 σ_f²/(2E) 重标 MATERIAL_STRENGTH 属后续工作（§5）。", indent=True)

heading("4.5 导航软碰撞决策验证", 2)
body("在真实 nuScenes 帧（000260ba…，4096 点）上，改变航向观察决策变化（select_soft_contact，前向 50m、半锥角 40°）：", indent=True)
body("• 航向 0°：候选 1 个，推荐 **static.vegetation**（可变形，代价 0.20，前向 12.4m）", indent=True)
body("• 航向 90°：候选 1 个，推荐 **static.vegetation**（可变形，代价 0.20，前向 24.8m）", indent=True)
body("• 航向 180°：候选 0 个，视锥内无安全目标", indent=True)
body("• 航向 −90°：候选 3 个，推荐 **movable_object.trafficcone**（可破坏，代价 0.10，前向 46.5m）；次优 static.vegetation"
     "（0.20，17.4m）、movable_object.barrier（0.30，48.4m）", indent=True)
body("结果符合预期：决策始终优先低致命性、可破坏/可变形目标，排除致命与硬刚性目标；同帧不同航向下推荐目标随视锥内实际物体变化。"
     "本节展示的是**模式 A（正碰软目标）**的行为；该帧未检出满足几何判据的并行锚定墙面，故模式 B 无候选——这与 §4.4.4 的结论一致："
     "真实数据中可刮蹭长墙受掠射采样限制而稀缺。", indent=True)

heading("4.6 跨帧实例跟踪验证", 2)
body("在 scene-0003 连续 6 帧上：71 个静态物体保持稳定 ID 并融合 9,410 点（fused_map.npz），7 辆动态车辆记录轨迹。静态物体"
     "点云跨帧累积，动态物体只更新位置不融合，符合地图级重建的预期。", indent=True)

heading("4.7 系统集成验证", 2)
body("**Unity**：GameSceneLoader 读取 game_scene.json 自动生成带碰撞体、刚体与破坏组件的场景物体；DestructibleObject 复刻"
     "位置相关破坏逻辑。", indent=True)
body("**Unreal**：objects.csv/parts.csv 可直接导入 DataTable。", indent=True)
body("**ROS2**：soft_contact 包 colcon build 通过，pytest 端到端测试（PointCloud2 → 分割 → 决策 → 发布推荐目标）2 passed。", indent=True)
body("**Nav2 语义+几何融合导航**：离线仿真（假 TF + 假激光 + AMCL 定位）验证了 Nav2 定位与路径规划全链路；坡度障碍节点在 "
     "Semantic3D 场景上离线 dry-run 输出 114 个陡坡 cell / 16,785 个障碍点，语义+坡度两个障碍观测源通过 ROS 发布"
     "（/semantic_obstacles 50 万点、/slope_obstacles 16,785 点），Nav2 加载增强参数后 local/global costmap 均订阅 "
     "scan semantic_obstacles slope_obstacles 三个观测源，smoke test 无报错。", indent=True)

heading("4.8 实例分割与级联判类结果", 2)
body("实例分割在 16k 降采样 val 子集（300 场景）上的 PQ（IoU>0.5 贪心匹配，表 5）：", indent=True, space_after=3)
caption("表 5　实例分割 PQ（IoU>0.5 贪心匹配）")
table(
    ["方法", "PQ"],
    [
        ["几何基线（原始坐标 DBSCAN，eps=0.5）", "0.381"],
        ["冻结 backbone + offset 回归", "0.509"],
        ["解冻微调（联合语义+offset）", "0.516"],
    ],
    col_widths=[3.5, 0.9],
)
body("端到端级联在 thing 点（11 个 B2 覆盖类）上的语义精度（500 val 场景，表 6）：", indent=True, space_after=3)
caption("表 6　端到端级联在 thing 点上的语义精度")
table(
    ["配置", "dense 逐点", "级联判类", "提升"],
    [
        ["冻结 offset", "0.485", "0.687", "+20.2"],
        ["解冻微调", "0.502", "0.690", "+18.8"],
    ],
    col_widths=[1.5, 1.1, 1.1, 0.9],
)
body("级联判类将摩托车（0→0.76）、公交（0.20→0.72）、施工（0.11→0.64）等盲类大幅救回；唯一倒退的是 barrier（护栏，0.73→0.40），"
     "根因与修复见 §4.10。", indent=True)
body("注：实例 PQ（表 5）与级联 thing 语义精度（表 6）分别在 300 与 500 场景的 val 子集上评测，两组实验的评测脚本与采样范围不同，"
     "故两表不跨子集比较；本节结论均基于各自子集内的同组对照。", indent=True, size=9)

heading("4.9 消融：聚类阈值与「PQ 提升 ≠ 级联精度提升」", 2)
body("对微调 offset 模型做聚类阈值 eps 扫描，得到**本文最重要的一条消融发现**（表 7）：", indent=True, space_after=3)
caption("表 7　聚类阈值 eps 扫描（实例 PQ vs 级联 thing 精度）")
table(
    ["eps", "实例 PQ", "级联 thing 精度"],
    [
        ["0.5", "0.516", "0.690（峰）"],
        ["0.6", "0.550", "0.663"],
        ["0.7", "0.578", "0.673"],
        ["1.0", "0.606（峰）", "0.647"],
    ],
    col_widths=[1.0, 1.3, 1.6],
)
body("**发现 1（PQ 与级联精度方向相反）**：offset 平移点分散约 1 m 量级，故实例 PQ 在较大 eps≈1.0 处达到峰值（0.606）；但级联"
     "下游的物体判类需要「干净的单物体实例」，较大 eps 会把相邻实例过合并成一个 blob，分类器给整个 blob 判单一类别，导致第二物体"
     "整段判错——故级联 thing 精度反而在 eps=0.5 处最高（0.690），随 eps 增大单调下降。**以 PQ 为目标的实例分割最优点，并非级联"
     "语义精度的最优点**，二者存在本质矛盾。", indent=True)
body("**发现 2（offset 回归贡献的诚实量化）**：在相同 eps 下对照原始坐标 DBSCAN（不做 offset，表 8）：", indent=True, space_after=3)
caption("表 8　offset 回归贡献（几何基线 vs offset PQ）")
table(
    ["eps", "几何基线 PQ", "offset 回归 PQ"],
    [
        ["0.5", "0.381", "0.516（+13.5）"],
        ["1.0", "0.587", "0.606（+1.9）"],
    ],
    col_widths=[1.0, 1.4, 1.6],
)
body("offset 投票的真正价值在**小 eps 下把点聚紧**（0.5 处 +13.5 点），从而无需依赖大半径合并即可区分相邻实例；而在各自最优大 "
     "eps 下，offset 的增量仅 +1.9 点。此前常被报告的「+12 点」实为在次优默认 eps=0.5 下的对比产物。", indent=True)
body("**发现 3（稀疏瓶颈不变）**：即使取最优 eps，实例召回率仍仅约 0.54——nuScenes 16k 降采样后 **84% 的 thing 实例不足 50 点**，"
     "大量实例整段漏检，这是逐点语义与实例分割共同的物理上限，单纯的特征学习或解冻微调无法突破。", indent=True)

heading("4.10 类别门控级联：修复 barrier 误判", 2)
body("**根因定位**：对 barrier 点做失败分解（500 val 场景），发现 DBSCAN 噪声仅占 2.1%、过合并仅 1%——barrier 聚类本身是干净的，"
     "问题出在**物体级分类器把 barrier 误判成 trailer**：被聚类的 barrier 点中 49.2% 被判为 trailer，而只有 46.1% 判对。barrier"
     "（长条护栏）与 trailer（拖车）都是长条水平结构，居中 + 单位球归一化后形状高度相似，B2 分类器无法区分。而 dense 模型对 "
     "barrier 本身有 0.73，级联用 B2 覆盖 dense 反而把 barrier 从 0.73 拖到 0.40。", indent=True)
body("**修复（类别门控级联）**：对每个聚出的实例，取 dense 逐点预测的多数投票作为门控信号——若多数投票为「连续结构类」（barrier），"
     "则保留 dense 预测、不再经 B2 判类；否则照常 B2 判类。该规则只用模型预测、不依赖真值，可直接部署。结果见表 9。", indent=True, space_after=3)
caption("表 9　类别门控级联修复 barrier 误判")
table(
    ["类", "dense", "无门控级联", "类别门控级联"],
    [
        ["barrier", "0.731", "0.405", "0.828"],
        ["整体（11 类）", "0.502", "0.690", "0.724"],
    ],
    col_widths=[1.6, 1.0, 1.3, 1.4],
)
body("门控后 barrier 由唯一倒退（0.405）转为正向提升（0.828，超过 dense 本身），且 11 类无一倒退，整体 thing 精度 0.690→0.724"
     "（+3.4 点）。这验证了「**级联判类应选择性作用于离散紧凑物体，而非连续长条结构**」这一设计原则。", indent=True)
body("**门控敏感性消融（是否推广到其他类）**：门控收益的本质取决于「该类的 dense 精度是否高于 B2」。为验证 {barrier} 是否就是"
     "最优门控集合，做单类门控扫描（同一轮前向缓存每实例的 dense 多数投票与 B2 预测后，纯后处理评估所有单类门控，表 10）：", indent=True, space_after=3)
caption("表 10　单类门控扫描（门控是否推广到其他类）")
table(
    ["单类门控", "整体 thing 精度", "相对无门控(0.690)"],
    [
        ["无门控（全 B2）", "0.690", "—"],
        ["barrier", "0.731", "+4.1"],
        ["ped.adult / construction / trafficcone / trailer / pushable", "0.687–0.689", "中性偏负（−0.1~−0.3）"],
        ["bus.rigid", "0.685", "−0.5"],
        ["truck", "0.676", "−1.4"],
        ["car", "0.643", "−4.8"],
    ],
    col_widths=[2.6, 1.4, 1.6],
)
body("且 barrier+construction 联合门控（0.730）低于单独 barrier（0.731）。**结论：门控不应推广**——barrier 是唯一 dense（0.73）> "
     "B2（0.40）的类，门控保 dense 才有益；其余 10 类 B2 均优于 dense（construction 的 dense 仅 0.11、远低于 B2 0.64），门控只会"
     "把好的 B2 判类替换成差的 dense，其中 car 伤害最大（dense 0.72 < B2 0.79）。最优门控集合严格等于 {barrier}，这从实验上印证"
     "门控判据是「dense 与 B2 的**逐类精度差**」，而非某种类别先验。（注：本表为同一轮前向缓存的后处理结果，barrier 门控 0.731 与"
     "上表 0.724 的差异为跨轮 crop 采样随机方差，排名一致。）", indent=True)

heading("4.11 跨数据集泛化（Semantic3D）", 2)
body("为验证 RGB+XYZ 早期融合分割前置的跨数据集泛化，在第二真实室外 LiDAR 数据集 Semantic3D（semantic-8，15 训练场景，8 类）上"
     "做了两类实验。", indent=True)
body("**零样本迁移**：nuScenes 32 类模型不做任何再训练，经文档化类别映射（noise→scanning artefacts、vehicle.car→cars、"
     "flat.terrain→natural terrain、static.vegetation→vegetation、static.manmade→buildings、flat.*→man-made terrain/hard "
     "scape）在 Semantic3D 上直接预测（表 11）：", indent=True, space_after=3)
caption("表 11　Semantic3D 零样本迁移")
table(
    ["指标", "值"],
    [
        ["OA（映射后整体准确率）", "0.536"],
        ["mIoU", "0.196"],
        ["映射覆盖率", "96.8%"],
    ],
    col_widths=[2.4, 1.0],
)
body("迁移良好的类：man-made terrain（recall 0.71）、high vegetation（0.76）、buildings（0.56）、natural terrain（0.43）、"
     "cars（0.30）。hard scape、low vegetation 为 0 是标签体系差异（nuScenes 将 high/low vegetation 合并为单一 vegetation 类；"
     "flat.other 与 Semantic3D hard scape 边界不一致），属跨数据集标签固有不可比，而非特征失效。", indent=True)
body("**迁移学习**：冻结 nuScenes 预训练 backbone、仅微调 8 类分割头（可训练 200K / 12.33M 参数，即 1.6%），12 场景训练 / "
     "3 场景验证（表 12）：", indent=True, space_after=3)
caption("表 12　Semantic3D 迁移学习")
table(
    ["组", "best mIoU"],
    [
        ["冻结 backbone + 微调头", "0.263"],
        ["从零训练整模型", "0.224"],
    ],
    col_widths=[2.6, 1.0],
)
body("迁移组仅用 1.6% 可训练参数即超过从零整模型 +3.9 mIoU——即便训练样本仅 12 个场景，nuScenes 学到的几何+颜色特征仍能跨数据集"
     "迁移，验证了 RGB+XYZ 早期融合作为通用室外点云特征前置的价值。", indent=True)

heading("4.12 置信度门控与选择性预测", 2)
body("§4.10 的类别门控把「该用 dense 还是 B2」按类别硬编码，本质上是一种**结构化的选择性预测**。本节将其推广为**通用置信度门控**："
     "对每个 B2 判类实例计算 softmax 不确定度，按不确定度从高到低「拒识」交人工，量化剩余点的精度（拒识曲线），回答「模型何时该说"
     "不知道」这一可信决策问题，也是开集识别与主动学习的落地入口。", indent=True)
body("**方法**：在 500 val 场景上，缓存每个被聚类 thing 实例的 (GT、B2 预测、B2 softmax、dense 逐点预测) 后纯后处理。考察三种 B2 "
     "softmax 不确定度——最大概率 maxp、top1−top2 margin、熵 entropy——以及一条跨模块分歧信号 agree（dense 多数投票是否等于 B2 判类）。"
     "注意本节的基线与 §4.10 略有不同：只统计被聚类的 thing 点（噪声点不计入），故基线为 0.712 而非含噪声点的 0.690。拒识曲线见表 13。", indent=True, space_after=3)
caption("表 13　置信度门控拒识曲线（softmax 不确定度 vs 随机）")
table(
    ["信号", "拒 0%（基线）", "拒 10%", "拒 30%", "拒 50%"],
    [
        ["entropy（熵）", "0.712", "0.748", "0.818", "0.857"],
        ["maxp（最大概率）", "0.712", "0.749", "0.817", "0.855"],
        ["margin（top1−top2）", "0.712", "0.748", "0.816", "0.847"],
        ["random（随机对照）", "0.712", "0.712", "0.712", "0.712"],
    ],
    col_widths=[1.7, 1.2, 0.9, 0.9, 0.9],
)
body("三种 softmax 信号都单调有效（熵略优），且明显高于随机对照的平坦基线——证明 B2 的 softmax 本身校准良好，能可靠区分对错：拒掉"
     "最不确定的 50% 实例，thing 精度从 0.712 提升到 0.857（+14.5 点）；即便只拒 10%，也有 +3.7 点。", indent=True)
body("**跨模块分歧是最强置信度代理**（表 14）：", indent=True, space_after=3)
caption("表 14　跨模块分歧（dense↔B2 一致 vs 分歧）")
table(
    ["子集", "点数", "thing 精度"],
    [
        ["一致（dense 多数投票 == B2 判类）", "234 620", "0.953"],
        ["分歧", "332 389", "0.543"],
    ],
    col_widths=[2.6, 1.0, 1.0],
)
body("当两条独立通路（逐点 dense 分割与物体级 B2 分类）意见一致时，B2 判类几乎全对（0.953）；一旦分歧，B2 精度骤降至 0.543。"
     "这一单比特信号远强于任何单一 softmax 不确定度，因为它抓住了「结构化的、跨模态的不确定」而非「某个 softmax 分布的平坦」。"
     "分歧点占 58.6%，实际部署需按任务的拒识代价设定阈值（全拒会损失一半以上数据），但它天然对应「识别到相似/未知物体，交人工判断」"
     "——与 §3.8 系统导出中的人机协同位正好衔接。", indent=True)
body("**结论**：门控机制可以推广到整模型。用 softmax 置信度 + 跨模块分歧做拒识，把不确定实例交人工，系统即可在「覆盖更广」与"
     "「可靠可靠」之间按需折中；随数据与类别增长，可拒识边界会持续外扩，但「识别所有物体」仍受开集性质限制——置信度门控的意义不是"
     "让系统认识一切，而是让系统可靠地说出「这个我没把握」。", indent=True)

heading("4.13 罕见类的 2D/3D 深度融合：机制上界与可部署效果", 2)
body("逐点分割对 long-tail 罕见类（行人各子类、barrier、trafficcone、bicycle、motorcycle 等 11 类）整体近乎失效（base 罕见类 "
     "mIoU 仅 0.0995），是 mIoU 的主要拖累。这一失败高度集中于**体积小、样本少**的类：7 个行人子类中 5 类 IoU 为 0（child / "
     "personal_mobility / police_officer / stroller / wheelchair）、adult 仅 0.169、construction_worker 0.021，bicycle 0.010、"
     "motorcycle 0.026；而几何尺寸较大的 movable_object.barrier（0.540）与 trafficcone（0.230）其实分割得尚可。本节验证一条互补"
     "路径：**用 2D 图像检测的罕见类框对 3D 点做门控投票回填**，不改变 3D 网络结构，只作为后处理融合。", indent=True)
body("**三种设置：把「机制上界」与「可部署效果」严格分开**（表 15）。S0 = 无融合基线；S1 的 2D 框由 nuScenes 3D **真值标注框**"
     "投影得到（574,241 个）——**推理阶段使用真值属于数据泄漏**，故 S1 仅作 **oracle 上界**，只用于回答「若 2D 框完全正确，这套"
     "机制的天花板在哪」，**不作为方法效果报告**；S2 的 2D 框全部来自相机图像上 **COCO 预训练检测器的零样本预测**，不含任何 "
     "nuScenes 监督信号，是可部署设置，也是本文方法效果的准绳。", indent=True, space_after=3)
caption("表 15　2D 框来源的三种设置与主结果")
table(
    ["设置", "2D 框来源", "可部署", "mIoU(罕见类)", "mIoU(全部)"],
    [
        ["S0", "无融合（3D 基线）", "—", "0.0995", "0.3090"],
        ["S1", "nuScenes 3D 真值标注框投影（574,241 个）", "否（oracle 上界，θ=1.0）", "0.5060", "0.4457"],
        ["S2", "COCO 预训练检测器（RT-DETR-R50）零样本预测", "是（主结果）", "0.1519", "0.3199"],
    ],
    col_widths=[0.7, 2.5, 1.3, 1.1, 1.0],
)
body("S2 取三种深度来源中最佳的 size 门控、θ=0.95。两套设置各自取其扫描最优的 θ（S1 在 θ=1.0 处仍在上升，S2 在 θ≈0.95 "
     "处达峰后回落，见表 17），选取规则与由此带来的 val 集选择偏差已在 §5 局限中说明。相对基线，S2 把罕见类 mIoU 相对提升 53%、全类 mIoU 提升 1.1 个点；但它只兑现了 "
     "oracle 上界增益的 12.9%（(0.1519−0.0995)/(0.5060−0.0995)）。这个落差本身就是本节的主要结论之一，其成因在本节末给出定量归因。",
     indent=True)
body("**方法（S1/S2 共用）**：把 2D 框与 3D 点云投到同一坐标系后，对每个 3D 点：投影到 6 相机（CAM_FRONT / FRONT_LEFT / "
     "FRONT_RIGHT / BACK / BACK_LEFT / BACK_RIGHT）并取并集，若某点落在某罕见类 2D 框内、且点深度 ∈ [d_min−δ, d_max+δ]"
     "（深度门控，δ=0.5 m），则该点获得 2D 类别标签 sem_2d；门控投票只对 3D 置信度 conf_3d < θ 的点用 sem_2d 覆盖 3D 预测。", indent=True)
body("**深度从哪来，是 S1 与 S2 的本质差异**：S1 的 [d_min, d_max] 可直接取 3D 真值框在相机系的角点；而 S2 的 2D 检测器"
     "**不提供深度**，必须自行估计。本文因此比较三种深度来源（表 16）：lidar——框内 LiDAR 投影点的深度分位数 [p05, p90]（只用"
     "真实传感器测距）；size——由框高反推深度的单目高度先验 d̂ = f_y·H_class / h_px（不依赖 LiDAR，只需类别的典型高度常数）；"
     "none——不设深度门控，作为下界对照。", indent=True, space_after=3)
caption("表 16　S2 的三种深度来源（θ=0.95）与关联质量")
table(
    ["深度来源", "罕见类 mIoU", "全类 mIoU", "关联 precision", "罕见类 precision", "罕见类 recall"],
    [
        ["size（单目高度先验）", "0.1519", "0.3199", "0.425", "0.907", "0.170"],
        ["lidar（框内点云分位数）", "0.1472", "0.3182", "0.296", "0.895", "0.182"],
        ["none（无深度门控）", "0.1443", "0.3172", "0.274", "0.894", "0.183"],
        ["（参考）S1 真值框+真值深度（θ=1.0）", "0.5060", "0.4457", "0.701", "0.991", "0.861"],
    ],
    col_widths=[1.6, 1.0, 0.9, 1.1, 1.2, 0.9],
)
body("「罕见类 recall」一列的口径需写明：它是**类别匹配的点级召回**——GT 类 c 的 3D 点必须落在**类别为 c** 的 2D 框内"
     "（而不仅仅是落在某个罕见类框内）才计入分子，分母是 val 集**全部**罕见类 GT 点（1,157,054）。分母必须包含「检测器整帧无输出」"
     "的帧（零样本设置下有 1,201 帧，按零框计入），否则这些必然漏检会被从分母中剔除，使该指标系统性虚高——本文此前报告的 0.192/0.205 "
     "即来自未含这些帧的旧口径，此处已按同一分母更正。注意该列与「关联 precision」是两个不可混用的量：前者是**类别匹配**的召回，"
     "后者是**类别无关**的提案纯度（框内点不论类别的正确比例）。", indent=True, size=9, space_after=3)
body("**深度门控的作用被此前结果高估了，这里如实更正**：在 S1（真值框）设置下，去掉深度门控会让关联 precision 从 0.701 腰斩到 "
     "0.358——因为真值框携带的是物体的完整 3D 范围，框内投影点包含大量物体背后的遮挡背景，不设深度区间就会被错误吸收入框。但在 "
     "S2 的真实检测器设置下，检测框本身就是围绕可见像素的紧致框，框内 LiDAR 回波绝大多数落在物体上：三种深度来源的罕见类 "
     "precision 几乎相同（0.894~0.907），端到端罕见类 mIoU 差距也仅 0.008（0.1443~0.1519）。也就是说，此前「深度门控不可省」的"
     "结论只在 oracle 框条件下成立，在可部署条件下深度估计不是瓶颈——size 这种最朴素的单目先验甚至略优于用 LiDAR 分位数，因为它"
     "给出更窄的深度区间、关联 precision 更高（0.425 vs 0.296）。6 相机并集仍是必要的（单 CAM_FRONT 只见约 12.6% 的点，6 相机达 "
     "86.9%）。", indent=True)
body("**瓶颈是检测器召回，不是融合机制**：上表最后一列给出关键归因——oracle 框能覆盖 86.1% 的罕见类真值点，而零样本检测器只覆盖 "
     "17.0%~18.3%，相差 4.7 倍以上；precision 一侧两者都接近 0.9，说明关联与融合这一环没有问题，漏检才是 S1→S2 落差的主因。这一"
     "结论把改进方向从「更好的深度估计／融合算法」明确转向「更高的检测器召回」，也正是后续用 nuScenes train split 微调检测器的直接"
     "依据。", indent=True)
body("**θ 门控阈值扫描**（g256 模型 + 6 相机；θ 越大越信任 2D，表 17）。两套设置共用同一扫描：S1 随 θ 单调上升，因为真值框不会"
     "误报；S2 在 θ≈0.95 处达峰后回落，因为 θ 越大、检测器的误检被采纳得越多：", indent=True, space_after=3)
caption("表 17　θ 门控阈值扫描：oracle 上界 vs 可部署实际")
table(
    ["θ", "S1 oracle 罕见类", "S1 oracle 全类", "S2 可部署 罕见类", "S2 可部署 全类"],
    [
        ["0.5", "0.2071", "0.3453", "0.1159", "0.3142"],
        ["0.6", "0.2780", "0.3691", "0.1267", "0.3174"],
        ["0.7", "0.3414", "0.3904", "0.1351", "0.3195"],
        ["0.8", "0.3971", "0.4092", "0.1423", "0.3207"],
        ["0.9", "0.4512", "0.4273", "0.1488", "0.3211"],
        ["0.95", "0.4857", "0.4389", "0.1519", "0.3199"],
        ["1.0", "0.5060", "0.4457", "0.1517", "0.3144"],
    ],
    col_widths=[0.7, 1.5, 1.4, 1.5, 1.4],
)
body("S1 单类亮点（θ=1.0）：ped.police_officer 0→0.741、ped.adult 0.169→0.614、ped.construction_worker 0.021→0.574、"
     "motorcycle 0.026→0.565、ped.stroller 0→0.490、trafficcone 0.230→0.475、bicycle 0.010→0.446。S2 的增益则精确地落在 COCO "
     "能命名的三类上（θ=0.95）：ped.adult 0.169→0.375、motorcycle 0.026→0.185、bicycle 0.010→0.168；而 barrier 0.540→0.541、"
     "trafficcone 0.230→0.230（COCO 无对应类，纹丝不动），7 个行人子类中除 adult 外全部不变（--group-refine 只让 2D 判定「是不是"
     "人」，具体子类仍交给 3D，故不会因粗粒度映射而误伤子类）。这一逐类分布与「COCO 覆盖边界」的预测完全一致，是 S2 结果可信度的"
     "一个旁证。", indent=True)
body("**覆盖边界（必须如实说明）**：COCO 80 类中与本文 11 个罕见类语义对应的只有 person / bicycle / motorcycle 三类，且 COCO 的 "
     "person 只到「人」这一粒度，对应 nuScenes 的 7 个行人子类——因此 S2 评测时用 2D 判定「是不是人」、再由 3D 在 2..8 内细化具体"
     "子类（关闭该选项的对照仅为 0.1512 vs 0.1519，说明该取舍对结果不敏感）。**barrier 与 trafficcone 在 COCO 中没有对应类，零样本"
     "设置下无法覆盖**，这是 S2 与 S1 差距的来源之一；另一来源是前述的检测器召回（0.18 vs 0.861）。两者共同指明了后续方向：用 "
     "nuScenes train split 的 3D 真值框投影出 2D 标签来微调检测器（训练阶段用真值标注是标准的有监督做法，不构成泄漏），既可补齐 "
     "barrier / trafficcone 两类，也可显著提高召回。**§4.15 完成了这一实验**，并发现该方向的收益需附加条件才能兑现。", indent=True)
body("**结论**：3D 逐点分割对长尾罕见类整体难以学到（11 类中 10 类的 IoU ≤0.23，7 个行人子类里 5 类为 0），而 2D 图像检测对同类"
     "物体判别力强。「2D 检测 + 深度门控 + 置信度投票」在 oracle 条件下把罕见类 mIoU 从 0.0995 提升到 0.5060（5.1 倍），给出该融合"
     "机制的增益上界；在完全可部署的零样本设置下提升到 0.1519（相对 +53%），兑现上界增益的 12.9%。这一落差不应被读作融合机制的"
     "失败：关联 precision 在两种设置下都接近 0.9，瓶颈明确落在检测器召回（0.18 vs 0.861）与 COCO 的类别覆盖上。换句话说，本文"
     "验证的是「机制可行且上界可观」，而「可部署效果」受限于当前零样本检测器这一外部组件。**§4.15 按此方向完成了微调实验，并据此"
     "把上述归因修正为更严格的形式**：检测器召回可以提高 4.3 倍、barrier / trafficcone 可以从严格 0 补齐，但端到端收益只有在"
     "**注入标签的点级精度同时提高**时才兑现——在默认操作点下罕见类 mIoU 反而下降；且即便把操作点提到让检测器在召回与精度上同时"
     "优于零样本，**融合规则本身缺少逐点可信度判断**仍是最后一层约束。", indent=True)

# ======================================================================
# 4.14 罕见类失效的归因
# ======================================================================
heading("4.14 罕见类失效的归因：采样密度、分组粒度与标注量", 2)
body("§4.2 与 §4.13 都指出：3D 逐点分割在长尾罕见类上近乎全灭（11 个罕见类中 10 类的 IoU ≤0.23，7 个行人子类中 5 类恒为 0）。"
     "一个自然的解释是「输入点云太稀疏，采样密度不够」。本节用受控实验把三种候选成因分开——**采样密度**、**分组粒度（token 容量）**、"
     "**标注量与场景出现频率**——结论是：**采样密度只是门槛型因素，主因是分组粒度；而对其中 5 个类，决定因素是标注量，与采样无关。**",
     align="justify")
body("为便于讨论，先定义一个与模型无关的度量。**点加权可达性 R_c(K)**：类别 c 的 GT 点中，落在「点数 ≥ K 的实例」上的比例。由于"
     "任何逐点分割器对一个实例的最好结果就是把它的点全部判对，把「点数 < K 的实例」视为不可分时，R_c(K) 就给出了该类逐点 IoU 的可达"
     "部分。它只是采样密度的函数，与网络结构和训练策略无关，因此可以作为归因的参照系。", align="justify")

heading("4.14.1 采样密度是门槛型因素，不是主因", 3)
body("先做一组最干净的对照：取 §4.2 的 32768 点 / num_group=1024 模型的**同一组权重**，只在**推理时**改变输入点数（数据侧随机"
     "子采样），其余完全不变（表 18）：", indent=True, space_after=3)
caption("表 18　同一组权重下仅改变推理输入点数（罕见类 = 11 个 rare thing 类去 val 无 GT 者）")
table(
    ["推理输入点数", "罕见类 mIoU", "全类 mIoU", "OA"],
    [
        ["32768（训练同密度）", "0.1924", "0.3697", "0.9266"],
        ["16384", "0.1873", "0.3582", "0.9239"],
        ["4096", "0.0650", "0.2035", "0.8446"],
    ],
    col_widths=[1.7, 1.3, 1.2, 1.1],
)
body("32768→16384 使罕见类 mIoU 只变化 **−0.005**（全类 −0.012），而 16384→4096 则骤降 **−0.122**（全类 −0.155）。也就是说，"
     "**采样密度在 16k 以上已经饱和，在 4k 以下才崩塌，响应呈门槛型而非梯度型**。这解释了 §4.2 中「升点数只带来有限边际增益」的"
     "负结果：16384 点已经越过门槛。", indent=True)

heading("4.14.2 分组粒度（token 容量）才是主因", 3)
body("把点数**固定在 16384 不变**，只改分组方式，得到决定性的对照（表 19）：", indent=True, space_after=3)
caption("表 19　点数固定为 16384、仅改变分组方式")
table(
    ["配置", "输入点数", "分组", "训练轮数", "罕见类 mIoU", "全类 mIoU"],
    [
        ["A", "16384", "num_group=64，FPS/kNN", "200（跑满）", "0.0676", "0.2323"],
        ["B₁", "16384", "num_group=1024，GridGroup 0.5 m", "21（过热中断）", "0.1736", "0.3606"],
        ["B₂", "16384", "num_group=1024，GridGroup 0.5 m", "40（跑满）", "0.2145", "0.3923"],
        ["C", "32768", "num_group=1024，GridGroup 0.5 m", "200（跑满，best@145）", "0.1924", "0.3697"],
    ],
    col_widths=[0.5, 0.8, 2.1, 1.2, 1.1, 1.0], font_size=8.5,
)
body("A→B₂ 输入点数**一个都没有增加**，罕见类 mIoU 却提升 **3.17 倍**（0.0676→0.2145）、全类提升 0.1600。B₁ 是 B₂ 在过热中断前的"
     "快照，二者给出同一条训练曲线上的两点，用于说明 B₂ 的结果不是配置偶然。", indent=True)
body("关键在于把 B₂ 与 C 相比：**C 把输入点数翻倍（16384→32768）、且训练轮数是 B₂ 的 5 倍（200 轮 vs 40 轮），两项指标却都更低**"
     "——罕见类 0.1924 对 0.2145（−0.0221）、全类 0.3697 对 0.3923（−0.0226）。也就是说，**在 token 预算同为 1024 时，点数翻倍不仅"
     "没有增益，反而有害**；而 A→B₂ 在点数完全不变的前提下靠改变 token 化就拿到了 3.17 倍。两相对照可以判断：**罕见类的失效首先"
     "是 token 化问题，不是点数问题。**", indent=True)
body("这一结论曾因证据不足而被保留：早期只有 B₁（21 轮，过热中断），与 C（200 轮）之间存在 21 vs 145 的训练轮数混淆，因此当时只能"
     "主张\"B 在不到 1/7 训练量下达到 C 的 90%\"，并把\"点数翻倍贡献 +0.019\"标记为待核实。B₂ 把该配置跑满 40 轮后，混淆被消除，且"
     "结论方向反转：点数翻倍的真实效果是 −0.022，而非 +0.019。原先那个正值是低训练轮数造成的低估——B₁ 尚未收敛（0.1736 → 0.2145）。"
     "两组数据均以 eval_seg_full.py 在完整 val 集上评测，与训练内 val 同口径（评测值与训练内 best 逐位吻合，差 ≤0.00003）。", indent=True)
body("机理是清楚的。FPS/kNN 分组下每个 token 覆盖场景中一大片连续区域，一个行人（约 0.5 m 宽）的十几个点会被并入周围地面或墙面的"
     "token，其独立语义在序列建模之前就已经丢失；而 0.5 m 体素 + 1024 token 的分组让行人量级的物体**拥有自己的 token**，语义得以"
     "进入 Mamba 序列。这也与 §4.2 已定位的「瓶颈主要是 token 数」互为印证——只是那里的证据来自全类 mIoU，这里在罕见类维度上"
     "独立复现了一次。", indent=True)
body("需要说明的是，配合 GridGroup 的 num_group 还带有一个实现细节：当**体素数超过 num_group** 时，代码会对体素中心做 FPS 选出 "
     "num_group 个 token，并把落选的体素**并入最近的保留体素**。因此在 token 预算固定时，单纯缩小 grid_size 会被这一步合并抵消——"
     "这提示真正的杠杆是 token 预算本身而非体素边长，也说明「让每个行人拥有独立 token」的必要条件是 **token 预算与场景中的物体"
     "数量同量级**。", indent=True)

heading("4.14.3 采样收益的分配极端不均匀", 3)
body("既然点数并非主因，它究竟改善了什么？统计每个实例的点数分布（表 20，val 集抽样 2991 帧）：", indent=True, space_after=3)
caption("表 20　每实例点数随采样密度的变化（val 集抽样 2991 帧）")
table(
    ["类别", "密度", "实例数", "点数/实例 中位", "点数/实例 均值", "≥20 点实例", "≥50 点实例"],
    [
        ["ped.adult", "4096", "13828", "2", "2.1", "0", "0"],
        ["ped.adult", "16384", "14096", "4", "8.4", "1526", "322"],
        ["ped.adult", "34720", "14096", "**4**", "**12.9**", "1789", "676"],
        ["ped.child", "16384", "59", "4", "4.8", "0", "0"],
        ["ped.child", "34720", "59", "4", "5.7", "3", "0"],
        ["motorcycle", "16384", "1101", "5", "18.9", "223", "105"],
        ["motorcycle", "34720", "1101", "5", "33.0", "229", "134"],
    ],
    col_widths=[1.1, 0.7, 0.8, 1.1, 1.1, 1.0, 1.0], font_size=8.5,
)
body("三点值得注意：**其一，实例数完全不随密度变化**（ped.adult 14096→14096、barrier 11088→11088、bicycle 980→980）——物体个数"
     "是场景的固有属性，逐点标注的「标注量」因此只体现在每实例点数上。**其二，密度提升几乎全部落在均值而非中位数上**：ped.adult 的"
     "每实例点数均值 2.1→8.4→12.9（6 倍），中位数却只有 2→4→4。**其三，受益的是本就大的实例**：≥50 点的行人实例从 322 增到 676"
     "（2.1 倍），而 ≥20 点的只从 1526 增到 1789（1.17 倍），中位数以下的那一半实例（占实例总数 50%）**完全不受益**。", indent=True)
body("这解释了为什么加大采样救不了小目标：新增的点按比例落在原本就点多的近处大物体上，远处的小实例从 3 个点变成 4 个点，仍然不"
     "构成任何可辨的局部形状。", indent=True)

heading("4.14.4 恒零类之一：标注量决定能不能学到", 3)
body("最后是那些**在所有配置、所有密度下 IoU 都恒为 0** 的类。表 21 给出它们的场景出现频率、实例数、点加权可达性与最佳 IoU：",
     indent=True, space_after=3)
caption("表 21　罕见类的标注量、可达性与最佳 IoU（按出现帧占比升序，val 集抽样 2991 帧）")
table(
    ["类别", "出现帧占比", "val 实例数", "点加权可达性 R(20)", "最佳 IoU"],
    [
        ["ped.wheelchair", "0.0%", "0", "—", "0（val 无 GT）"],
        ["ped.personal_mobility", "0.1%", "2", "0.000", "0.000"],
        ["**ped.police_officer**", "**1.3%**", "**38**", "**0.765**", "**0.000**"],
        ["ped.child", "1.3%", "59", "0.257", "0.000"],
        ["ped.stroller", "1.6%", "51", "0.248", "0.000"],
        ["ped.construction_worker", "7.7%", "585", "0.624", "0.139"],
        ["vehicle.bicycle", "22.6%", "980", "0.563", "0.076"],
        ["vehicle.motorcycle", "24.0%", "1101", "0.883", "0.210"],
        ["movable_object.barrier", "31.0%", "11088", "**0.928**", "**0.687**"],
        ["movable_object.trafficcone", "38.1%", "6096", "0.563", "0.361"],
        ["ped.adult", "75.0%", "14096", "0.692", "0.452"],
    ],
    col_widths=[2.1, 1.0, 0.9, 1.4, 1.0], font_size=8.5,
)
body("**ped.police_officer 是关键证据**：它的点加权可达性 0.765 **比 ped.adult 的 0.692 还高**——按点数密度衡量，警员比普通行人"
     "更「看得清」，却 IoU 恒为 0.000。可达性高而 IoU 为零，只能归因于**类别先验没有建立**：它在 val 仅 38 个实例、出现在 1.3% "
     "的帧中（按比例推算训练集约 357 个实例）。反方向的对照同样干净：barrier 的实例数 11088（约为警员的 290 倍）、可达性 0.928，"
     "IoU 达到 0.687，是全部罕见类中最高的。**实例数低于约 100（出现帧占比低于约 2%）的类，无论怎样提高采样密度都是 0**。这一点由"
     "两组「**可达性更高、IoU 反而更低**」的对照坐实：ped.police_officer 的可达性 0.765 高于 ped.adult 的 0.692，IoU 却是 0.000 对 "
     "0.452；它也高于 ped.construction_worker 的 0.624，IoU 是 0.000 对 0.139。可达性更高而 IoU 更低，能解释差别的变量只剩实例数"
     "（38 对 14096 / 585）。", indent=True)

heading("4.14.5 恒零类之二：用地面作参考的几何可分性检验", 3)
body("上一节说明标注量决定「学没学到」，但它回答不了另一个问题：**这些类在物理上到底有没有可区分的特征？** 二者的实践含义完全不同"
     "——若特征存在，则增加监督是有效的干预方向；若特征不存在，则再多标注也无济于事，只能引入外部模态（如相机外观）。为此本文以"
     "地面为参考，测量每个实例的离地高度与水平尺寸：每帧用真值地面类（driveable / flat.other / sidewalk / terrain）迭代最小二乘"
     "拟合平面 z = ax + by + c（两轮残差裁剪抗离群），再统计实例点在平面之上的高度分布与 XY 主成分两轴跨度"
     "（`scripts/rare_geometry_probe.py`，6019 帧 val）。结果见表 22。", indent=True, space_after=3)
caption("表 22　以地面为参考的实例几何可分性检验（6019 帧 val，33 351 个实例，每实例 ≥5 点）")
table(
    ["类别", "val 实例数", "离地高度中位(m)", "竖直跨度中位(m)", "水平长轴中位(m)", "水平短轴中位(m)"],
    [
        ["ped.adult（参照）", "12380", "1.50", "1.09", "0.47", "0.26"],
        ["**ped.police_officer**", "55", "**1.62**", "**1.08**", "**0.50**", "**0.27**"],
        ["ped.child", "60", "1.20", "0.68", "0.34", "0.16"],
        ["ped.stroller", "25", "0.88", "0.45", "0.49", "0.27"],
        ["ped.personal_mobility", "0", "—", "—", "—", "—"],
        ["ped.wheelchair", "0", "—", "—", "—", "—"],
        ["movable_object.barrier（对照）", "15170", "0.76", "0.54", "2.12", "0.19"],
        ["movable_object.trafficcone（对照）", "3612", "0.89", "0.67", "0.25", "0.12"],
    ],
    col_widths=[1.9, 0.9, 1.1, 1.1, 1.1, 1.1], font_size=8.5,
)
body("两点关于口径的说明。其一，本表与表 21 的实例数口径不同、**不可直接比较**：表 21 来自 instance_size_hist.py，在 stride-2 抽样"
     "（2991 帧）、64k 全分辨率下统计**所有**实例（≥1 点）；本表来自 rare_geometry_probe.py，在**全部 6019 帧**、16k 子采样下只统计"
     "「≥5 个点」的实例（并舍去地面拟合失败的帧）。两个口径的差异方向并不单一：帧数翻倍（2991→6019）会**增加**计数（police 38→55、"
     "barrier 11088→15170），而 16k 子采样与 ≥5 点门槛会**减少**计数（stroller 51→25、trafficcone 6096→3612、adult 14096→12380）"
     "——前者主导较大实例、后者主导较小实例。其二，本表使用**真值地面类别**拟合平面，是无监督地面分割条件下的上界；可部署版本需替换"
     "为 Patchwork++ 一类的无监督地面分割。", indent=True)
body("表 22 把恒零类清晰地劈成三种情况：", indent=True)
body("**ped.police_officer 是几何上不可分的。** 它的四项几何量——离地高度 1.62 m、竖直跨度 1.08 m、水平长轴 0.50 m、短轴 0.27 m"
     "——与 ped.adult 的 1.50 / 1.09 / 0.47 / 0.26 **逐项重叠**。也就是说，警员在 LiDAR 几何上与普通行人没有可区分的差异；其失效是"
     "**双重决定**的：既缺监督（§4.14.4），又缺判别信号。因此对 police 而言，增加标注量是必要但**不充分**的，唯一的出路是引入外部"
     "模态——这正是 §4.13 的 2D/3D 门控投票所能提供的，其 oracle 设置能把 police 从 0 提升到 0.741。", indent=True)
body("**ped.child 与 ped.stroller 是几何可分、仅监督不足的。** child 的中位离地高度 1.20 m、竖直跨度 0.68 m，相对 ped.adult 的 "
     "1.50 m 与 1.09 m 有明显差异；stroller 更低（0.88 m）且竖直跨度仅 0.45 m，与行人的直立轮廓区分更大。**判别信息真实存在于输入"
     "之中**，它们的失效源于监督量（val 仅 797 与 429 个 GT 点），而非表征能力或几何不可分所致。这一区分的直接"
     "方法学含义是：**对 child / stroller，实例级增广、类别平衡损失等增加监督的手段是有效干预方向**，而不必像 police 那样依赖外部"
     "模态。§4.13 的 oracle 设置同样能把 stroller 从 0 提升到 0.490。", indent=True)
body("**ped.personal_mobility 与 ped.wheelchair 在 val 中不可测。** 二者的实例数在「≥5 个点」的口径下均为 0（前者全 val 仅 5 个 "
     "GT 点，后者为 0）。它们的 IoU 为 0 不构成模型失败的证据，本文**不对其可学性下结论**；这也意味着任何声称「已解决」或「无法"
     "解决」这两类的说法在 nuScenes val 上都缺乏统计基础。", indent=True)
body("**干预实验（P3）：增加监督只兑现 recall、不兑现 IoU。** §4.14.4 的「监督不足」与本节的三类分工都可以用一个直接干预来检验："
     "若失效确因监督量不足，则**人为抬高这些类的监督权重**应当让它们的召回显著上升，即便 IoU 未必同步。为此训练 P3 配置——把逐点"
     "交叉熵换成**逆平方根频率加权的类别平衡损失**（权重下限 0.25、上限 10），并对含罕见类的帧做过采样（police ×17.2、child ×8.8、"
     "stroller ×12.3、personal_mobility ×17.1），其余超参与 P1 完全相同、训练 40 轮。以「该类 GT 点被预测为自身」的比例（混淆矩阵"
     "对角线，即 **recall**，--dump_confusion 统计）衡量模型是否「学到」，结果与预言高度一致，但方向比「单纯监督不足」更精细：",
     indent=True)
body("• **recall 全面上升**：construction_worker 0.2985→**0.4346**（+13.6pp）、bicycle 0.1085→**0.2844**（+17.6pp）、"
     "trafficcone 0.4989→**0.6931**（+19.4pp）、motorcycle 0.3687→0.4073、adult 0.5952→0.6312；child 由精确 0 抬到 0.0088、"
     "personal_mobility 由 0 抬到 0.40（5 点中 2 点）——**脱离精确零**。", indent=True)
body("• **IoU 不升反降**：全类 mIoU 0.3923→**0.3622**（−0.030），child 的 IoU 几乎不动（0→0.0024），construction_worker 的 IoU "
     "甚至 0.1815→0.1250。原因是平衡损失把预测向罕见类偏移，recall 上升的同时引入更多误报——**precision 下降抵消了 recall 上升**。",
     indent=True)
body("• **安全指标净改善**：致命误判率（lethal 点被预测成安全/可通行/可碾压类）5.469%→**4.842%**（−0.63pp），因为 "
     "construction_worker、bicycle、motorcycle、trafficcone 等致死/致损类的召回上升，把原本落入「安全」桶的 lethal GT 点捞了回来。",
     indent=True)
body("三点结论。(1) **「监督不足」是 recall 层面的真实主因**：几乎所有罕见类的 recall 都被这一干预拉高，排除了「这些类在表征上"
     "完全不可学」的可能。(2) **但增加监督是必要条件而非充分条件**：它让模型能「说出」这些类，却不能「说准」实例边界，故 IoU 不"
     "兑现——纯 3D 增加监督的天花板有限，**兑现 recall 收益、把 IoU 拉上去仍需 §4.13/§4.15 的 2D 外部模态**（oracle 把 police "
     "0→0.741、stroller 0→0.490）。(3) **三类分工被独立复现**：police 的 recall 与 IoU 双双保持 0，坐实「几何不可分、监督非"
     "瓶颈」；stroller 的 recall 仍为 0，说明对这类最低召回、最易混成 manmade 的类，监督增加也不够；child / personal_mobility 脱离"
     "精确 0，确认它们是「仅监督不足」。", indent=True)

heading("4.14.6 成因分工与可救性判据", 3)
caption("表 23　§4.14 罕见类失效的成因分工与可救性判据汇总")
table(
    ["成因 / 判据", "作用", "证据", "可否通过加采样/加算力解决"],
    [
        ["标注量与场景出现频率", "决定**能不能学到**（不可约下限）",
         "police 可达性 0.765 而 IoU 恒 0；实例数 <100 的类全部为 0", "否"],
        ["分组粒度 / token 容量", "决定**学到了能兑现多少**",
         "同为 16384 点，g64→g1024 使罕见类 0.0676→0.2145（3.17 倍）；点数翻倍（16k→32k）则 −0.0221", "是（改 tokenization，成本低）"],
        ["采样密度", "门槛型，≥16k 饱和",
         "同权重推理 32768→16384 仅 −0.005，16384→4096 则 −0.122", "部分（过门槛即可）"],
        ["几何可分性（§4.14.5）", "决定**值不值得救**：特征不存在时加监督无效",
         "police 四项几何量与 ped.adult 逐项重叠；child / stroller 明显可分", "否（须引入外部模态）"],
    ],
    col_widths=[1.5, 1.4, 2.4, 1.4], font_size=8.5,
)
body("这一分工对方法设计有三点直接含义。其一，本文选择 16384 点作为主实验密度是合理的，但正确的理由是**它已越过饱和门槛**，而非"
     "「点数越多越好」——继续加点的收益应优先让给 token 容量。其二，**提高采样密度无法改善小目标**，因为新增的点按比例落在近处大"
     "物体上；要救小目标必须改变 tokenization（网格粒度）或引入实例级重采样/增强。其三，恒零的 5 个类必须分三种情况区别对待，"
     "**笼统称为「不可约」是不准确的**：(a) **ped.police_officer 是几何不可分的**（表 22），即便标注量充足，纯 LiDAR 逐点范式也无法"
     "把它与普通行人分开，只能引入外部模态——这正是 §4.13 的 2D/3D 门控投票所能提供的，其 oracle 设置能把 police 从 0 提升到 0.741；"
     "(b) **ped.child 与 ped.stroller 是几何可分、仅监督不足的**，判别信息真实存在于输入中，因此实例级增广、类别平衡损失等增加监督"
     "的手段是有效干预方向，§4.13 的 oracle 设置同样能把 stroller 从 0 提升到 0.490；P3 干预实验已部分证实这一点（监督增加使 "
     "child/personal_mobility 脱离精确 0、construction_worker recall +13.6pp，但 IoU 未同步兑现，见 §4.14.5 末）；(c) "
     "**ped.personal_mobility 与 ped.wheelchair "
     "在 val 中不可测**（≥5 点实例数为 0），本文不对其可学性下结论。", indent=True)

# ======================================================================
# 4.15 用真值框投影微调检测器
# ======================================================================
heading("4.15 用真值框投影微调检测器：召回补齐、操作点与融合规则", 2)
body("§4.13 把 S1（oracle 真值框）与 S2（零样本检测器）的落差归因于**检测器召回**（0.18 vs 0.861）与 **COCO 的类别覆盖边界**"
     "（barrier / trafficcone 在 COCO 中无对应类，零样本下纹丝不动）。本节直接检验这一归因：用 nuScenes **train split** 的 3D 真值"
     "框投影出 2D 标签，对检测器做有监督微调，补齐这两类并提高召回，看端到端融合能否随之提升。训练阶段使用真值标注是标准的有监督做法；"
     "评测仍在 val split 上、且与 S2 使用**完全相同**的融合机制与评测口径，**不构成泄漏**。", indent=True)
body("**设置**。标签为 nuScenes train split 的 3D 真值框投影到 6 相机的 891,167 个 2D 框（投影时已裁到画幅内，实测 0% 越界）。"
     "RT-DETR-R50 由 COCO 80 类扩到 **82 类**，新增 barrier(80) / trafficcone(81)；类别不平衡用 RepeatFactorSampler"
     "（target_per_class=4000）与 VFL 正样本类别加权（class_boost=4.0）缓解。训练 16 轮 × 12,000 样本，lr 2e-5（主干 1/10），bs 8；"
     "checkpoint 按留出集上的关注类均值选取。", indent=True)
body("微调中出现过一个反直觉的失败，这里如实记录，因为它决定了本节结果能否复现：**dev 召回从 66/170 崩到 3/170，而训练损失反而从 "
     "30 降到 19**——即「损失在降、模型在废」。逐臂对照排除了学习率、翻转、类别加权与冻结策略（把学习率减半只改变崩塌速度、不改变方向；"
     "把框损失权重归零、几何全程不变，照样崩），最终定位到所用 transformers 版本的 RT-DETR 损失函数**在主分支未切掉去噪 query**："
     "开启对比去噪训练时 decoder 实际运行 num_denoising + num_queries 个 query，而该函数对主分支的 logits 未做切分，却对 aux 分支"
     "做了切分。逐分支统计实测到后果——**主分支的匈牙利匹配把 90~100% 的正样本配给了去噪 query**（六个 batch 分别 48/55、43/43、"
     "53/54、48/53、41/43、51/53）。去噪 query 的参考点就是「加噪后的真值框」，匹配时几乎必胜，于是推理时真正被使用的 300 个真实 "
     "query 在训练中被大面积当作背景压制。在主分支按真实 query 切分后，**去噪保持开启反而优于关闭去噪**（81/170 vs 77/170），说明"
     "去噪本身有益，问题只在那一处未切分。", indent=True)
body("**检测侧：召回确实被补齐了**（表 24）。两侧跑同一份 val GT（nuScenes val split 的 3D 真值框 2D 投影，79,172 个目标框）、"
     "同一 score 阈值 0.35 与 IoU 阈值 0.5。分母取**全部 GT 帧**——检测器整帧无输出时按零检测计入，否则这些必然漏检会被从分母中"
     "剔除而系统性虚高召回。", indent=True, space_after=3)
caption("表 24　检测侧召回与精度（score 阈值 0.35，IoU 阈值 0.5）")
table(
    ["类别", "GT 框数", "零样本 COCO 召回", "微调后召回", "零样本 precision", "微调后 precision"],
    [
        ["person", "34,591", "0.3316", "**0.6223**", "0.3953", "**0.5056**"],
        ["bicycle", "2,382", "0.2800", "**0.5823**", "0.2634", "0.2640"],
        ["motorcycle", "2,514", "0.1842", "**0.6010**", "0.3884", "0.2901"],
        ["**barrier**", "26,215", "**0.0000**", "**0.7546**", "—", "0.2510"],
        ["**trafficcone**", "13,470", "**0.0000**", "**0.6705**", "—", "0.2664"],
        ["关注类平均", "—", "0.1160", "**0.6521**", "—", "—"],
    ],
    col_widths=[1.2, 0.8, 1.2, 1.0, 1.2, 1.1],
)
body("零样本下 barrier 与 trafficcone 的召回是**严格 0**（不是「很低」）：COCO 80 类中没有对应类名，检测器根本不会输出这两个标签"
     "的任何框——这正是 §4.13「COCO 覆盖边界」的直接证据。微调后不仅补齐了这两类（0.75 / 0.67），COCO 中**本就有对应类名**的三类也"
     "大幅提升（person 0.33→0.62、motorcycle 0.18→0.60），说明此前的低召回不只是「类名缺失」，零样本检测器在 nuScenes 这个域上的"
     "整体召回本就很低，域内微调同时解决了这两个问题。检测器输出框数从 32,752 增至 165,922；val 集上「整帧无检测」的帧从 1,153 降到 "
     "148。", indent=True)
body("口径：召回分母是**全部 GT 帧**（含检测器无输出的帧，按零检测计）。若只统计「检测器有输出的帧」，person 召回会被虚高到 0.63 "
     "以上。precision 一列中 barrier / trafficcone 在零样本下无定义（分母为 0）。", indent=True, size=9)
body("**端到端：融合效果反而下降**（表 25）。把微调后的检测框接入**完全相同**的融合管线（同一 3D 分割权重、同一配置、同一 θ 扫描、"
     "同一 --group-refine），罕见类 mIoU **没有提升，反而下降**：", indent=True, space_after=3)
caption("表 25　端到端融合（微调后 vs S2 零样本）")
table(
    ["深度门控", "S2 零样本（最佳 θ）", "微调后（最佳 θ）", "变化"],
    [
        ["lidar", "0.1472 (θ=0.95)", "0.1213 (θ=0.8)", "**−0.0259**"],
        ["none", "0.1443 (θ=0.95)", "0.1181 (θ=0.7)", "**−0.0262**"],
        ["size", "0.1519 (θ=0.95)", "0.1427 (θ=0.9)", "**−0.0092**"],
    ],
    col_widths=[1.0, 1.6, 1.6, 1.0],
)
body("两侧 base 均为 0.0995，即融合在微调后仍然有效、只是增益变小。**这不是因为框变差了**——恰恰相反，微调后的框在每一个关联指标"
     "上都优于零样本（表 26）：", indent=True, space_after=3)
caption("表 26　关联指标（lidar 门控，6 相机）")
table(
    ["关联指标（lidar 门控，6 相机）", "S2 零样本", "微调后"],
    [
        ["点级类别匹配覆盖（匹配recall）", "0.182", "**0.778**"],
        ["点级 precision", "0.296", "0.289"],
        ["框级关注类平均召回", "0.1160", "**0.6521**"],
        ["平均框数 / 帧", "6.7", "27.8"],
    ],
    col_widths=[2.6, 1.1, 1.1],
)
body("即：**召回提高了 4.3 倍，端到端融合却更差**。逐类拆开看（表 27，lidar 门控，6 相机；「注入」= 该类的 2D 标签覆盖了 3D 预测）：",
     indent=True, space_after=3)
caption("表 27　逐类关联与融合 IoU（lidar 门控，6 相机）")
table(
    ["类别", "S2 匹配recall", "S2 精度", "微调后匹配recall", "微调后精度", "融合 IoU（S2→微调后）"],
    [
        ["ped.adult", "0.769", "0.302", "0.851", "**0.217**", "0.3127 → 0.2540 ↓"],
        ["bicycle", "0.579", "0.151", "0.703", "**0.106**", "0.1674 → 0.1334 ↓"],
        ["motorcycle", "0.392", "0.422", "0.564", "**0.152**", "0.1996 → 0.1271 ↓"],
        ["barrier", "0.000（从不注入）", "—", "0.787", "0.386", "0.5413 → 0.4849 ↓"],
        ["trafficcone", "0.000（从不注入）", "—", "0.696", "0.143", "0.2299 → 0.1974 ↓"],
    ],
    col_widths=[1.0, 1.0, 0.8, 1.0, 0.8, 1.4],
)
body("微调后**在每一类上都是「召回更高、点级精度更低」**。前一张表中「总体 precision 几乎没变」（0.296→0.289）是聚合造成的假象——"
     "新增的 barrier 精度较高（0.386）把均值抬平了，按类拆开后无一例外更低。而**每一类的融合 IoU 变化都与其注入标签的点级精度同向**："
     "motorcycle 的点级精度掉得最狠（0.422→0.152），融合 IoU 也掉得最狠（−0.072）；barrier 是最干净的一例——零样本设置对它在 COCO 中"
     "无类名、**从不注入**，3D 自身的 0.5413 因此完整保留；微调后开始以 38.6% 的精度注入，IoU 即降到 0.4849。", indent=True)
body("**决定性验证一：提高检测器操作点，两个门控给出相反结论**（表 28）。上述结果提示约束可能不是召回而是**注入标签的点级精度**，"
     "而精度低只因操作点定得太松——score 阈值 0.35 是在检测阶段定死的，且检测分数在建 bbox 缓存时被丢弃（缓存 8 列为 "
     "[x,y,w,h,cls,dmin,dmax,cam_idx]，无 score），融合阶段无法再取舍。为验证，在 0.5 / 0.7 两个阈值上重建 bbox 缓存、重跑完全相同"
     "的融合管线：", indent=True, space_after=3)
caption("表 28　提高检测器操作点的端到端效果")
table(
    ["门控", "检测 score 阈值", "点级匹配recall", "点级 precision", "mIoU罕见类（最佳 θ）", "mIoU全类"],
    [
        ["lidar", "0.35", "0.778", "0.289", "0.1213 (θ=0.8)", "0.3157"],
        ["lidar", "0.5", "0.763", "0.346", "0.1396 (θ=0.9)", "0.3222"],
        ["lidar", "0.7", "0.669", "0.390", "0.1456 (θ=0.9)", "0.3245"],
        ["size", "0.35", "0.727", "0.382", "0.1427 (θ=0.9)", "0.3223"],
        ["size", "0.5", "0.707", "0.454", "0.1672 (θ=0.95)", "0.3316"],
        ["size", "**0.7**", "0.614", "**0.512**", "**0.1731 (θ=1.0)**", "**0.3336**"],
        ["（对照）S2 零样本 lidar", "0.35", "0.182", "0.296", "0.1472 (θ=0.95)", "0.3182"],
        ["（对照）S2 零样本 size", "0.35", "—", "—", "0.1519 (θ=0.95)", "0.3199"],
    ],
    col_widths=[1.5, 0.9, 1.0, 1.0, 1.4, 1.0],
)
body("**结论是门控依赖的、且两者相反**：", indent=True)
body("• lidar 门控：提高操作点把 −0.0259 的落差收回 **+0.0243（94%）**，全类 mIoU 超过 S2（0.3245 vs 0.3182），但罕见类 mIoU "
     "**仍差 0.0016 未反超**——即使此时微调后的检测器在召回与精度上已**同时**优于 S2（0.669 / 0.390 vs 0.182 / 0.296）。", indent=True)
body("• size 门控：提高操作点**直接反超**——罕见类 mIoU 0.1427 → **0.1731**（相对 S2 的 0.1519 **+0.0212，相对提升 +13.9%**），"
     "全类 mIoU 0.3336 亦高于 S2 的 0.3199。", indent=True)
body("也就是说，§4.13 指明的「微调检测器」这条路**是有效的**，但它的收益只有在**注入精度被同时提高**时才会兑现；而注入精度既取决于"
     "检测器本身，也取决于**深度门控能否拒绝错误的检测框**。", indent=True)
body("**决定性验证二：门控的差别在于「能否独立校验检测框」**（表 29）。同为 thr=0.7，两个门控的点级指标是：", indent=True, space_after=3)
caption("表 29　门控在 thr=0.7 下的点级指标")
table(
    ["门控（thr=0.7）", "点级匹配recall", "点级 precision", "端到端 mIoU罕见类"],
    [
        ["lidar（框内点云分位数）", "0.669", "0.390", "0.1456（未反超）"],
        ["size（单目高度先验）", "0.614", "**0.512**", "**0.1731（反超）**"],
    ],
    col_widths=[1.7, 1.1, 1.0, 1.6],
)
body("逐类看，size 门控的精度优势是**全面的**，不是被某一类拉高的（同为 thr=0.7，点级 precision）：", indent=True, space_after=3)
caption("逐类点级 precision（lidar vs size，thr=0.7）")
table(
    ["类别", "lidar 门控", "size 门控"],
    [
        ["ped.adult", "0.269", "**0.436**"],
        ["barrier", "0.520", "**0.560**"],
        ["trafficcone", "0.200", "**0.406**"],
        ["bicycle", "0.192", "**0.247**"],
        ["motorcycle", "0.264", "**0.521**"],
        ["罕见类汇总", "0.390", "**0.512**"],
    ],
    col_widths=[1.4, 1.0, 1.0],
)
body("两者是**召回换精度**的关系，而融合奖励精度：size 门控牺牲 0.055 的匹配recall，换来 +0.122 的点级 precision，端到端即由"
     "「未反超」变为「反超」。", indent=True)
body("为什么 size 门控的精度更高？一个自然的解释是**它能否独立校验检测框**：lidar 门控的深度区间 [p05, p90] 取自**框内 LiDAR 点的"
     "深度分位数**——对一个误检框，门控会锁定框内恰好落进去的那些点，**用被检验对象自身的证据去检验它**，因而难以拒绝误检；而 size "
     "门控的深度由框高经类别高度先验 d̂ = f_y·H_class / h_px 反推，**不依赖框内有哪些点**，是一个独立估计，框高与实际几何不符的检测框"
     "会在深度一致性上被拒掉。换言之，size 门控不只是另一种深度来源，它是三种门控中**唯一对检测框本身构成独立一致性检验**的一种。这也"
     "从机理上解释了 §4.13 中「size 优于 lidar」这一此前仅作为经验事实报告的结论——并给出一个可检验的推论：门控越能独立校验检测框，"
     "提高检测器操作点的收益就越大。", indent=True)
body("**决定性验证三：融合规则本身仍是约束**（表 30）。即使取 size 门控 thr=0.7 的最佳配置，θ 扫描在 θ=1.0 处**仍在上升**（θ=0.9 "
     "为 0.1674、0.95 为 0.1727、1.0 为 0.1731），这与 S1 oracle 的行为一致，说明 2D 一侧被信任得越多越好。把 lidar 门控 thr=0.7 下"
     "每个类的 θ 曲线拆开，两类目标的方向是**相反**的：", indent=True, space_after=3)
caption("表 30　逐类 θ 曲线（lidar 门控，thr=0.7）")
table(
    ["类别（lidar，thr=0.7）", "θ=0.5", "θ=0.6", "θ=0.7", "θ=0.8", "θ=0.9", "θ=0.95", "θ=1.0", "趋势"],
    [
        ["barrier（3D 本就会，base 0.5396）", "**0.5430**", "0.5426", "0.5412", "0.5383", "0.5335", "0.5266", "0.4784", "**单调↓**"],
        ["trafficcone（3D 本就会，base 0.2296）", "0.2489", "**0.2519**", "0.2515", "0.2461", "0.2343", "0.2213", "0.1838", "**单调↓**"],
        ["ped.adult（3D 学不会，base 0.1687）", "0.2107", "0.2366", "0.2564", "0.2723", "0.2839", "**0.2867**", "0.2406", "**单调↑**"],
        ["bicycle（3D 学不会，base 0.0101）", "0.0862", "0.1345", "0.1651", "0.1862", "0.1999", "**0.2061**", "0.1699", "**单调↑**"],
        ["motorcycle（3D 学不会，base 0.0259）", "0.0890", "0.1222", "0.1466", "0.1663", "0.1840", "0.1938", "**0.2173**", "**单调↑**"],
    ],
    col_widths=[2.2, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], font_size=8.5,
)
body("• **3D 本来就学得会的类**（barrier / trafficcone）随 θ **单调下降**：这些点上 3D 的预测本来就是对的，任何 2D 覆盖都是损伤，"
     "θ 越大覆盖越多、损伤越大。", indent=True)
body("• **3D 学不会的类**（ped.adult / bicycle / motorcycle）随 θ **单调上升**：这些点上 3D 恒定错，2D 覆盖越多越好。", indent=True)
body("一个**全局标量 θ** 无法同时服务这两组——取高 θ 伤害前者，取低 θ 放弃后者；聚合曲线上的「最优 θ」只是这两条相反曲线的折中"
     "交点。**这是融合规则本身的结构缺陷，不是阈值调参能解决的**：规则对所有 conf_3d < θ 的点**无条件覆盖**，既不比较 2D 与 3D 谁"
     "更可信，也不知道 3D 在该类上本来是否可靠。barrier 是最干净的一例——即使阈值提到 0.7，它在低 θ 下仍从 base 的 0.5396 被压到 "
     "0.5335；而 S2 因为**根本不注入**这一类，完整保住了 0.5413。", indent=True)
body("**决定性验证四：融合规则的两种候选改进，只有按类 θ 有效**（表 31）。针对验证三暴露的「无条件覆盖」缺陷，实测两种候选改进。"
     "其一，**score 门控**——仅当 2D 检测分数高于阈值 s 时才允许覆盖：因 bbox 缓存内 score 全 ≥0.70，s=0.7 即等价于无门控；把 s 逐步"
     "提高到 0.9，罕见类 mIoU 由 0.1731 **单调下降**到 0.0995（s=0.7→0.75→0.8→0.85→0.9 对应 0.1731→0.1654→0.1419→0.1118→"
     "0.0995，各取自身最优 θ），s=0.9 时已退回 base。这说明**融合增益不来自「高分检测」**：提高 score 阈值只会砍掉覆盖罕见类的有效 "
     "2D 标签，而不筛掉错误的覆盖。其二，**按类自适应 θ**——对每类取各自最优 θ（adult 0.95 / barrier 0.70 / trafficcone 0.95 / "
     "bicycle 0.95 / motorcycle 1.0，其余罕见类 2D 从不输出、默认 1.0 不影响）：罕见类 mIoU 由全局最优的 0.1731 提升到 **0.1815**，"
     "全类 mIoU 达 0.3367（也高于任何全局 θ 的 0.3337 峰值）——按类 θ 在两项指标上**严格支配**全局 θ。另需说明，候选改进中「仅在 "
     "2D 与 3D 判定不一致时覆盖」在实现上是**空条件**——对「2D 覆盖 3D 已对」的点，覆盖一个本就正确的标签等于不覆盖（torch.where "
     "下恒无操作），故真正能区分「谁更可信」的信号只有 2D 分数与逐类 θ 两个，而前者恰被本实验否证。", indent=True, space_after=3)
caption("表 31　融合规则改进（size 门控，thr=0.7）")
table(
    ["融合规则（size 门控，thr=0.7）", "罕见类 mIoU", "全类 mIoU"],
    [
        ["base（无融合）", "0.0995", "0.3090"],
        ["全局 θ（θ=1.0）", "0.1731", "0.3336"],
        ["score 门控 s=0.75 / 0.80 / 0.85 / 0.90（各取最优 θ）", "0.1654 / 0.1419 / 0.1118 / 0.0995", "—"],
        ["**按类自适应 θ**", "**0.1815**", "**0.3367**"],
    ],
    col_widths=[2.8, 1.4, 1.0],
)
body("**本节结论。** 按 §4.13 指明的方向微调检测器是**有效的**，但有效是有条件的，且条件比 §4.13 的归因更严格。三条结论：",
     indent=True)
body("1. **检测侧指标全面改善是真实的**：框级关注类平均召回 0.1160→0.6521，两类零样本盲区（barrier / trafficcone）从**严格 0** "
     "补齐到 0.7546 / 0.6705，整帧无检测从 1,153 帧降到 148。", indent=True)
body("2. **但端到端收益被注入精度门控**：默认操作点（score 0.35）下，微调后检测器「召回更高、逐类点级精度更低」，罕见类 mIoU 反而"
     "下降（size 0.1519→0.1427）。只有在**同时提高操作点**后收益才兑现，且兑现程度取决于深度门控：size 门控下 **0.1731（反超 S2 的 "
     "0.1519，相对 +13.9%）**，lidar 门控下 0.1456（仍差 0.0016）。", indent=True)
body("3. **融合规则自身仍是约束，且两种候选改进中只有按类 θ 有效**：两组类（3D 已对 vs 3D 错的）需要方向相反的 θ，全局单值 θ 必然"
     "是折中（size 门控下最优 θ 抵到扫描边界 1.0 也在提示同一件事）。实测两条改进（表 31）：(a) **score 门控被否证**——提高 2D 分数"
     "阈值单调降低融合收益，增益不来自「高分检测」；(b) **按类 θ 有效**——罕见类 mIoU 0.1731→**0.1815**，是唯一兑现的改进。",
     indent=True)
body("因此 §4.13「瓶颈在检测器召回」的归因应修正为：**瓶颈首先在注入标签的点级精度，而精度由检测器与深度门控共同决定；此外融合规则"
     "缺少逐点可信度判断**。这也意味着 §4.13 的 oracle 上界（0.5060）无法仅靠更换外部检测器兑现——融合规则本身同样需要改进，按类 θ"
     "（0.1815）是其中一条已验证的具体改进，但与上界之间仍有约 0.32 的差距，说明逐点可信度判断之外的约束（2D 框的召回上限、以及 2D "
     "从未输出的恒零类）依然存在。", indent=True)

# ======================================================================
# 4.16 安全通道
# ======================================================================
heading("4.16 安全通道：2D 检测器的实时致死目标召回与延迟", 2)
body("§3.6 的软碰撞决策在选择减速目标前，必须先回答一个更前置的问题——**哪些物体是「致死」的、必须无条件避开**。§4.13–§4.15 已"
     "表明纯 3D 分割对行人、自行车、摩托车这些致死类的逐点 IoU 近乎为零，因此致死目标的识别必须依赖 2D 检测器作为**独立的安全通道**。"
     "本节按安全通道的工程口径（而非融合口径）评估微调后的检测器：以**每图假正例率 FPPI=0.1** 处的召回衡量「检测器能否在实际可用的"
     "告警率下不漏掉致命目标」，并实测其端到端延迟是否满足实时避撞。", indent=True)
body("**召回**（表 32）。零样本 COCO 检测器作为安全通道有两个致命缺口：barrier 与 trafficcone 的召回是**严格 0**（COCO 无对应类，"
     "检测器根本不输出），而 person 的召回在 FPPI=0.1 下仅 0.178——即在每图 0.1 个假正例的可接受告警率下，五分之四以上的行人会漏检。"
     "微调后：", indent=True, space_after=3)
caption("表 32　安全通道召回（recall@FPPI=0.1）")
table(
    ["类别（组）", "零样本 recall@FPPI0.1", "微调后 recall@FPPI0.1"],
    [
        ["person（致死）", "0.178", "**0.472**"],
        ["bicycle（致死）", "0.310", "**0.582**"],
        ["motorcycle（致死）", "0.266", "**0.604**"],
        ["car（致死）", "0.360", "**0.558**"],
        ["bus（致死）", "0.589", "**0.717**"],
        ["truck（致死）", "0.316", "**0.465**"],
        ["barrier（非致死）", "0.000", "**0.456**"],
        ["trafficcone（非致死）", "0.000", "**0.503**"],
    ],
    col_widths=[1.7, 1.5, 1.5],
)
body("致死六类的召回全面抬升，其中 **person 提升 2.7 倍**（0.178→0.472）、motorcycle 提升 2.3 倍（0.266→0.604）、bus 最高达 "
     "0.717；两类 COCO 盲区被彻底补齐（0→0.456 / 0.503）。这补上了 §4.13 指出的「检测器召回」这一最薄弱环节——安全通道的漏检代价"
     "不是 mIoU 下降，而是**直接撞上未被识别的行人或车辆**。", indent=True)
body("**延迟**。端到端一次 6 相机前向（单张 1600×900）实测 **67.66 ms**（预处理 10.02 ms + 前向 55.44 ms + 后处理 0.78 ms，另有约 "
     "1.4 ms 图像解码与传输开销），峰值显存 5.27 GiB。按自车速度折算，一次推理周期内的行进距离在 30 / 50 / 70 km/h 下分别为 "
     "**0.564 / 0.94 / 1.316 m**——在 50 km/h 下，检测器的感知滞后仅约 0.94 m，低于常规 2.5 m 的安全制动余量，说明该安全通道的延迟"
     "满足实时避撞要求；55.44 ms 的骨干前向是延迟主体，若需进一步压缩，应换轻量骨干（如 RT-DETR-R18）或降低输入分辨率。", indent=True)
body("**本节结论。** 微调后的 2D 检测器同时满足安全通道的两项硬指标：致死六类在 FPPI=0.1 下的平均召回由 0.337 提升到 **0.566**，"
     "两类 COCO 盲区补齐，端到端延迟 67.66 ms 满足 50 km/h 下 <1 m 感知滞后的实时要求。这为 §3.6 的软碰撞决策提供了一个可部署的"
     "致死目标感知前置——对「非致死」类别（barrier / trafficcone）的召回不必求全，漏检的后果只是少一个可选软目标；而对「致死」类别"
     "的漏检代价高，这正是安全通道按 FPPI 而非 mIoU 评估的意义所在。", indent=True)

# ======================================================================
# 5 结论
# ======================================================================
heading("5 结论", 1)
body("本文针对\"语义标签不足以驱动物理交互\"这一瓶颈，提出了一条以语义分割为前置、以物理交互链路为主体的完整方法：构建 32 类语义→"
     "物理材质映射库，实现实例级聚类、位置相关破坏模型、导航软碰撞决策与跨帧实例跟踪，并完成 Unity/Unreal/ROS2 的工程化导出。实验验证"
     "了分割前置的可行性（16384 点、num_group=1024 时 mIoU 0.3923、OA 0.9027，num_group 扩到 2048 为 0.3965；且在 Semantic3D 上零样本 "
     "OA 0.536、冻结 backbone 微调头"
     "优于从零 +3.9 mIoU）；通过「升点数 vs 升 token 数」的消融定位了 Mamba 的规模化瓶颈为 token 数——num_group 64→1024 带来 "
     "**+0.1600 mIoU**，而 token 数同为 1024 时把点数翻倍（16k→32k）**反而下降 0.0226**；受控超参扫描进一步表明 token 预算在 1024 处"
     "进入饱和（1024→2048 的增益仅为前一段的 5.4%）、体素边长 0.5 m 为驻点；RGB 早期融合经严格配对消融净贡献 **+0.0331 mIoU**"
     "（相对 +9.2%），增益集中于几何相似而须靠外观区分的类（警车 +0.330、工程车 +0.164、交通锥 +0.118），"
     "并以受控实验把罕见类失效的成因分解为「分组粒度为主因、采样密度仅为门槛、标注量决定恒零"
     "类」的三元分工，再以地面参考的几何测量把这些恒零类区分为「几何不可分、只能引入外部模态」与「几何可分、仅监督不足」两种情况"
     "（§4.14），随后以 6 相机深度门控的"
     "2D/3D 投票融合对长尾罕见类的补全上限与可部署实际（真值框 oracle：罕见类 mIoU 0.0995→0.5060、全类 0.3090→0.4457；COCO 预训练"
     "检测器零样本的可部署设置：罕见类 0.0995→0.1519、全类 0.3090→0.3199，兑现上界增益的 12.9%），并通过关联精度分析把该落差定量"
     "归因于**检测器召回（0.18 vs oracle 0.861）**而非深度估计或融合算法——同时更正了「深度门控不可省」这一此前仅在 oracle 框条件下"
     "成立的结论；**并按该方向完成了检测器微调实验（§4.15）**：框级关注类平均召回 0.1160→0.6521、barrier / trafficcone 由严格 0 升至 "
     "0.75 / 0.67，但端到端收益须以**注入标签的点级精度**为前提——默认操作点下罕见类 mIoU 反而下降（0.1519→0.1427），提高操作点并采用"
     "单目高度先验门控后反超至 **0.1731（相对零样本 +13.9%）**，并验证按类自适应 θ 可进一步提升至 **0.1815**，同时指出融合规则对所有低"
     "置信点**无条件覆盖、缺少逐点可信度判断**是残余瓶颈；并进一步把微调后的检测器作为实时安全通道做了评估（致死六类 recall@FPPI0.1 平均"
     "0.337→0.566、barrier / trafficcone 盲区补齐、6 相机端到端延迟 67.66 ms，见 §4.16）；级联实例分割"
     "（offset 回归 + 聚类 + 物体判类）经类别门控后把 thing 类语义精度从 dense 的 0.485 提升到 0.724，并如实报告了「实例 PQ 最优与级联"
     "语义精度最优不一致」这一反直觉消融发现；进一步将类别门控推广为通用置信度门控，用 softmax 不确定度与 dense↔B2 跨模块分歧做拒识，"
     "验证了「知道何时说不知道」的可信决策能力（拒识 50% 最不确定实例后精度 +14.5 点，跨模块一致时 B2 精度达 0.953）；此外对撞击后果"
     "做了闭合形式标定（自车 1500 kg、30 m/s），定量指出原软碰撞候选集的减速上限仅初速的 **2.75%**——即原规则只优化「不伤人」而未优化"
     "「减速」，据此将决策扩展为「正碰软目标 + 刮蹭硬面」双模式并纳入锚定类（刮蹭在合成几何下减速 5.150 m/s、与手算误差 0.00%）；"
     "同时展示了位置相关破坏与航向相关软碰撞决策的真实行为。该方法为自动驾驶安全碰撞与游戏实景建图提供了统一的技术框架。", align="justify")
body("**局限与不足**：其一，§4.13 的 2D/3D 深度融合是一种**后处理**方案，可部署设置只兑现了 oracle 上界增益的 12.9%（罕见类 "
     "0.1519 vs 上界 0.5060）。经关联精度归因，该落差的主因是**零样本检测器的召回仅 0.18（oracle 为 0.861）**以及 COCO 缺失 "
     "barrier / trafficcone 两个对应类；而关联 precision 在两种设置下均接近 0.9。§4.15 完成了「在 nuScenes train split 上微调检测器」"
     "这一方向，把罕见类 mIoU 提高到 **0.1731**（相对零样本 +13.9%），但该结果有三点限制须如实说明：(a) 收益只有在**同时提高检测器"
     "操作点**后才兑现，默认操作点（score 0.35）下罕见类 mIoU 反而下降（0.1519→0.1427）——即「更高的检测器召回」本身不是充分条件；"
     "(b) 提高操作点后最优 θ 抵到扫描边界 θ=1.0 仍在上升，说明 2D 一侧被信任得越多越好、该配置的调参尚未到边界，且操作点 0.7 与 θ "
     "一样是在 val 集上选取的；(c) 即便在该配置下仍有 ped.adult（0.3555 vs 0.3754）与 barrier（0.5043 vs 0.5413）两类低于零样本设置："
     "融合规则对所有 conf_3d < θ 的点**无条件覆盖**、既不比较 2D 与 3D 谁更可信、也不知道 3D 在该类上本来是否可靠，这一残余约束尚未"
     "解决。其二，该方案依赖相机图像，对**无图像的数据集（如 Semantic3D）无法"
     "使用**。其三，逐点 3D 分割对 long-tail 罕见类本质上难以学习（类别严重不平衡，见 §4.2），§4.14 进一步用地面参考的几何测量"
     "表明：恒零类中只有 **ped.police_officer 是几何不可分**的（高度、竖直跨度与水平长短轴与普通行人逐项重叠），child / stroller 属"
     "**几何可分但监督不足**，personal_mobility / wheelchair 在 val 中**不可测**。因此「让 3D 网络通过 Focal Loss / "
     "copy-paste 自身学会罕见类」这一根本方向对 child / stroller 等几何可分类成立，但对 police 无效——后者仍需外部模态，这是减少"
     "对外部检测器依赖这一目标上**不可逾越的边界**。其四，分割规模化受 token 数（num_group）上限约束，扩展到"
     "单帧十万级点仍需 GridSampling + 序列化 patch（见未来工作）。其五，本文所有融合实验均在 nuScenes val 集上进行、且深度来源与 "
     "θ 均在该集上选取，虽然选取规则已显式声明（三档差异 <0.008、θ 曲线单峰），但严格意义上缺少独立的验证/测试划分。其六，"
     "分割前置的超参扫描（§4.2.1）只覆盖了 num_group ∈ {256, 1024, 2048} 与 grid_size ∈ {0.25, 0.5, 1.0} 三个取值，且均在 16384 点、"
     "40 轮预算下完成；更大的 token 预算（4096 及以上）与单帧十万级点尚未验证，§4.2.1 给出的\"1024 是性价比拐点\"是该预算区间内的结论，"
     "不外推。其七，§4.4 虽给出了撞击减速的闭合形式解，但 §3.5 "
     "的材料强度参数与文献失效强度按应变能密度 σ_f²/(2E) 排序**不相关**（Spearman ρ = 0.105，p = 0.734，13 种材质），因此位置相关破坏"
     "模型的**绝对**强度与破坏阈值目前未经标定，本文只主张其**相对**性质（同一能量作用于不同部位得到不同结果）成立。其八，刮蹭模式"
     "（§4.4.4）的有效性目前**仅在合成几何下验证**：真实数据中与行进方向并行的墙面受旋转式激光雷达掠射采样限制（采样密度衰减 325 倍），"
     "nuScenes val 的 200 帧上墙面长度中位数为 0 m，无法提供真实验证样本；该模式还需在专门采集的窄街/隧道类数据上验证，并接入真实"
     "规划器确认小接近角下的可控性。其九，本文的标定与决策均在运动学层面进行（瞬时冲量 + 持续摩擦），未涉及轮胎侧向力饱和、悬架与"
     "横摆动力学，故给出的 Δv 应视为**下界性质**的估计。", indent=True)
body("未来工作包括：", indent=True)
body("（1）以 GridSampling + 序列化 patch 进一步规模化到单帧十万级点、突破 token 数上限（token 数瓶颈已定位，方向明确）；", indent=True)
body("（2）改进 2D/3D 融合规则本身——§4.15 已表明仅靠更换外部检测器不足以兑现 oracle 上界，并实测了两条候选改进（§4.15 决定性"
     "验证四）：score 门控被否证（提高 2D 分数阈值单调降低收益）、「仅在 2D 与 3D 判定不一致时覆盖」在实现上是空条件，唯一兑现的是"
     "**按类自适应 θ**（罕见类 mIoU 0.1731→0.1815）；下一步应把按类 θ 推广为**逐点可信度判断**——对每个低置信点估计 2D 与 3D 谁更"
     "可信（如结合 3D 在该类上的历史可靠性与 2D 框的空间一致性）后决定是否覆盖，以进一步逼近 oracle 上界 0.5060；", indent=True)
body("（3）用 Focal Loss / copy-paste / 实例级重采样让 3D 网络自身学习长尾罕见类——§4.14 已表明单纯提高采样密度无效（新增点按比例"
     "落在近处大物体上），必须走实例级重采样与 tokenization 改进，这也是减少对 §4.13 中 2D 检测框依赖的根本方向；", indent=True)
body("（4）将软碰撞决策接入真实规划器与 TF 变换；", indent=True)
body("（5）利用跨帧跟踪的动态轨迹进行失控避撞预测；", indent=True)
body("（6）为损毁模型引入基于物理的碎裂渲染；", indent=True)
body("（7）按 σ_f²/(2E) 重标 MATERIAL_STRENGTH 并引入断裂功模型，使锚定类的 Δv 从区间收敛到点值、并使位置相关破坏的绝对阈值"
     "可对外复核（§4.4.5）；", indent=True)
body("（8）在专门采集的窄街/隧道数据上验证刮蹭模式，并接入真实规划器（§4.4.4）。", indent=True)

# ======================================================================
# 参考文献
# ======================================================================
heading("参考文献", 1)
refs = [
    "C. R. Qi, H. Su, K. Mo, and L. J. Guibas, \"PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation,\" "
    "in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 652–660.",
    "C. R. Qi, L. Yi, H. Su, and L. J. Guibas, \"PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space,\" "
    "in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017, pp. 5099–5108.",
    "H. Zhao, L. Jiang, J. Jia, P. H. S. Torr, and V. Koltun, \"Point Transformer,\" "
    "in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 16259–16268.",
    "A. Gu and T. Dao, \"Mamba: Linear-Time Sequence Modeling with Selective State Spaces,\" arXiv:2312.00752, 2023.",
    "D. Liang et al., \"PointMamba: A Simple State Space Model for Point Cloud Analysis,\" "
    "in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2024.",
    "V. Deschaintre, M. Aittala, F. Durand, G. Drettakis, and A. Bousseau, \"Single-Image SVBRDF Capture with a Rendering-Aware "
    "Deep Network,\" ACM Trans. Graph., vol. 37, no. 4, pp. 1–15, 2018, Art. no. 128.",
    "E. G. Parker and J. F. O'Brien, \"Real-Time Deformation and Fracture in a Game Environment,\" "
    "in Proc. ACM SIGGRAPH/Eurographics Symp. Comput. Animation (SCA), 2009, pp. 165–175.",
    "M. Althoff and J. M. Dolan, \"Online Verification of Automated Road Vehicles Using Reachability Analysis,\" "
    "IEEE Trans. Robot., vol. 30, no. 4, pp. 903–918, 2014.",
]
for i, r in enumerate(refs, 1):
    body(f"[{i}] {r}", size=10, indent=False, space_after=2)

# ======================================================================
# 待办清单
# ======================================================================
heading("待办清单（写作侧）", 1)
body("• [x] 补表：表 1 物理材质库全表、表 2 逐类 IoU、表 3 类别映射全表", size=10, space_after=2)
body("• [x] 补参考文献 [6][7][8]（材质估计/游戏碎裂/可达集安全）", size=10, space_after=2)
body("• [x] 写入正向结果：num_group 64→256 规模化（mIoU 0.3090，§4.2）+ 罕见类 2D/3D 深度门控投票融合（§4.13）", size=10, space_after=2)
body("• [x] 整合 docs 补充内容：Nav2 语义+几何融合导航（§3.8 / §4.7）", size=10, space_after=2)
body("• [x] 补图：图 1 流水线、图 2 网络结构、图 3 语义→物理映射可视化（均已嵌入 docx）", size=10, space_after=2)
body("• [x] 结果表统一编号（表 4–15）与 caption", size=10, space_after=2)
body("• [x] **§4.4 由「位置相关破坏验证」改写为「撞击后果标定」**（`scripts/impact_calibration.py`）：删除原同义反复表（其 `部位强度 "
     "0.225 / 能量 270J / destroyed` 三列即 §3.5 公式的输入回代，规则既当前提又当结论）；新增表 4（30 m/s、1500 kg 自车下每类 Δv 的"
     "闭合形式标定，自由体给精确值、锚定类只给区间）、P0b PyBullet 负结果（恒定约 8% 动量亏损，与 dt/ERP/迭代数/质量比均无关 ⇒ 不可作 "
     "ground truth）、决策缺口（原候选集减速上限仅初速 2.75%）、双模式决策扩展（正碰软目标 + 刮蹭硬面，合成几何误差 0.00%）、以及强度"
     "参数标定（MATERIAL_STRENGTH 与文献 σ_f²/(2E) 排序 ρ=0.105 ⇒ §3.5 只主张相对性质）。同步更新 §3.3/SEM 表、§3.5、§3.6、§4.3 "
     "表 3 的 terrain 行（deformable→passable）、§4.5、摘要、§5 结论与局限（新增其七/其八/其九）", size=10, space_after=2)
body("• [ ] 按 IEEE Sensors Journal（IEEEtran）模板统一正文/摘要/关键词格式，转 LaTeX", size=10, space_after=2)
body("• [x] 同步更新 docs/paper_draft_semphyspc.docx（投稿版 docx，已补 §4.13 与正向结果）", size=10, space_after=2)
body("• [x] 核对参考文献引用格式（[1]–[8] 已归一化为 IEEEtran）", size=10, space_after=2)
body("• [x] 新增 §4.14 罕见类失效归因（采样密度 / 分组粒度 / 标注量三元分工，表 18–表 23）", size=10, space_after=2)
body("• [x] 新增 §4.14.5 地面参考的几何可分性检验（表 22，`scripts/rare_geometry_probe.py`），把原先「4 个类不可约」的过强结论修正为"
     "三组：police 几何不可分（须外部模态）/ child+stroller 几何可分但监督不足（增加监督有效）/ personal_mobility+wheelchair 在 val "
     "不可测（不下结论）；摘要、§4.14.6 与 §5 局限同步更正", size=10, space_after=2)
body("• [x] 期刊口径确定：**IEEE Sensors Journal 首选**，中文/英文 EI 或 SCI Q3 为导师底线", size=10, space_after=2)
body("• [x] 罕见类混淆分布（`scripts/eval_seg_full.py --dump_confusion`）：**child 主要被预测成 ped.adult**（P1 49.1%、P3 59.5%），"
     "stroller 主要被预测成 manmade/adult（49.2%/41.5%）；P3 把 child 的背景类（manmade）混淆从 39.4% 压到 27.6%——监督增加后模型已"
     "把 child「当行人看」而非背景，HAG（child/adult 高度差）与实例级后处理仍可行", size=10, space_after=2)
body("• [x] 安全通道评测（带车辆 + 低阈值 0.05）：S3 微调把 person recall@FPPI0.1 由 0.178 提到 **0.472**、barrier/trafficcone 由严格 "
     "0 补到 0.456/0.503；6 相机端到端延迟 **67.66 ms**（前向 55.44 ms，显存 5.27 GiB）→ 已写成 **§4.16**（表 32）", size=10, space_after=2)
body("• [ ] HAG（离地高度）作为第 7 输入通道 / 实例级 copy-paste 增强：混淆分布显示 child→ped.adult（P1 49.1%），P3 已证明「增加监督"
     "有效」，二者**降级为未来工作**（非本文必需）", size=10, space_after=2)
body("• [x] 补训「16k 点 / num_group=1024」组至收敛 + 全量评测：**40 轮跑满后 mIoU 0.3923**（完整 val），而 32768 点组跑满 200 轮为 "
     "0.3697——**点数翻倍无增益反而低 0.0226，原「+0.019」被证实为 B₁ 未收敛造成的低估**；§4.14.2 与 §5 局限「其六」已按此改写"
     "（口径统一为 `eval_seg_full.py` 全量 val）", size=10, space_after=2)
body("• [x] num_group {256,1024,2048} 与 grid_size {0.25,0.5,1.0} 系统扫描完成（16k 点、bs 8、40 epoch，全量评测）："
     "mIoU 0.3144/0.3923/0.3965 与 0.3887/0.3923/0.3731 → 已写入 §4.2.1", size=10, space_after=2)
body("• [x] RGB 消融（XYZ 3 通道 vs XYZ+RGB 6 通道，其余条件严格相同，40 轮跑满）：0.3592 vs 0.3923 = **+0.0331（相对 +9.2%）**，"
     "8 个同 epoch 配对点全为正 → 已写入 §4.2.2；同步更正「颜色判别力弱 ⇒ RGB 融合价值有限」的推论", size=10, space_after=2)
body("• [x] S3 第 1 步：`scripts/build_s3_det_labels.py` 用 nuScenes **train split** 的 3D 真值框投影出 6 相机 2D 标签（28 071 帧 / "
     "891 167 框，新增 barrier 127 168 框、trafficcone 68 167 框）", size=10, space_after=2)
body("• [x] S3 第 2 步：`scripts/finetune_rtdetr_nuscenes.py` 头部 80→82 类改造 + VFL 正样本类别加权 + 稀有类重复因子采样（**冒烟通过，"
     "排队训练中**）", size=10, space_after=2)
body("• [x] P3：类别平衡损失（逆平方根频率，floor 0.25 / cap 10）+ 罕见帧过采样（police ×17.2、child ×8.8、stroller ×12.3、pers_mob "
     "×17.1；`scripts/make_p3_config.py` 生成配置）**训练完成**：罕见类 **recall** 全面提升（construction_worker 0.30→0.43、bicycle "
     "0.11→0.28、trafficcone 0.50→0.69、child/pers_mob 脱离精确 0），但 IoU 未升（全类 mIoU 0.392→0.362），致命误判率 5.47%→**4.84%**"
     "（见 §4.14.5 末）", size=10, space_after=2)
body("• [x] S3 第 3 步：检测器微调完成（16 轮，dev 关注类均值选中 ep13；期间定位并修复了 `transformers` RT-DETR 损失**主分支未切去噪 "
     "query** 的实现缺陷，详见 §4.15）；下游 det → 三门控 bbox 缓存 → 门控投票评测全部完成", size=10, space_after=2)
body("• [x] **新增 §4.15**（表 24–表 31）：微调后框级关注类平均召回 0.1160→0.6521、barrier/trafficcone 由严格 0 升至 0.7546/0.6705；"
     "但默认操作点下罕见类 mIoU 反而下降（size 0.1519→0.1427）；在 score 阈值 0.5/0.7 上重建缓存重跑后，`size` 门控 **0.1731（反超"
     "零样本 +13.9%）**、`lidar` 门控 0.1456（仍差 0.0016）；据此把 §4.13「瓶颈在检测器召回」的归因修正为**注入精度 + 融合规则缺少"
     "逐点可信度判断**", size=10, space_after=2)
body("• [x] §4.13 表 16 口径更正：罕见类 recall 一列原用**会整帧跳过「检测器无输出」样本**的旧版 `verify_2d3d_fusion.py` 算得（虚高），"
     "修正分母后 size 0.192→**0.170**、lidar 0.205→**0.182**、none 0.205→**0.183**、oracle 0.861 不变；摘要（中/英）、贡献点、"
     "§4.13 正文与结论、§5 结论与局限、未来工作均同步更正", size=10, space_after=2)
body("• [x] 融合规则改进（θ 按类自适应 + score 门控）——§4.15 决定性验证四：全局 θ 0.1731、score 门控被否证（s=0.9 回落 0.0995）、"
     "**per-class θ 0.1815**（相对 baseline +82%）", size=10, space_after=2)
body("• [ ] 实例级 copy-paste 增强：**降级为未来工作**（P3 的类别平衡损失 + 罕见帧过采样已足够证明「监督不足」是主因）", size=10, space_after=2)

doc.save(OUT)
print(f"已生成：{OUT}")
print(f"段落数：{len(doc.paragraphs)}，表格数：{len(doc.tables)}")
