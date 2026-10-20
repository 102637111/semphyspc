#!/usr/bin/env python3
"""把「阶段 3 语义软代价层 Gazebo 闭环」同步进权威载体 docs/语义地图.docx。

只同步软代价层闭环 7 处，不碰叙事方向（游戏弱化 / 撞击后果标定 等用户既定决策）：
  1. 中文摘要：critical_rate 句 → 软代价层句（run 级替换）
  2. 英文 Abstract：critical_rate 句 → 软代价层句（run 级替换）
  3. §3.8：导航 bullet 后插入「语义软代价层」bullet
  4. §4.5：结果段后插入衔接句
  5. §4.7：游戏导出 bullet 后插入「语义软代价层 Gazebo 闭环」bullet
  6. §5 结论：删 critical_rate 句 + 结尾加软代价层句
  7. 待办清单：末尾插入阶段 3 条目

用法（项目根 /home/saw/ros2_ws/src/semphyspc/ 下）：
    python3 scripts/sync_semdocx_softcost.py
"""
import copy
import re
import shutil
import docx
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

SRC = 'docs/语义地图.docx'
BAK = 'docs/语义地图.docx.bak-20261002-softcost'

WEST = 'Times New Roman'
EAST = '宋体'
SIZE = docx.shared.Pt(10.5)


def find_para(doc, needle, exclude=None):
    for p in doc.paragraphs:
        if needle in p.text:
            if exclude and any(e in p.text for e in exclude):
                continue
            return p
    raise RuntimeError(f'段落未找到: {needle}')


def replace_in_run(p, old, new):
    """在段落 p 的单个 run 内做文本替换（old 必须完整落在某个 run 内）。"""
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new, 1)
            return True
    # 若跨 run（罕见），退回段落全文替换并保持加粗结构
    raise RuntimeError(f'run 内未找到待替换文本: {old[:60]}')


def _set_run_font(r, bold):
    r.font.name = WEST
    r.font.size = SIZE
    r.font.bold = bold
    r._element.get_or_add_rPr()
    rf = r._element.rPr.get_or_add_rFonts()
    rf.set(qn('w:eastAsia'), EAST)


def make_md_para(template_para, text):
    """deepcopy 模板段落，按 **bold** / `code` 分段填充，返回新 w:p 元素。

    模板段落需含至少一个 bold run（作 bold 格式源）与一个 normal run（作 normal 格式源）。
    """
    el = copy.deepcopy(template_para._p)
    p = Paragraph(el, template_para._parent)

    bold_src = None
    normal_src = None
    for r in p.runs:
        if r.font.bold and bold_src is None:
            bold_src = r._r
        elif not r.font.bold and normal_src is None:
            normal_src = r._r

    for r in list(p.runs):
        r._r.getparent().remove(r._r)

    # 解析 **bold** 与 `code`
    tokens = re.split(r'(\*\*[^*]+\*\*|`[^`]*`)', text)
    for tok in tokens:
        if not tok:
            continue
        bold = False
        if tok.startswith('**') and tok.endswith('**'):
            bold = True
            tok = tok[2:-2]
        elif tok.startswith('`') and tok.endswith('`'):
            tok = tok[1:-1]  # code 用普通文本，去掉反引号

        src = bold_src if bold else normal_src
        if src is None:
            src = bold_src or normal_src
        new_r = copy.deepcopy(src)
        # 设置文本
        t_el = new_r.find(qn('w:t'))
        if t_el is None:
            t_el = new_r.makeelement(qn('w:t'), {})
            new_r.append(t_el)
        t_el.text = tok
        # 确保 bold 状态与目标一致
        rPr = new_r.find(qn('w:rPr'))
        if rPr is None:
            rPr = new_r.makeelement(qn('w:rPr'), {})
            new_r.insert(0, rPr)
        b = rPr.find(qn('w:b'))
        if bold and b is None:
            b = rPr.makeelement(qn('w:b'), {})
            rPr.append(b)
        elif not bold and b is not None:
            rPr.remove(b)
        p._p.append(new_r)
    return el


def insert_after(anchor, el):
    anchor._p.addnext(el)


def new_para_like(template, text):
    el = copy.deepcopy(template._p)
    p = Paragraph(el, template._parent)
    for r in p.runs:
        r.text = ''
    if p.runs:
        p.runs[0].text = text
    else:
        p.add_run(text)
    return el


def main():
    shutil.copy2(SRC, BAK)
    doc = docx.Document(SRC)

    # 1. 中文摘要：critical_rate → 软代价层
    p_abs = find_para(doc, '并展示了多航向下正确选择软碰撞目标的决策行为')
    replace_in_run(
        p_abs,
        '同时以致命类误判率 critical_rate 量化决策安全，用 2D 深度门控投票将其从 5.30% 降到 3.80%（可部署）/ 0.92%（oracle 上界）。',
        '并把软碰撞决策落地为规划器可消费的语义软代价层（致命类 254、可穿越类 30~80 软代价），在 Gazebo 物理仿真闭环中验证了规划器仅凭连续软代价即可「绕开致命行人、压过可穿越植被」。',
    )

    # 2. 英文 Abstract：critical_rate → 软代价层
    p_abs_en = find_para(doc, 'heading-dependent soft-collision decisions')
    replace_in_run(
        p_abs_en,
        'and we quantify decision safety with the lethal-class misclassification rate (critical_rate), which the 2D depth-gated voting lowers from 5.30% to 3.80% (deployable) / 0.92% (oracle ceiling).',
        'a semantic soft-cost layer (lethal classes 254, traversable classes 30–80) turns the soft-collision decision into a planner-consumable cost function, and a Gazebo closed-loop simulation verifies that the planner, using only this continuous cost, automatically bypasses a lethal pedestrian and drives through traversable vegetation.',
    )

    # 3. §3.8：导航 bullet 后插入「语义软代价层」bullet
    p_nav = find_para(doc, '导航：封装为 ROS2 节点')
    sem38 = (
        '**语义软代价层**：上述「语义障碍」是二元的（命中 `obstacle_classes` 即标为致命），'
        '规划器只能「绕开/不绕开」，无法表达「可穿越但代价更高」的软障碍（如植被）。'
        '为此把语义升级为**连续软代价**：将逐点物理属性 `lethality`（致命性代价，§3.3）投影到 map 帧 2D 网格、'
        '逐 cell 取 `max(lethality)`，再映射为 Nav2 costmap 的连续代价（0~254）——致命类（行人/动物）→ 254，'
        '硬刚性类 → 150~220（强避让），可破坏/可变形类 → 30~80（软代价、可穿越），地面/可通行 → 0。'
        '由 C++ costmap 插件（`semantic_costmap_layer`）订阅 `/semantic_cost`、以 `updateWithMax` 写入 master costmap'
        '（不覆盖障碍层已写的致命值）。由此 §3.6 的软碰撞决策被落地为**规划器可消费的代价函数**：'
        '在连续代价栅格上，规划器自然「绕开致命目标、压过可穿越软目标」。'
    )
    insert_after(p_nav, make_md_para(p_nav, sem38))

    # 4. §4.5：结果段后插入衔接句
    p_s45 = find_para(doc, '结果符合预期：决策始终优先低致命性')
    bridge = '本节验证的是「该撞哪个软目标」这一决策层面；而「如何在规划器层绕开致命目标、压过选中的软目标」由 §3.8 的语义软代价层承接，其 Gazebo 闭环验证见 §4.7。'
    insert_after(p_s45, new_para_like(p_s45, bridge))

    # 5. §4.7：游戏导出 bullet 后插入「语义软代价层 Gazebo 闭环」bullet
    p_game = find_para(doc, '游戏导出（可扩展）')
    sem47 = (
        '**语义软代价层 Gazebo 闭环**：为验证软代价层在物理仿真中的闭环效果，'
        '在 Gazebo Classic 11 + ROS2 Humble 中搭建走廊场景（y∈[−1.9, 1.9]）：'
        '正中 (3.0, 0.0) 放置**带碰撞体**的行人圆柱（半径 0.3 m、高 1.7 m，激光与语义双重致命 254），'
        '其两侧 (3.0, ±0.9) 各放一个**仅视觉、无碰撞体**的植被盒（1.0×1.2×1.0 m，激光可穿透、仅语义软代价 46）。'
        '差速底盘（TurtleBot3 Waffle，footprint 0.30×0.30 m）从 A(0,0) 导航至 B(8,0)，'
        '全局规划器用 SmacPlanner2D（REEDS_SHEPP）、局部控制器用 Regulated Pure Pursuit（`use_rotate_to_heading=false`）。'
        '实测路径在接近行人时抬升到 y≈0.72~0.73（植被软代价区 y∈[0.3, 1.5] 内侧），'
        '同时始终保持在行人致命区（|y|<0.3）之外；轨迹 y 峰值 0.77 m @ x=3.12，'
        '29 s 后无碰撞停车于 (7.98, −0.03)、末速度归零。'
        '**仅凭连续软代价（行人 254 / 植被 46）、不引入任何显式绕行规则，'
        '规划器自动选择「压过可穿越植被、绕开致命行人」**，验证了语义软代价层作为规划器代价函数的可行性。'
    )
    insert_after(p_game, make_md_para(p_game, sem47))

    # 6. §5 结论：删 critical_rate 句 + 结尾加软代价层句
    p_concl = find_para(doc, '同时展示了航向相关软碰撞决策的真实行为')
    replace_in_run(
        p_concl,
        '并把语义按物理碰撞属性合并为 12 类、以致命类误判率 critical_rate 量化决策安全，用 2D 深度门控投票将其从 5.30% 降到 3.80%（可部署）/ 0.92%（oracle 上界），见 §4.17；',
        '',
    )
    replace_in_run(
        p_concl,
        '同时展示了航向相关软碰撞决策的真实行为。',
        '同时展示了航向相关软碰撞决策的真实行为；并将软碰撞决策落地为规划器可消费的语义软代价层，在 Gazebo 物理仿真闭环中验证了规划器仅凭连续软代价即可「绕开致命行人、压过可穿越植被」。',
    )

    # 7. 待办清单：末尾插入阶段 3 条目
    p_todo = find_para(doc, '新增 §4.17')
    todo = '• [x] 阶段 3 语义软代价层 Gazebo 闭环（TurtleBot3 绕 person 穿 vegetation）已写入 §3.8（方法）/ §4.7（闭环验证）/ §4.5（衔接句）；摘要（中/英）与 §5 结论同步'
    insert_after(p_todo, new_para_like(p_todo, todo))

    doc.save(SRC)
    print(f'完成：{SRC} 已更新（备份 -> {BAK}）')


if __name__ == '__main__':
    main()
