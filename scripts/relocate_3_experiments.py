#!/usr/bin/env python3
"""Remote Sensing 重定位（3/3）：Experiments 压缩 + Conclusion 重写。

- 4.4 撞击标定压缩（15 段 -> 1 段 + Table 4，保留闭式 Δv + PyBullet 负结果）
- 4.5 + 4.6 合并压缩为「4.5 Downstream Navigation Application Validation」（保留 Gazebo 闭环 + 真实地图 + Fig.4）
- 4.7 -> 4.6（跨数据集顺延）
- Conclusion 重写（1021 词 -> ~260 词，突出分割 + 融合 + 门控投票）

用法（项目根下）：python3 scripts/relocate_3_experiments.py
"""
import re
import shutil

import docx
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.text.paragraph import Paragraph

SRC = 'docs/paper_draft_semphyspc.docx'
BAK = SRC + '.bak-relo3'

NEW_44 = (
    'The calibration answers how much deceleration an impact imposes on the ego vehicle. We calibrate Δv '
    'with a closed-form solution instead of a physics simulator: for a free object the head-on velocity change '
    'has the exact solution Δv = (1+e)·m₂/(m₁+m₂)·v (m₁ = 1500 kg, v = 30 m/s, '
    'e the restitution), an anchored object reaches Δv = (1+e)v in the rigid limit, and a destructible '
    'anchored object falls in between. We once used PyBullet as the calibration reference, but it exhibited a '
    'systematic 8% momentum non-conservation independent of time step, solver iterations and mass ratio, so it '
    'cannot serve as ground truth. Table 4 lists the per-class head-on deceleration under a 30 m/s, 1500 kg ego '
    'vehicle.'
)

NEW_45_TITLE = '4.5 Downstream Navigation Application Validation'
NEW_45_P1 = (
    'On a real nuScenes frame (000260ba…, 4096 points) we vary the heading and the decision always '
    'prioritizes low-lethality targets, recommending traversable vegetation or a traffic cone and reporting no '
    'safe target when only lethal/rigid objects lie in the view cone; this validates the decision level of '
    '“which soft target to hit”.'
)
NEW_45_P2 = (
    'The semantic soft-cost layer is validated in a Gazebo closed-loop simulation and a Semantic3D real-map '
    'experiment (Fig. 4). In the closed loop, the planner, using only the soft cost, bypasses a lethal '
    'pedestrian while driving through traversable vegetation. In the real-map experiment (domfountain_station3), '
    'the semantic soft cost is projected onto the Nav2 costmap and the planned path keeps a 1.68 m clearance '
    'from a pedestrian while crossing vegetation.'
)
NEW_45_P3 = (
    'The soft_contact package passes colcon build and pytest end-to-end tests, and the Nav2 semantic + geometric '
    'fusion navigation is verified in an offline simulation (fake transform and fake laser, driven by the real '
    'map).'
)

NEW_CONCL_MAIN = (
    'This paper presented SemPhysPC, a semantic point-cloud segmentation method oriented to autonomous navigation. '
    'Its front-end fuses camera RGB with LiDAR XYZ through an early-fusion six-channel input and Mamba sequence '
    'modeling; a strictly paired ablation shows that RGB contributes +0.0331 mIoU (0.3923 mIoU on nuScenes), '
    'with the gain concentrated on geometrically similar classes separable only by appearance. For the rare but '
    'safety-critical lethal classes that the long-tail distribution nearly wipes out, we add a depth-gated 2D/3D '
    'voting mechanism that lifts a high-confidence 2D detection into the 3D labels, lowering the critical_rate '
    'from 5.30% to 3.80%, with a GT-oracle upper bound of 0.92%. The per-point semantics are then mapped to '
    'physical interaction attributes (collision-safety level, mass, friction, restitution, destructibility) and '
    'exported as a continuous semantic soft-cost layer for a ROS2 navigation stack. A Gazebo closed-loop '
    'simulation and a Semantic3D real-map experiment verify that the planner, using only this soft cost, bypasses '
    'a lethal pedestrian while driving through traversable vegetation, and cross-dataset transfer to Semantic3D '
    'further confirms the generalization of the early-fusion front-end.'
)

NEW_CONCL_LIMIT = (
    'The main limitations are threefold. First, segmentation still scales as the token budget rather than the '
    'point count, so single-frame 100k-level points need further upscaling. Second, the depth-gated voting is '
    'bounded by the 2D detector quality (the 0.92% oracle). Third, the closed loop is validated in simulation '
    'and on recorded maps rather than on a physical robot. Future work will validate the pipeline on a real '
    'platform and extend the attribute mapping to cross-frame tracked trajectories.'
)


def text(el):
    ts = el.findall('.//' + qn('w:t'))
    return ''.join(t.text or '' for t in ts)


def set_para_text(p, txt, size=12, bold=False):
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    run = p.add_run(txt)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)
    run.font.bold = bold
    rf = run._element.get_or_add_rPr().get_or_add_rFonts()
    rf.set(qn('w:eastAsia'), '宋体')
    return run


def to_paragraph(el):
    return Paragraph(el, None)


def make_para(txt, size=12, bold=False):
    p = OxmlElement('w:p')
    pPr = OxmlElement('w:pPr')
    spacing = OxmlElement('w:spacing')
    spacing.set(qn('w:line'), '360')
    spacing.set(qn('w:lineRule'), 'auto')
    pPr.append(spacing)
    p.append(pPr)
    r = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    rf = OxmlElement('w:rFonts')
    rf.set(qn('w:ascii'), 'Times New Roman'); rf.set(qn('w:hAnsi'), 'Times New Roman')
    rf.set(qn('w:eastAsia'), '宋体')
    rPr.append(rf)
    sz = OxmlElement('w:sz'); sz.set(qn('w:val'), str(size * 2)); rPr.append(sz)
    szCs = OxmlElement('w:szCs'); szCs.set(qn('w:val'), str(size * 2)); rPr.append(szCs)
    if bold:
        rPr.append(OxmlElement('w:b'))
    r.append(rPr)
    t = OxmlElement('w:t'); t.text = txt
    r.append(t)
    p.append(r)
    return p


def find_body_para(doc, prefix):
    for el in doc.element.body:
        if el.tag == qn('w:p') and text(el).strip().startswith(prefix):
            return el
    return None


def compress_44(el_44, el_45, new_body):
    """4.4：替换首段，删其余段落，保留 Table 4（caption + table + 注释）。"""
    first = el_44.getnext()
    while first is not None and first.tag != qn('w:p'):
        first = first.getnext()
    set_para_text(to_paragraph(first), new_body)
    el = first.getnext()
    while el is not None and el is not el_45:
        nxt = el.getnext()
        t = text(el).strip()
        keep = (el.tag == qn('w:tbl')) or t.startswith('Table 4. Head-on') or t.startswith('Free objects are exact')
        if not keep:
            el.getparent().remove(el)
        el = nxt


def merge_45_46(el_45, el_47):
    """4.5+4.6 合并：删除区间内容，保留 Fig.4（图 + caption），插入压缩段。"""
    # 先保存 Fig.4 图和 caption 元素引用
    fig_el = None
    caption_el = None
    el = el_45.getnext()
    while el is not None and el is not el_47:
        if el.tag == qn('w:p') and el.findall('.//' + qn('w:drawing')):
            fig_el = el
            caption_el = el.getnext()
        el = el.getnext()
    # 删除区间内所有元素
    el = el_45.getnext()
    while el is not None and el is not el_47:
        nxt = el.getnext()
        el.getparent().remove(el)
        el = nxt
    # 重新插入：段1 -> 段2 -> 图 -> caption -> 段3
    el_45.addnext(make_para(NEW_45_P1))
    p1 = el_45.getnext()
    p1.addnext(make_para(NEW_45_P2))
    p2 = p1.getnext()
    p2.addnext(fig_el)
    if caption_el is not None:
        fig_el.addnext(caption_el)
        caption_el.addnext(make_para(NEW_45_P3))
    else:
        fig_el.addnext(make_para(NEW_45_P3))


def main():
    shutil.copy2(SRC, BAK)
    doc = docx.Document(SRC)

    el_44 = find_body_para(doc, '4.4 Impact-Consequence')
    el_45 = find_body_para(doc, '4.5 Navigation Soft-Collision Decision Validation')
    el_46 = find_body_para(doc, '4.6 System-Integration')
    el_47 = find_body_para(doc, '4.7 Cross-Dataset')
    el_concl = find_body_para(doc, '5 Conclusion')
    assert all([el_44, el_45, el_46, el_47, el_concl]), '章节标题定位失败'

    # 1. 4.4 压缩
    compress_44(el_44, el_45, NEW_44)

    # 2. 4.5 + 4.6 合并
    set_para_text(to_paragraph(el_45), NEW_45_TITLE, bold=True)
    merge_45_46(el_45, el_47)

    # 3. 4.7 -> 4.6
    set_para_text(to_paragraph(el_47), '4.6 Cross-Dataset Generalization (Semantic3D)', bold=True)

    # 4. Conclusion 重写
    p = el_concl.getnext()
    while p is not None and p.tag != qn('w:p'):
        p = p.getnext()
    set_para_text(to_paragraph(p), NEW_CONCL_MAIN)
    p2 = p.getnext()
    while p2 is not None and p2.tag != qn('w:p'):
        p2 = p2.getnext()
    set_para_text(to_paragraph(p2), NEW_CONCL_LIMIT)
    # 删除 p2 之后到 Data Availability 之间的段落（Future work 标题 + 5 条）
    el = p2.getnext()
    while el is not None:
        t = text(el).strip()
        if t.startswith('Data Availability') or t.startswith('References'):
            break
        nxt = el.getnext()
        el.getparent().remove(el)
        el = nxt

    doc.save(SRC)

    # ---- 验证 ----
    d2 = docx.Document(SRC)
    joined = '\n'.join(p.text for p in d2.paragraphs)
    checks = {
        '4.5 合并标题': '4.5 Downstream Navigation Application Validation' in joined,
        '旧 4.6 标题已消除': '4.6 System-Integration Validation' not in joined,
        '4.6 跨数据集顺延': '4.6 Cross-Dataset Generalization (Semantic3D)' in joined,
        '旧 4.7 标题已消除': '4.7 Cross-Dataset Generalization' not in joined,
        '4.4 闭式解保留': '(1+e)' in joined and 'PyBullet' in joined,
        '4.4.4 刮蹭细节已删': '4.4.4 Improvement' not in joined,
        'Conclusion 新主体': 'SemPhysPC, a semantic point-cloud segmentation method' in joined,
        '旧 Future work 已删': 'Future work includes:' not in joined,
    }
    for k, v in checks.items():
        print(f'{k}: {"OK" if v else "FAIL"}')
    # 关键数字
    print('关键数字在:', all(s in joined for s in ['5.30%', '3.80%', '0.92%', '1.68 m', '0.3923', '+0.0331']))
    # 各节词数
    print('--- 章节标题 + 词数 ---')
    for p in d2.paragraphs:
        t = p.text.strip()
        if re.match(r'^(4\.\d|5)\s+[A-Z]', t) or t == 'References':
            wc = len(re.findall(r"[A-Za-z0-9.]+(?:-[A-Za-z0-9.]+)*", t))
            print(f'  {t[:55]}  ({wc}词)')


if __name__ == '__main__':
    main()
