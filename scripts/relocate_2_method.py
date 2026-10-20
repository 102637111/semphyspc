#!/usr/bin/env python3
"""Remote Sensing 重定位（2/3）：Method 章节。

- 3.2 分割前端扩充一段网络结构形式化描述
- 新增 3.3 Depth-Gated 2D/3D Voting 方法节
- 原 3.3 物理映射 -> 3.4（重命名 + 微调定位）
- 原 3.4 撞击标定 -> 3.5（压缩为 1 段）
- 原 3.5 软碰撞决策 -> 3.6（压缩为 1 段）
- 原 3.6 系统导出 -> 3.7（压缩为 1 段）
- 交叉引用 Section 3.3/3.4/3.5/3.6 顺延更新（+1）

用法（项目根下）：python3 scripts/relocate_2_method.py
"""
import re
import shutil

import docx
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.text.paragraph import Paragraph

SRC = 'docs/paper_draft_semphyspc.docx'
BAK = SRC + '.bak-relo2'

ADD_32 = (
    'Formally, given a LiDAR point cloud X ∈ R^{N×3} and the camera RGB projected onto the same points '
    'C ∈ R^{N×3}, the front-end concatenates them into a six-channel tensor [X, C]. The encoder is '
    'non-hierarchical: farthest-point sampling (FPS) selects num_group centre points, and Mamba state-space blocks '
    'extract features with linear complexity in the sequence length, which is precisely why the token budget '
    '(num_group) is the scaling knob analysed in Section 4.2.1. A KNN-upsampling decoder propagates the grouped '
    'features back to the full resolution to recover per-point semantics. Because the token sequence length, not '
    'the input point count, dominates the scaling bottleneck, the encoder is deliberately kept non-hierarchical '
    'with the budget controlled by num_group.'
)

NEW_33_TITLE = '3.3 Depth-Gated 2D/3D Voting for Rare Lethal Classes'
NEW_33_BODY = (
    'Rare thing classes, especially pedestrians and animals, are nearly wiped out by the long-tail distribution, '
    'and as Section 4.2 shows, raising mIoU alone does not lower the safety-critical misclassification rate. We '
    'therefore add an orthogonal 2D sensing path. A 2D detector (RT-DETR) runs on the camera images and produces '
    '2D boxes with class labels and confidence. Each 3D point is projected into the camera views, and a point '
    'falling inside a detected box becomes a candidate for that box’s class. A depth gate then suppresses '
    'false associations: the point’s LiDAR depth must agree with the box depth, so only points within a depth '
    'tolerance are voted. The 3D label is overridden by the 2D class only when the gate passes and the confidence '
    'exceeds a threshold θ, allowing a high-confidence 2D detection of a lethal class to rescue a point that '
    'the 3D front-end misclassified as collidable. The sweep over θ and the GT-oracle upper bound are reported '
    'in Section 4.2.3.'
)

NEW_34_TITLE = '3.4 Semantic-to-Physical Attribute Mapping for Navigation'
NEW_34_INTRO = (
    'The 32 semantic classes are mapped to five collision-safety levels and 13 physical materials, giving each '
    'point attributes usable by a downstream navigation decision:'
)

NEW_35_BODY = (
    'The downstream decision needs how much deceleration an impact imposes, Δv. We calibrate Δv in '
    'closed form rather than with a physics simulator (which we found unreliable, Section 4.4): for a free body '
    'the head-on velocity change is Δv = (1+e)·m₂/(m₁+m₂)·v, where m₁ = 1500 kg '
    'is the ego mass, v = 30 m/s, and e is the restitution; an anchored object reaches Δv = (1+e)v in the '
    'rigid limit, and a destructible anchored object falls in between. The per-class values are tabulated in '
    'Section 4.4.'
)

NEW_36_BODY = (
    'The decision selects a soft-collision target under two contact modes that share the same safety constraint. '
    'Mode A (head-on soft target): within the forward view cone, frangible and deformable classes are ranked by '
    'ascending lethality and the lowest-cost target is recommended, while lethal (pedestrian/animal) and hard-rigid '
    '(vehicle/wall) classes are excluded. Mode B (glancing scrape): within a lateral corridor, a hard-rigid '
    'anchored surface roughly parallel to the heading is scraped at a small approach angle to decelerate through '
    'sustained Coulomb friction. The decision maximizes Δv subject to risk ≤ risk_limit; the deceleration '
    'and gap analysis is in Section 4.4.'
)

NEW_37_BODY = (
    'The mapping and decision are exported to a navigation stack. Semantics are wrapped as a ROS2 node '
    '(soft_contact) that publishes the recommended impact point and candidate targets, and fused with the Nav2 '
    'costmap through marker-type observation sources. To let the planner express a traversable-but-costlier soft '
    'obstacle rather than a binary one, the per-point lethality (Section 3.4) is projected onto a 2D grid in the '
    'map frame, each cell takes the maximum lethality, and the result is mapped to a continuous Nav2 cost; the '
    'planner then minimizes this semantic soft cost together with the geometric cost, bypassing lethal targets '
    'while accepting low-cost soft targets such as vegetation.'
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
    spacing.set(qn('w:line'), '360')          # 1.5 倍行距
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


def update_refs(doc):
    """交叉引用 Section 3.x 顺延 +1（从大到小避免连锁）。"""
    mapping = [('Section 3.6', 'Section 3.7'), ('Section 3.5', 'Section 3.6'),
               ('Section 3.4', 'Section 3.5'), ('Section 3.3', 'Section 3.4')]
    n = 0
    for p in doc.paragraphs:
        t = p.text
        if any(old in t for old, _ in mapping):
            for old, new in mapping:
                t = t.replace(old, new)
            set_para_text(p, t, bold=any(r.font.bold for r in p.runs))
            n += 1
    print(f'交叉引用更新段落数: {n}')


def compress_section(title_el, next_prefixes, new_body):
    """删除 title_el 与下一个章节标题之间的正文段落，替换为 new_body 单段。"""
    nxt = None
    el = title_el.getnext()
    while el is not None:
        t = text(el).strip()
        if el.tag == qn('w:p') and any(t.startswith(p) for p in next_prefixes):
            nxt = el
            break
        el = el.getnext()
    # 删除 title 与 next 之间的段落
    el = title_el.getnext()
    while el is not None and el is not nxt:
        nxt_el = el.getnext()
        el.getparent().remove(el)
        el = nxt_el
    title_el.addnext(make_para(new_body))


def main():
    shutil.copy2(SRC, BAK)
    doc = docx.Document(SRC)

    # 0. 先做交叉引用顺延（针对保留的旧文本）
    update_refs(doc)

    # 定位各标题段（元素引用）
    el_33 = find_body_para(doc, '3.3 Semantic-to-Physical')
    el_34 = find_body_para(doc, '3.4 Impact Consequence')
    el_35 = find_body_para(doc, '3.5 Navigation Soft-Collision Decision')
    el_36 = find_body_para(doc, '3.6 System Export')
    assert all([el_33, el_34, el_35, el_36]), 'Method 标题段定位失败'

    # 1. 3.2 扩充：在 el_33 前插入扩充段
    el_33.addprevious(make_para(ADD_32))

    # 2. 新增 3.3 门控投票节（在 el_33 前插入标题+正文）
    el_33.addprevious(make_para(NEW_33_BODY))
    el_33.addprevious(make_para(NEW_33_TITLE, bold=True))

    # 3. 原 3.3 -> 3.4 重命名 + 微调开头
    set_para_text(to_paragraph(el_33), NEW_34_TITLE, bold=True)
    p = el_33.getnext()
    while p is not None and p.tag != qn('w:p'):
        p = p.getnext()
    if p is not None:
        old = text(p)
        if old.startswith('This is one of the core contributions'):
            set_para_text(to_paragraph(p), NEW_34_INTRO)

    # 4. 原 3.4 -> 3.5 重命名 + 压缩
    set_para_text(to_paragraph(el_34), '3.5 Impact Consequence and Deceleration Calibration', bold=True)
    compress_section(el_34, ['3.5 Navigation Soft-Collision Decision',
                             '3.6 Navigation Soft-Collision Decision', '3.7', '4 Experiments'], NEW_35_BODY)

    # 5. 原 3.5 -> 3.6 重命名 + 压缩
    set_para_text(to_paragraph(el_35), '3.6 Navigation Soft-Collision Decision', bold=True)
    compress_section(el_35, ['3.6 System Export', '3.7 System Export', '4 Experiments'], NEW_36_BODY)

    # 6. 原 3.6 -> 3.7 重命名 + 压缩
    set_para_text(to_paragraph(el_36), '3.7 System Export and Integration', bold=True)
    compress_section(el_36, ['4 Experiments'], NEW_37_BODY)

    doc.save(SRC)

    # ---- 验证 ----
    d2 = docx.Document(SRC)
    joined = '\n'.join(p.text for p in d2.paragraphs)
    checks = {
        '新增 3.3 门控投票': '3.3 Depth-Gated 2D/3D Voting for Rare Lethal Classes' in joined,
        '3.4 物理映射': '3.4 Semantic-to-Physical Attribute Mapping for Navigation' in joined,
        '3.5 撞击标定': '3.5 Impact Consequence and Deceleration Calibration' in joined,
        '3.6 软碰撞决策': '3.6 Navigation Soft-Collision Decision' in joined,
        '3.7 系统导出': '3.7 System Export and Integration' in joined,
        '旧 3.3 标题已消除': '3.3 Semantic-to-Physical Material Mapping Library' not in joined,
        '旧 Section 3.6 已消除': 'Section 3.6' not in joined,
        '旧 Section 3.3 已消除': 'Section 3.3' not in joined,
    }
    for k, v in checks.items():
        print(f'{k}: {"OK" if v else "FAIL"}')
    print('--- Method 章节标题 ---')
    for p in d2.paragraphs:
        if re.match(r'^3\.\d\s+[A-Z]', p.text.strip()):
            print(' ', p.text.strip()[:60])


if __name__ == '__main__':
    main()
