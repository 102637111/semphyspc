#!/usr/bin/env python3
"""Remote Sensing 重定位（1/3）：标题、摘要、Introduction、Related Work。

用法（项目根下）：python3 scripts/relocate_1_front.py
"""
import re
import shutil

import docx
from docx.oxml.ns import qn
from docx.shared import Pt

SRC = 'docs/paper_draft_semphyspc.docx'
BAK = SRC + '.bak-relo1'

NEW_TITLE = ('Semantic Point-Cloud Segmentation with RGB–LiDAR Early Fusion '
             'and Depth-Gated Voting for Autonomous Navigation')

NEW_ABSTRACT = (
    'Point-cloud semantic segmentation is a core perception task for autonomous driving, yet its value usually '
    'stops at per-point labels: the rare but safety-critical classes (pedestrians, animals) are nearly wiped out '
    'by the long-tail distribution, and the labels themselves cannot tell a planner what happens upon physical '
    'contact. We present a semantic segmentation method oriented to autonomous navigation with three components. '
    '(1) A PointMambaSeg front-end fuses camera RGB with LiDAR XYZ through an early-fusion six-channel input and '
    'Mamba sequence modeling; a strictly paired ablation shows that RGB contributes +0.0331 mIoU (0.3923 mIoU on '
    'nuScenes), with the gain concentrated on geometrically similar classes separable only by appearance. '
    '(2) To recover the rare lethal classes, 2D detection with depth-gated 2D/3D voting lowers the critical_rate '
    '(lethal points misclassified as collidable) from 5.30% to 3.80%, with a GT-oracle upper bound of 0.92%. '
    '(3) The 32 nuScenes classes are mapped to physical interaction attributes (collision-safety level, mass, '
    'friction, restitution, destructibility) as a semantic enhancement for downstream decision, and the result is '
    'exported as a continuous semantic soft-cost layer for a ROS2 navigation stack. A Gazebo closed-loop simulation '
    'and a Semantic3D real-map experiment verify that the planner, using only this soft cost, bypasses a lethal '
    'pedestrian while driving through traversable vegetation.'
)

INTRO_PARAS = [
    ('Three-dimensional point-cloud semantic segmentation is a foundational perception task for autonomous driving '
     'and robotics. LiDAR provides accurate geometry but no appearance; cameras provide dense colour but suffer from '
     'depth ambiguity. Fusing the two sensors, especially through early fusion where colour is injected into the '
     'point features before segmentation, is therefore a central design question, and its benefit is not uniform '
     'across classes: colour matters only where geometry is insufficient to tell classes apart. At the same time, '
     'segmentation performance is dominated by the long-tail class distribution, so the rare but safety-critical '
     'classes (pedestrians, animals) are the most error-prone, and their misclassification is exactly the error that '
     'a navigation-safety decision can least afford.'),
    ('Beyond per-point labels, a navigation system that must act under loss of control (brake failure, road '
     'slippage) needs more than "what is ahead": it needs "what happens on impact". Per-point semantics must '
     'therefore be translated into physical interaction attributes, collision-safety level, mass, friction, '
     'restitution and destructibility, so that a planner can prefer a low-cost soft-collision target (a bush, a '
     'traffic cone, a guardrail) over a lethal or rigid one. This turns segmentation from a purely '
     'classification task into a sensor-driven physical-reasoning pipeline for autonomous navigation.'),
    ('Existing segmentation research aims to improve mIoU and stops at per-point labels, leaving two gaps open: the '
     'rare lethal classes are unreliable, and the labels carry no physical meaning. We address both within one '
     'method, named SemPhysPC (Semantic Physical Point Cloud). The main contributions are as follows:'),
    ('1. An early-fusion segmentation front-end: a PointMambaSeg network with a six-channel RGB+XYZ input, whose '
     'strictly paired ablation quantifies the RGB contribution (+0.0331 mIoU) and localizes it to '
     'appearance-separable classes.'),
    ('2. A depth-gated 2D/3D voting mechanism: to recover the rare lethal classes, 2D detection boxes are '
     'associated with 3D points under a depth gate, lowering the critical_rate (lethal misclassified as collidable) '
     'from 5.30% to 3.80%, with a GT-oracle upper bound of 0.92%.'),
    ('3. A semantic-to-physical attribute mapping: the 32 nuScenes classes are mapped to a collision-safety level '
     'and a physical material (mass, friction, restitution, destructibility), giving every point a lethality cost '
     'that a planner can consume.'),
    ('4. A navigation-oriented downstream application: the per-point lethality is exported as a continuous semantic '
     'soft-cost layer integrated into a ROS2 navigation stack, validated in a Gazebo closed-loop simulation and a '
     'Semantic3D real-map experiment.'),
    ('In the experiments, we validate the segmentation front-end on nuScenes (0.3923 mIoU, with the point-upscaling '
     'ablation reported honestly), the depth-gated voting mechanism, the physical-attribute mapping, and the '
     'navigation closed-loop application.'),
]

RW_MERGED = (
    '2.2 Physical-attribute mapping and navigation-safety application. Beyond per-point labels, a class-to-property '
    'mapping for physical interaction (collision safety, mass, friction, destructibility) still lacks a systematic '
    'definition, while most point-cloud material estimation targets visual rendering [6]. Collision-consequence '
    'assessment usually relies on physical simulators or fracture models [7], which are hard to construct from '
    'sparse point clouds, and general-purpose physics engines (e.g., PyBullet) exhibit systematic momentum '
    'non-conservation. Existing emergency-avoidance methods are mostly geometric/dynamic [8], treating obstacles as '
    'impenetrable rigid bodies. We bridge these lines by mapping semantics to physical attributes and feeding a '
    'lethality cost to a navigation planner, so that under loss of control it can prefer a low-cost soft-collision '
    'target.'
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


def find_para(doc, prefix):
    for p in doc.paragraphs:
        if p.text.strip().startswith(prefix):
            return p
    return None


def delete_paras_between(doc, start_prefix, end_prefix):
    """删除 start_prefix 标题之后、end_prefix 标题之前的正文段落（不含两个标题）。返回 end 标题段。"""
    body = doc.element.body
    els = list(body)
    start_i = end_i = None
    for i, el in enumerate(els):
        t = text(el).strip()
        if start_i is None and t.startswith(start_prefix):
            start_i = i
        elif start_i is not None and t.startswith(end_prefix):
            end_i = i
            break
    assert start_i is not None and end_i is not None, f'{start_prefix}/{end_prefix} 未找到'
    for el in els[start_i + 1:end_i]:
        if el.tag == qn('w:p'):
            el.getparent().remove(el)
    return end_i


def insert_paras_before(anchor_para, paras):
    for txt in paras:
        p = anchor_para.insert_paragraph_before()
        p.paragraph_format.line_spacing = 1.5
        set_para_text(p, txt)


def main():
    shutil.copy2(SRC, BAK)
    doc = docx.Document(SRC)

    # ---- 标题 ----
    title = doc.paragraphs[0]
    set_para_text(title, NEW_TITLE, size=14, bold=True)

    # ---- 摘要 ----
    abs_p = find_para(doc, 'In loss-of-control') or find_para(doc, 'Point-cloud semantic segmentation')
    assert abs_p is not None, '摘要段未找到'
    set_para_text(abs_p, NEW_ABSTRACT)

    # ---- Introduction 重写 ----
    intro_title = find_para(doc, '1 Introduction')
    rw_title = find_para(doc, '2 Related Work')
    assert intro_title is not None and rw_title is not None
    delete_paras_between(doc, '1 Introduction', '2 Related Work')
    insert_paras_before(rw_title, INTRO_PARAS)

    # ---- Related Work：2.2 合并、删 2.3/2.4 ----
    p22 = find_para(doc, '2.2 Semantic-to-material')
    assert p22 is not None, '2.2 未找到'
    set_para_text(p22, RW_MERGED)
    # 删除 2.3 段、2.4 段（标题+正文合一段）
    for el in list(doc.element.body):
        t = text(el).strip()
        if el.tag == qn('w:p') and (t.startswith('2.3 Impact') or t.startswith('2.4 Navigation')):
            el.getparent().remove(el)

    doc.save(SRC)

    # ---- 验证 ----
    d2 = docx.Document(SRC)
    paras = d2.paragraphs
    print('标题:', paras[0].text[:60])
    for p in paras:
        if p.text.strip().startswith('Point-cloud semantic segmentation is a core'):
            w = len(re.findall(r"[A-Za-z0-9.]+(?:-[A-Za-z0-9.]+)*", p.text))
            print(f'摘要词数: {w}')
            break
    # Introduction 贡献存在
    joined = '\n'.join(p.text for p in paras)
    print('贡献1-4存在:', all(s in joined for s in ['early-fusion segmentation front-end',
          'depth-gated 2D/3D voting', 'semantic-to-physical attribute mapping', 'navigation-oriented downstream']))
    print('2.3/2.4 已删:', ('2.3 Impact' not in joined) and ('2.4 Navigation' not in joined))
    print('2.2 合并段存在:', 'Physical-attribute mapping and navigation-safety application' in joined)


if __name__ == '__main__':
    main()
