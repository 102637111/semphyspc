#!/usr/bin/env python3
"""把「门控投票补短板」小节（§4.2.3）插入权威中文版与英文投稿版 docx。

插入位置：§4.2.2 之后、§4.3 之前（以「4.3 」开头的标题段落为 anchor）。
内容：标题 + 4 段正文 + 表 caption + 一张 7×6 的 Table Grid 表（表 2d）。
两个版本各自用本版现有格式约定。

用法（项目根 /home/saw/ros2_ws/src/semphyspc/ 下）：
    python3 scripts/insert_gated_voting_section.py
"""
import shutil

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt

# ---------------- 中文版文案 ----------------
ZH = dict(
    path='docs/语义地图.docx',
    anchor='4.3 物理属性映射结果',
    heading='4.2.3 致命类补短板：2D 检测 + 深度门控投票',
    paras=[
        ('§4.2 已显示罕见 thing 类近乎全灭，其中致命类（行人、动物）的漏检对导航安全最致命：'
         '一旦把行人误判为「可撞类」（可破坏物 frangible 与可变形植被 deformable 的并集，正是软碰撞决策 §3.5 推荐撞击的候选集），'
         '决策层就可能选中不可接受的目标。为量化这一风险，定义 critical_rate 为「致命类 GT 点被判为可撞类的比例」。'
         '为让罕见行人子类至少落入正确的「person」桶、并提升 person 类召回，先按安全等级把 32 类合并为 12 类'
         '（person 子类 2–8 合并，见 §3.3），得到 merged 分割前置：mIoU 0.620、person IoU 0.545，'
         '但 critical_rate 仍有 5.30%（251,219 个致命点中 13,319 点落入可撞类）。'),
        ('值得注意的是，单纯提升分割精度并不能降低 critical_rate——12 类合并较之 32 类前置，'
         '同口径 mIoU 提升 0.017、person IoU 提升 0.011，critical_rate 却不降反微升（5.18%→5.30%），'
         '说明「识别更准」与「消除致命安全误判」是两个不同的目标。为此引入 2D–3D 深度门控投票：'
         '在 6 相机上运行 RT-DETR 行人检测器，对 3D 分割置信度低（softmax 最大概率低于阈值 θ）的致命类点，'
         '若其投影落入 2D 检测框、且深度处于该框的深度范围内（深度门控，用以消除前景/背景歧义），则回填为 person。'
         '6 相机拼接覆盖了 86.9% 的致命实例（单相机仅 12.6%），深度门控把回填的精确率从 0.358 提到 0.701。'),
        ('表 2d 给出不同置信度阈值 θ 的扫描。θ=0.9 时 critical_rate 从 5.30% 降到 3.80%（相对 −28%），'
         'severe 从 3.40% 降到 1.58%，safe 从 90.5% 升到 94.3%，代价是 person IoU 从 0.545 略降至 0.520'
         '（门控覆写了个别本已正确的 3D 点）。θ=0.7 是「安全—精度」平衡点（critical 4.01%、person IoU 0.559）。'
         'θ=1.0（完全信任 2D、放弃 3D 门控）时 person IoU 崩到 0.332、critical 反升到 4.43%——'
         '证明 3D 置信度门控不可省，纯 2D 投票会把背景误判为行人。'),
        ('用 GT 检测框替换 RT-DETR 预测框可得机制上界：θ=0.9 时 critical 仅 0.92%、person IoU 0.713、mIoU 0.662。'
         '真实检测器与上界间约 2.9 个百分点的差距几乎全部来自检测器召回，说明当前 3.80% 的下限受 RT-DETR 所限、'
         '而非门控机制本身；换用更强的检测器可进一步逼近 0.92%。这一补短板机制与分割前置互补：'
         '低 mIoU 的损失集中在罕见 thing 类，而这些类恰是导航安全最敏感的目标，门控投票以 2D 视觉冗余把它们拉回软碰撞决策的可用区间。'
         '其余规模/颜色方向的改进（num_group 1024→6144 再 +0.0154、RGB 深度消融）见补充材料。'),
    ],
    caption=('表 2d　门控投票在不同置信度阈值 θ 下的安全指标（12 类 merged 前置，full val；'
             'critical = 致命类被判为可撞类、severe = 致命类被判为地面、safe = 致命类被正确判为致命或硬类）'),
    header=['配置', 'critical_rate', 'severe_rate', 'safe_rate', 'person IoU', 'mIoU'],
    rows=[
        ['无门控（merged base）', '5.30%', '3.40%', '90.49%', '0.545', '0.620'],
        ['RT-DETR θ=0.5', '4.68%', '2.90%', '91.86%', '0.564', '0.624'],
        ['RT-DETR θ=0.7', '4.01%', '2.23%', '93.36%', '0.559', '0.621'],
        ['RT-DETR θ=0.9', '3.80%', '1.58%', '94.26%', '0.520', '0.612'],
        ['RT-DETR θ=1.0', '4.43%', '0.93%', '94.29%', '0.332', '0.571'],
        ['GT oracle θ=0.9', '0.92%', '1.18%', '97.67%', '0.713', '0.662'],
    ],
    bold_row=4,  # RT-DETR θ=0.9 主结果加粗（data rows 的 1-based 行号：base=1,0.5=2,0.7=3,0.9=4,1.0=5,oracle=6）
)

# ---------------- 英文版文案 ----------------
EN = dict(
    path='docs/paper_draft_semphyspc.docx',
    anchor='4.3 Physical-Property Mapping Results',
    heading='4.2.3 Closing the Lethal-Class Gap: 2D Detection with Depth-Gated Voting',
    paras=[
        ('Section 4.2 already shows that the rare thing classes are nearly wiped out, and among them the missed '
         'detection of lethal classes (pedestrians, animals) is the most safety-critical: once a pedestrian is '
         'misclassified into the "collidable" set (the union of frangible objects and deformable vegetation, which is '
         'exactly the candidate set that the soft-collision decision of Section 3.5 recommends for impact), the '
         'decision layer may select an unacceptable target. To quantify this risk, we define critical_rate as the '
         'fraction of lethal-class GT points that are classified into the collidable set. To make the rare pedestrian '
         'sub-classes at least fall into the correct "person" bucket and to raise person recall, we first merge the 32 '
         'classes into 12 by safety level (pedestrian sub-classes 2–8 are merged, see Section 3.3), yielding the '
         'merged front-end: mIoU 0.620, person IoU 0.545, but critical_rate still 5.30% (13,319 of 251,219 lethal '
         'points fall into the collidable set).'),
        ('Notably, raising segmentation accuracy alone does not lower critical_rate — relative to the 32-class '
         'front-end, the 12-class merge raises mIoU by 0.017 and person IoU by 0.011, yet critical_rate even rises '
         'slightly (5.18%→5.30%), showing that "recognizing more accurately" and "eliminating lethal safety '
         'misclassification" are two distinct goals. We therefore introduce 2D–3D depth-gated voting: an RT-DETR '
         'pedestrian detector runs on the six cameras, and for each lethal-class point whose 3D segmentation '
         'confidence is low (softmax maximum probability below a threshold θ), if its projection falls inside a 2D '
         'detection box and its depth lies within the box’s depth range (depth gating, to remove foreground/'
         'background ambiguity), the point is back-filled as person. The six-camera coverage reaches 86.9% of lethal '
         'instances (a single camera covers only 12.6%), and depth gating raises the back-fill precision from 0.358 '
         'to 0.701.'),
        ('Table 2d shows the sweep over the confidence threshold θ. At θ = 0.9, critical_rate drops from 5.30% to '
         '3.80% (a relative −28%), severe from 3.40% to 1.58%, and safe rises from 90.5% to 94.3%, at the cost of '
         'person IoU falling slightly from 0.545 to 0.520 (the gating overwrites a few points that 3D had already '
         'classified correctly). θ = 0.7 is the safety–accuracy balance point (critical 4.01%, person IoU 0.559). '
         'At θ = 1.0 (trusting 2D entirely and dropping the 3D gate), person IoU collapses to 0.332 and critical '
         'rebounds to 4.43% — proving that the 3D confidence gate is indispensable, since pure 2D voting mislabels '
         'background as pedestrians.'),
        ('Replacing the RT-DETR boxes with GT boxes gives the mechanism’s upper bound: at θ = 0.9, critical is only '
         '0.92%, with person IoU 0.713 and mIoU 0.662. The ≈2.9 pp gap between the real detector and this bound '
         'comes almost entirely from detector recall, indicating that the current 3.80% floor is limited by RT-DETR '
         'rather than by the gating mechanism itself; a stronger detector can approach 0.92%. This gap-closing '
         'mechanism complements the segmentation front-end: the mIoU loss is concentrated in the rare thing classes, '
         'which are exactly the targets most sensitive to navigation safety, and gated voting pulls them back into the '
         'usable range of soft-collision decision-making with 2D visual redundancy. The remaining scale/colour '
         'improvements (num_group 1024→6144, an additional +0.0154; RGB depth ablation) are in the supplementary '
         'material.'),
    ],
    caption=('Table 2d. Safety metrics of gated voting at different confidence thresholds θ (12-class merged '
             'front-end, full val; critical = lethal classified as collidable, severe = lethal classified as ground, '
             'safe = lethal correctly classified as lethal or hard).'),
    header=['Configuration', 'critical_rate', 'severe_rate', 'safe_rate', 'person IoU', 'mIoU'],
    rows=[
        ['No gating (merged base)', '5.30%', '3.40%', '90.49%', '0.545', '0.620'],
        ['RT-DETR θ=0.5', '4.68%', '2.90%', '91.86%', '0.564', '0.624'],
        ['RT-DETR θ=0.7', '4.01%', '2.23%', '93.36%', '0.559', '0.621'],
        ['RT-DETR θ=0.9', '3.80%', '1.58%', '94.26%', '0.520', '0.612'],
        ['RT-DETR θ=1.0', '4.43%', '0.93%', '94.29%', '0.332', '0.571'],
        ['GT oracle θ=0.9', '0.92%', '1.18%', '97.67%', '0.713', '0.662'],
    ],
    bold_row=4,  # RT-DETR θ=0.9 主结果加粗
)


def _set_run(r, size_pt, bold=False):
    r.font.name = 'Times New Roman'
    if size_pt is not None:
        r.font.size = Pt(size_pt)
    r.font.bold = bold
    rf = r._element.get_or_add_rPr().get_or_add_rFonts()
    rf.set(qn('w:eastAsia'), '宋体')


def insert_section(cfg, size_body, size_heading, size_caption):
    """在 anchor 段之前插入小节，返回 (插入段落数, 表格行数)。"""
    shutil.copy2(cfg['path'], cfg['path'] + '.bak-gatedvoting')
    doc = docx.Document(cfg['path'])

    anchor = None
    for p in doc.paragraphs:
        if p.text.strip().startswith(cfg['anchor']):
            anchor = p
            break
    assert anchor is not None, f'anchor 未找到: {cfg["anchor"]}'

    # 标题（insert_paragraph_before 不传 text，避免与手动 add_run 重复）
    h = anchor.insert_paragraph_before()
    h.paragraph_format.space_before = Pt(6)
    h.paragraph_format.space_after = Pt(6)
    _set_run(h.add_run(cfg['heading']), size_heading, bold=True)

    # 正文 4 段
    for text in cfg['paras']:
        p = anchor.insert_paragraph_before()
        p.paragraph_format.first_line_indent = Pt(21)
        p.paragraph_format.space_after = Pt(3)
        _set_run(p.add_run(text), size_body, bold=False)

    # 表 caption
    cap = anchor.insert_paragraph_before()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_run(cap.add_run(cfg['caption']), size_caption, bold=True)

    # 表格
    nrows = 1 + len(cfg['rows'])
    tbl = doc.add_table(rows=nrows, cols=len(cfg['header']))
    tbl.style = 'Table Grid'
    # 表头
    for ci, htext in enumerate(cfg['header']):
        cell = tbl.rows[0].cells[ci]
        cell.text = ''
        _set_run(cell.paragraphs[0].add_run(htext), 9, bold=True)
    # 数据行
    for ri, row in enumerate(cfg['rows'], start=1):
        for ci, val in enumerate(row):
            cell = tbl.rows[ri].cells[ci]
            cell.text = ''
            _set_run(cell.paragraphs[0].add_run(val), 9, bold=(ri == cfg['bold_row']))

    # 把表格移到 anchor 之前（caption 之后）
    anchor._p.addprevious(tbl._tbl)
    doc.save(cfg['path'])
    return nrows


def main():
    # 中文版：标题 11pt、正文 10.5pt、caption 9pt
    nrows_zh = insert_section(ZH, size_body=10.5, size_heading=11, size_caption=9)
    # 英文版：标题/正文/caption 继承默认 10.5pt（与英文版现有格式一致）
    nrows_en = insert_section(EN, size_body=10.5, size_heading=10.5, size_caption=9)

    # 验证
    for cfg in (ZH, EN):
        d = docx.Document(cfg['path'])
        print(f'\n=== {cfg["path"]} ===')
        print(f'表格数：{len(d.tables)}')
        # 定位插入的标题
        for p in d.paragraphs:
            if p.text.strip() == cfg['heading']:
                print(f'  标题已插入: {p.text.strip()[:50]}')
                break
        # 打印插入小节的段落序列
        found = False
        for p in d.paragraphs:
            t = p.text.strip()
            if t == cfg['heading']:
                found = True
            if found:
                if t.startswith('4.3 '):
                    break
                if t:
                    print(f'  · {t[:60]}')


if __name__ == '__main__':
    main()
