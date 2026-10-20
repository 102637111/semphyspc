#!/usr/bin/env python3
"""把英文投稿版里的宽图/宽表用连续分节符包裹实现双栏跨栏。

原理：宽元素前插入一个「单栏」连续分节段（段落级 w:sectPr, w:cols num=1），
宽元素后插入一个「双栏」连续分节段（w:cols num=2），使宽元素处于全宽单栏区间。

只改 docs/paper_draft_semphyspc.docx（先备份 .bak-span）。

用法（项目根下）：python3 scripts/span_columns.py
"""
import copy
import shutil

import docx
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

SRC = 'docs/paper_draft_semphyspc.docx'
BAK = SRC + '.bak-span'

# (start_body_idx, end_body_idx) 均为 body 直接子元素索引，含 caption 与表格/图片本体
REGIONS = [
    (27, 28),    # 图1 流水线（IMG + caption）
    (39, 40),    # 表1 物理材质库（7 列）
    (58, 59),    # 表2 逐类 IoU（6 列）
    (61, 62),    # 表2 overall metric summary（6 列）
    (90, 91),    # 表2d 门控投票（6 列）
    (94, 95),    # 表3 语义→物理映射（5 列）
    (105, 106),  # 表4 减速标定（6 列）
    (129, 130),  # 图4 系统集成（IMG + caption）
]


def make_break_para(sectPr_template, num_cols):
    """构造一个带连续分节符的紧凑空段落（单栏 num=1 / 双栏 num=2）。"""
    sp = copy.deepcopy(sectPr_template)
    # 显式连续分节
    t = sp.find(qn('w:type'))
    if t is None:
        t = OxmlElement('w:type')
        sp.insert(0, t)
    t.set(qn('w:val'), 'continuous')
    # 改栏数
    cols = sp.find(qn('w:cols'))
    if cols is None:
        cols = OxmlElement('w:cols')
        sp.append(cols)
    cols.set(qn('w:num'), str(num_cols))

    p = OxmlElement('w:p')
    pPr = OxmlElement('w:pPr')
    spacing = OxmlElement('w:spacing')
    spacing.set(qn('w:before'), '0')
    spacing.set(qn('w:after'), '0')
    pPr.append(spacing)
    pPr.append(sp)          # w:sectPr 必须为 pPr 最后一个子元素
    p.append(pPr)
    # 1pt 空 run，压缩空隙
    r = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    sz = OxmlElement('w:sz'); sz.set(qn('w:val'), '2')
    szCs = OxmlElement('w:szCs'); szCs.set(qn('w:val'), '2')
    rPr.append(sz); rPr.append(szCs)
    r.append(rPr)
    p.append(r)
    return p


def main():
    shutil.copy2(SRC, BAK)
    doc = docx.Document(SRC)
    body = doc.element.body
    els = list(body)
    sectPr = body.find(qn('w:sectPr'))
    assert sectPr is not None, '未找到 body 级 sectPr'

    single = make_break_para(sectPr, 1)   # 单栏：结束前面的单栏（跨栏）区间
    double = make_break_para(sectPr, 2)   # 双栏：结束前面的双栏正文区间

    # 段落级 sectPr 描述的是「它所在的、即将结束的 section」：
    #   before = 双栏（结束前面正文的双栏 section）
    #   after  = 单栏（结束跨栏元素的单栏 section）
    # 从后往前处理，用元素引用（不受前插影响）
    for start_idx, end_idx in sorted(REGIONS, reverse=True):
        start_el = els[start_idx]
        end_el = els[end_idx]
        start_el.addprevious(copy.deepcopy(double))
        end_el.addnext(copy.deepcopy(single))
        print(f'已跨栏 [{start_idx}..{end_idx}]')

    doc.save(SRC)

    # ---- 验证 ----
    d2 = docx.Document(SRC)
    body2 = d2.element.body
    break_count = 0
    for el in body2:
        if el.tag == qn('w:p'):
            sp = el.find(qn('w:pPr') + '/' + qn('w:sectPr'))
            if sp is not None:
                break_count += 1
    print(f'段落级分节符数量: {break_count}（预期 {len(REGIONS) * 2}）')


if __name__ == '__main__':
    main()
