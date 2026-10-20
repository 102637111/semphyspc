#!/usr/bin/env python3
"""把英文投稿版从 IEEE 双栏格式重置为 MDPI Remote Sensing 单栏格式。

改动（先备份 .bak-mdpi）：
  1. 删除 16 个跨栏段落级 sectPr（连续分节符），body 级 sectPr cols num=2 -> 1。
  2. 页面 US Letter 8.5x11 -> A4（11906x16838 twips），边距四周 1in。
  3. 字号统一 12pt；标题 14pt 加粗居中；去掉 smallCaps。
  4. 行距 1.5 倍。

用法（项目根下）：python3 scripts/apply_mdpi_format.py
"""
import shutil

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt

SRC = 'docs/paper_draft_semphyspc.docx'
BAK = SRC + '.bak-mdpi'


def main():
    shutil.copy2(SRC, BAK)
    doc = docx.Document(SRC)
    body = doc.element.body

    # 1. 删除段落级分节段（从后往前，用元素引用）
    removed = 0
    for el in list(body):
        if el.tag == qn('w:p'):
            pPr = el.find(qn('w:pPr'))
            sp = pPr.find(qn('w:sectPr')) if pPr is not None else None
            if sp is not None:
                el.getparent().remove(el)
                removed += 1
    print(f'删除段落级分节段: {removed}')

    # 2. body 级 sectPr：单栏 + A4 + 边距
    sectPr = body.find(qn('w:sectPr'))
    cols = sectPr.find(qn('w:cols'))
    if cols is None:
        from docx.oxml import OxmlElement
        cols = OxmlElement('w:cols')
        sectPr.append(cols)
    cols.set(qn('w:num'), '1')
    cols.set(qn('w:space'), '0')
    pgSz = sectPr.find(qn('w:pgSz'))
    pgSz.set(qn('w:w'), '11906')   # A4 210mm
    pgSz.set(qn('w:h'), '16838')   # A4 297mm
    pgMar = sectPr.find(qn('w:pgMar'))
    for side in ('w:top', 'w:bottom', 'w:left', 'w:right'):
        pgMar.set(qn(side), '1440')  # 1in

    # 3. 字号 + 行距
    doc.styles['Normal'].font.size = Pt(12)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.5

    paras = doc.paragraphs
    title = paras[0]
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in title.runs:
        r.font.size = Pt(14)
        r.font.bold = True

    for i, p in enumerate(paras):
        if i == 0:
            continue
        if p.paragraph_format.line_spacing is None:
            p.paragraph_format.line_spacing = 1.5
        for r in p.runs:
            r.font.size = Pt(12)
            rpr = r._element.get_or_add_rPr()
            sc = rpr.find(qn('w:smallCaps'))
            if sc is not None:
                rpr.remove(sc)

    doc.save(SRC)

    # ---- 验证 ----
    d2 = docx.Document(SRC)
    body2 = d2.element.body
    sectPr2 = body2.find(qn('w:sectPr'))
    cols2 = sectPr2.find(qn('w:cols'))
    pgSz2 = sectPr2.find(qn('w:pgSz'))
    nbreak = sum(1 for el in body2 if el.tag == qn('w:p') and el.find(qn('w:pPr')) is not None
                 and el.find(qn('w:pPr')).find(qn('w:sectPr')) is not None)
    print(f'cols num={cols2.get(qn("w:num"))} (期望 1)')
    print(f'pgSz {pgSz2.get(qn("w:w"))}x{pgSz2.get(qn("w:h"))} (期望 11906x16838)')
    print(f'残留段落级分节段: {nbreak} (期望 0)')
    t = d2.paragraphs[0]
    print(f'标题字号: {[r.font.size.pt for r in t.runs]} (期望 [14.0])')


if __name__ == '__main__':
    main()
