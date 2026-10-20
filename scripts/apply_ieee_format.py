#!/usr/bin/env python3
"""把英文投稿版 docx 套用 IEEE Sensors Journal 双栏格式。

改动（只改 docs/paper_draft_semphyspc.docx，先备份 .bak-ieee）：
  1. 页面 A4 → US Letter 8.5×11 in；边距 上0.75/下1.0/左右0.625 in。
  2. 双栏：栏宽 3.5in（Word 自动）、栏间距 0.25in。
  3. Normal 样式 10.5pt → 10pt。
  4. 标题 24pt 居中；摘要/关键词 9pt；参考文献 8pt；caption 9pt；正文 10pt。
  5. 章节标题 10pt 小大写（smallCaps）；Abstract/References 标签同。
  6. 标题后插入作者占位（姓名/单位/邮箱/ORCID）。

用法（项目根下）：python3 scripts/apply_ieee_format.py
"""
import re
import shutil

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

SRC = 'docs/paper_draft_semphyspc.docx'
BAK = SRC + '.bak-ieee'

SECTION_PAT = re.compile(r'^\d+(\.\d+)*\s+[A-Z]')
REF_PAT = re.compile(r'^\[\d+\]')
CAPTION_PAT = re.compile(r'^(Table \d|Fig\. \d)')


def set_run(run, size, bold=None, small_caps=False):
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if small_caps:
        rpr = run._element.get_or_add_rPr()
        if rpr.find(qn('w:smallCaps')) is None:
            sc = OxmlElement('w:smallCaps')
            sc.set(qn('w:val'), 'true')
            rpr.append(sc)


def set_para_all_runs(p, size, bold=None, small_caps=False):
    for r in p.runs:
        set_run(r, size, bold=bold, small_caps=small_caps)


def insert_author_placeholder(abstract_para):
    """在 Abstract 标签前插入作者占位（3 行，11pt 居中）。"""
    def mk(text, bold=False):
        pp = abstract_para.insert_paragraph_before()
        pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pp.paragraph_format.space_after = Pt(2)
        set_run(pp.add_run(text), 11, bold=bold)
        return pp
    mk('[Author Name(s)]', bold=True)
    mk('[Department, University, City, Country]')
    mk('[email]  ·  ORCID: 0000-0000-0000-0000')


def main():
    shutil.copy2(SRC, BAK)
    doc = docx.Document(SRC)

    # ---- 页面 / 边距 / 双栏 ----
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(0.75)
    sec.bottom_margin = Inches(1.0)
    sec.left_margin = Inches(0.625)
    sec.right_margin = Inches(0.625)
    # 双栏：num=2, space=0.25in(=360 twips)
    sectPr = sec._sectPr
    cols = sectPr.find(qn('w:cols'))
    if cols is None:
        cols = OxmlElement('w:cols')
        sectPr.append(cols)
    cols.set(qn('w:num'), '2')
    cols.set(qn('w:space'), '360')

    # ---- Normal 样式 10pt ----
    doc.styles['Normal'].font.name = 'Times New Roman'
    doc.styles['Normal'].font.size = Pt(10)

    # ---- 遍历段落分类设置 ----
    paras = doc.paragraphs
    # 标题 = 第一个非空段落
    title_idx = next(i for i, p in enumerate(paras) if p.text.strip())
    title = paras[title_idx]
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para_all_runs(title, 24, bold=True)

    state = 'head'  # head / in_abstract
    for i, p in enumerate(paras):
        if i == title_idx:
            continue
        t = p.text.strip()
        if not t:
            continue
        # 作者占位：标题后第一个非空段如果是 Abstract，先插入作者
        if state == 'head' and t == 'Abstract':
            insert_author_placeholder(p)
            set_para_all_runs(p, 9, bold=True)
            state = 'in_abstract'
            continue
        if state == 'in_abstract':
            # 摘要正文（紧跟 Abstract 标签）
            set_para_all_runs(p, 9)
            state = 'head'
            continue
        if t.startswith('Keywords:'):
            set_para_all_runs(p, 9)  # 保留 run 各自 bold（Keywords 已加粗）
            continue
        if t == 'References':
            set_para_all_runs(p, 10, bold=True, small_caps=True)
            continue
        if REF_PAT.match(t):
            set_para_all_runs(p, 8)
            continue
        if SECTION_PAT.match(t):
            set_para_all_runs(p, 10, bold=True, small_caps=True)
            continue
        if CAPTION_PAT.match(t):
            set_para_all_runs(p, 9)
            continue
        # 其余正文
        set_para_all_runs(p, 10)

    doc.save(SRC)

    # ---- 验证 ----
    d2 = docx.Document(SRC)
    s2 = d2.sections[0]
    print(f'页面: {s2.page_width.inches:.2f}×{s2.page_height.inches:.2f} in')
    print(f'边距: 上{s2.top_margin.inches} 下{s2.bottom_margin.inches} 左{s2.left_margin.inches} 右{s2.right_margin.inches}')
    c2 = s2._sectPr.find(qn('w:cols'))
    print(f'cols: num={c2.get(qn("w:num"))} space={c2.get(qn("w:space"))} twips')
    # 标题与作者占位
    print('前 6 段:')
    for p in d2.paragraphs[:6]:
        print(f'  [{p.text.strip()[:45]!r}] align={p.alignment} size={[r.font.size.pt if r.font.size else None for r in p.runs]}')


if __name__ == '__main__':
    main()
