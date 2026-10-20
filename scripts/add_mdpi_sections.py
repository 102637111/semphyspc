#!/usr/bin/env python3
"""Remote Sensing 重定位（阶段 3a）：MDPI 强制版块。

- 删除 IEEE 风格的 Data Availability: / Code Availability: 两段
- 在 References 之后补 MDPI 标准版块（按官方顺序）：
    Author Contributions / Funding / Institutional Review Board Statement /
    Informed Consent Statement / Data Availability Statement / Conflicts of Interest
- 参考文献保留现有 [1]..[8] 编号格式（MDPI 投稿接受，接受后编辑部统一排版）

用法（项目根下）：python3 scripts/add_mdpi_sections.py
"""
import shutil

import docx
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

SRC = 'docs/paper_draft_semphyspc.docx'
BAK = SRC + '.bak-mdpi-sections'

AUTH_CONTRIB = (
    'Conceptualization, [A.B.]; methodology, [A.B.]; software, [A.B.]; validation, [A.B.]; formal analysis, '
    '[A.B.]; investigation, [A.B.]; resources, [A.B.]; data curation, [A.B.]; writing—original draft '
    'preparation, [A.B.]; writing—review and editing, [A.B.]; visualization, [A.B.]; supervision, [A.B.]; '
    'project administration, [A.B.]; funding acquisition, [A.B.]. All authors have read and agreed to the '
    'published version of the manuscript.'
)
FUNDING = 'This research received no external funding.'
IRB = 'Not applicable.'
INFORMED = 'Not applicable.'
DATA_AVAIL = (
    'The nuScenes dataset is publicly available at https://www.nuscenes.org, and the Semantic3D dataset is '
    'publicly available at http://www.semantic3d.net. The semantic-to-physical mapping library, trained model '
    'checkpoints, and source code will be made available at https://github.com/[repository] upon publication.'
)
CONFLICTS = 'The authors declare no conflicts of interest.'

SECTIONS = [
    ('Author Contributions', AUTH_CONTRIB),
    ('Funding', FUNDING),
    ('Institutional Review Board Statement', IRB),
    ('Informed Consent Statement', INFORMED),
    ('Data Availability Statement', DATA_AVAIL),
    ('Conflicts of Interest', CONFLICTS),
]


def text(el):
    ts = el.findall('.//' + qn('w:t'))
    return ''.join(t.text or '' for t in ts)


def find_body_para(doc, prefix):
    for el in doc.element.body:
        if el.tag == qn('w:p') and text(el).strip().startswith(prefix):
            return el
    return None


def make_run(txt, bold=False, size=12):
    r = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    rf = OxmlElement('w:rFonts')
    rf.set(qn('w:ascii'), 'Times New Roman'); rf.set(qn('w:hAnsi'), 'Times New Roman')
    rf.set(qn('w:eastAsia'), '宋体')
    rPr.append(rf)
    if bold:
        rPr.append(OxmlElement('w:b'))
    sz = OxmlElement('w:sz'); sz.set(qn('w:val'), str(size * 2)); rPr.append(sz)
    szCs = OxmlElement('w:szCs'); szCs.set(qn('w:val'), str(size * 2)); rPr.append(szCs)
    r.append(rPr)
    t = OxmlElement('w:t'); t.text = txt
    t.set(qn('xml:space'), 'preserve')
    r.append(t)
    return r


def make_mdpi_section(title, content):
    p = OxmlElement('w:p')
    pPr = OxmlElement('w:pPr')
    spacing = OxmlElement('w:spacing')
    spacing.set(qn('w:line'), '360'); spacing.set(qn('w:lineRule'), 'auto')
    pPr.append(spacing)
    p.append(pPr)
    p.append(make_run(title + ': ', bold=True))
    p.append(make_run(content))
    return p


def main():
    shutil.copy2(SRC, BAK)
    doc = docx.Document(SRC)
    body = doc.element.body

    el_da = find_body_para(doc, 'Data Availability:')
    el_code = find_body_para(doc, 'Code Availability:')
    el_refs = find_body_para(doc, 'References')
    assert el_da is not None and el_code is not None and el_refs is not None, '定位失败'

    # 1. 删除旧 Data/Code Availability 两段
    el_da.getparent().remove(el_da)
    el_code.getparent().remove(el_code)

    # 2. 在 body 级 sectPr 之前插入 MDPI 版块（顺序正确）
    sectPr = body.find(qn('w:sectPr'))
    assert sectPr is not None, '未找到 body 级 sectPr'
    for title, content in SECTIONS:
        sectPr.addprevious(make_mdpi_section(title, content))

    doc.save(SRC)

    # ---- 验证 ----
    d2 = docx.Document(SRC)
    joined = '\n'.join(p.text for p in d2.paragraphs)
    checks = {
        '旧 Data Availability 已删': 'Data Availability: The nuScenes' not in joined,
        '旧 Code Availability 已删': 'Code Availability: The source' not in joined,
        'Author Contributions': 'Author Contributions:' in joined,
        'Funding': 'Funding:' in joined,
        'IRB': 'Institutional Review Board Statement:' in joined,
        'Informed Consent': 'Informed Consent Statement:' in joined,
        'Data Availability Statement': 'Data Availability Statement:' in joined,
        'Conflicts of Interest': 'Conflicts of Interest:' in joined,
    }
    for k, v in checks.items():
        print(f'{k}: {"OK" if v else "FAIL"}')
    # 打印 References 之后的版块顺序
    print('--- References 后版块顺序 ---')
    start = False
    for p in d2.paragraphs:
        t = p.text.strip()
        if t == 'References':
            start = True
            continue
        if start and t:
            print(f'  {t[:60]}')


if __name__ == '__main__':
    main()
