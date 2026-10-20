#!/usr/bin/env python3
"""在结论段内补一句门控投票成果（致命类误判 5.30%→3.80%），与 §4.2.3 呼应。

在「置信度门控/拒识」成果之后、「撞击后果标定」之前插入，衔接原文并列结构。
锚点必须完整落在单个 run 内，否则跳过并报错。

用法（项目根下）：python3 scripts/sync_conclusion_gated_voting.py
"""
import shutil

import docx

JOBS = [
    dict(
        path='docs/语义地图.docx',
        para_prefix='本文针对',
        anchor='此外对撞击后果做了闭合形式标定',
        insert=('针对罕见类全灭导致的致命类安全误判，引入 2D 检测 + 深度门控投票，'
                '把致命类被误判为可撞类的 critical_rate 从 5.30% 压到 3.80%'
                '（GT oracle 上界 0.92%）；'),
    ),
    dict(
        path='docs/paper_draft_semphyspc.docx',
        para_prefix='Targeting the bottleneck',
        anchor='in addition, impact consequences are calibrated in closed form',
        insert=('targeting the lethal-class safety misclassification caused by the rare-class wipe-out, '
                '2D detection with depth-gated voting pushes the critical_rate (lethal misclassified as '
                'collidable) down from 5.30% to 3.80%, with a GT-oracle upper bound of 0.92%; '),
    ),
]


def main():
    for job in JOBS:
        shutil.copy2(job['path'], job['path'] + '.bak-concl')
        doc = docx.Document(job['path'])
        done = False
        for p in doc.paragraphs:
            if p.text.strip().startswith(job['para_prefix']):
                for r in p.runs:
                    if job['anchor'] in r.text:
                        r.text = r.text.replace(job['anchor'], job['insert'] + job['anchor'], 1)
                        done = True
                        break
                break
        assert done, f"{job['path']}: 锚点未在单个 run 内找到"
        doc.save(job['path'])

        # 验证
        d2 = docx.Document(job['path'])
        for p in d2.paragraphs:
            if p.text.strip().startswith(job['para_prefix']):
                assert '5.30%' in p.text and '3.80%' in p.text, '结论同步缺失'
                print(f"完成：{job['path']} 结论已补门控投票句")
                break


if __name__ == '__main__':
    main()
