"""统计 V3 训练数据配比（模拟 build_dataset.py 的裁剪逻辑）"""
import os
import random

random.seed(42)

files = {
    '中文维基': ('data/raw/wiki_clean_v3.txt', 0.38),
    '中文新闻': ('data/raw/news_clean.txt', 0.29),
    '百度百科': ('data/raw/baike_clean.txt', 0.21),
    '社区问答': ('data/raw/qa_clean.txt', 0.12),
}

total_target_chars = 1.2 * 1e9 * 1.4

totals = {}
actual_chars = 0
actual_lines = 0
for name, (path, r) in files.items():
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    src_chars = sum(len(l) for l in lines)
    target = total_target_chars * r
    if src_chars > target:
        indices = list(range(len(lines)))
        random.shuffle(indices)
        cum = 0
        cut = len(indices)
        for i, j in enumerate(indices):
            cum += len(lines[j])
            if cum >= target:
                cut = i + 1
                break
        kept = sorted(indices[:cut])
        kept_lines = [lines[i] for i in kept]
        kept_chars = sum(len(l) for l in kept_lines)
        totals[name] = (len(kept_lines), kept_chars)
    else:
        totals[name] = (len(lines), src_chars)
        kept_chars = src_chars
    actual_chars += kept_chars
    actual_lines += totals[name][0]

print('V3 训练数据配比')
print('=' * 68)
print(f'{"数据源":<10} {"行数":<10} {"字符(M)":<10} {"字符占比":<8} {"tokens(B)":<10}')
print('-' * 68)
for name, (lines, chars) in totals.items():
    ratio = chars / actual_chars * 100
    tk = chars / 1.4 / 1e9
    print(f'{name:<8}  {lines:<8,}  {chars/1e6:>8.1f}   {ratio:>5.1f}%     {tk:>6.2f}')
print('-' * 68)
print(f'合计      {actual_lines:<8,}  {actual_chars/1e6:>8.1f}   100.0%     {actual_chars/1.4/1e9:>6.2f}')
print()
print(f'train.txt: {int(actual_lines*0.9):,} 行 (90%) = {actual_chars*0.9/1e9:.2f}B 字符')
print(f'valid.txt: {int(actual_lines*0.05):,} 行 (5%)')
print(f'test.txt:  {int(actual_lines*0.05):,} 行 (5%)')
