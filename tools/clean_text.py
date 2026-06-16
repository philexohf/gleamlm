"""文本清洗。去 HTML、URL、空白，过滤短/纯符号行，可选简繁转换"""

import re
import os
import argparse

# 简繁转换
try:
    import zhconv
    HAS_ZhCONV = True
except ImportError:
    HAS_ZhCONV = False
    print("提示: pip install zhconv 可启用简繁转换")


def clean_text(text, min_len=10, max_len=2000, convert_zh=False):
    """
    清洗单条文本

    清洗规则：
        - 去除 HTML 标签
        - 去除多余空白
        - 过滤过短/过长文本
        - 过滤纯数字/符号行
        - 统一标点符号
        - 简繁转换（可选）
    """
    if not text or not text.strip():
        return None

    # 简繁转换（需 pip install zhconv）
    if convert_zh and HAS_ZhCONV:
        text = zhconv.convert(text, 'zh-cn')

    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)

    # 去除 URL
    text = re.sub(r'https?://\S+', '', text)

    # 统一空白字符
    text = re.sub(r'\s+', ' ', text).strip()

    # 过滤长度
    if len(text) < min_len or len(text) > max_len:
        return None

    # 过滤纯数字/符号行（中文/英文占比过低）
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    if chinese_chars + english_chars < len(text) * 0.3:
        return None

    return text


def clean_file(input_path, output_path, min_len=10, max_len=2000, convert_zh=False):
    """
    清洗整个文本文件

    Args:
        input_path: 输入文件路径（每行一条文本）
        output_path: 输出文件路径
        min_len: 最小文本长度
        max_len: 最大文本长度
        convert_zh: 是否自动转换繁体到简体
    """
    total = 0
    kept = 0

    if convert_zh and not HAS_ZhCONV:
        print("WARNING: zhconv 未安装，简繁转换已跳过。pip install zhconv")

    print(f"Cleaning: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as fin:
        with open(output_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                total += 1
                cleaned = clean_text(line, min_len, max_len, convert_zh)
                if cleaned:
                    fout.write(cleaned + '\n')
                    kept += 1

                if total % 100000 == 0:
                    print(f"  Processed {total} lines, kept {kept} ({100*kept/max(1,total):.1f}%)")

    print(f"Done: {total} lines processed, {kept} kept ({100*kept/max(1,total):.1f}%)")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='文本清洗工具')
    parser.add_argument('--input', type=str, required=True, help='输入文件')
    parser.add_argument('--output', type=str, required=True, help='输出文件')
    parser.add_argument('--min_len', type=int, default=10, help='最小文本长度')
    parser.add_argument('--max_len', type=int, default=2000, help='最大文本长度')
    parser.add_argument('--convert_zh', action='store_true', default=False,
                        help='繁体转简体 (需 pip install zhconv)')
    args = parser.parse_args()

    clean_file(args.input, args.output, args.min_len, args.max_len, args.convert_zh)


if __name__ == '__main__':
    main()
