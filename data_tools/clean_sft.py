"""清理 SFT JSONL 数据的格式污染。

模式:
  --mode prefix   : 移除 A:/B:/C: 等字母前缀（DeepSeek API 格式化污染）
  --mode markdown : 去除 **bold** / *italic* / # heading / 列表标记
  --mode all      : 两种清理都执行（默认）

用法:
  python data_tools/clean_sft.py --input data/sft_data.jsonl --output data/sft_data_clean.jsonl
  python data_tools/clean_sft.py --input data/sft_api_new.jsonl --output clean.jsonl --mode markdown
"""

import argparse
import json
import re
import sys


def clean_prefix(text: str) -> tuple[str, bool]:
    """移除 A:/B:/C: 等字母前缀。返回 (cleaned, changed)。"""
    new = re.sub(r"(?<![a-zA-Z])[A-F]:\s?(?=[\u4e00-\u9fffA-Z])", "", text)
    new = re.sub(r" {2,}", " ", new).strip()
    return new, new != text


def clean_markdown(text: str) -> tuple[str, dict[str, bool]]:
    """去除 markdown 格式。返回 (cleaned, stats dict)。"""
    stats: dict[str, bool] = {}

    # **bold** → bold
    new = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    stats["bold"] = new != text
    text = new

    # *italic* / _underline_
    new = re.sub(r"(?<!\*)\*(.+?)\*(?!\*)", r"\1", text)
    stats["italic"] = new != text
    text = new

    # 行首 # 标题
    new = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    stats["heading"] = new != text
    text = new

    # 行首列表标记
    new = re.sub(r"^(\d+[.)]\s*|[-*]\s+)", "", text, flags=re.MULTILINE)
    stats["list"] = new != text
    text = new

    # 多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text, stats


def main():
    parser = argparse.ArgumentParser(description="清理 SFT JSONL 格式污染")
    parser.add_argument("--input", type=str, required=True, help="输入 JSONL 文件")
    parser.add_argument("--output", type=str, required=True, help="输出 JSONL 文件")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prefix", "markdown", "all"],
        default="all",
        help="清理模式（默认 all）",
    )
    args = parser.parse_args()

    do_prefix = args.mode in ("prefix", "all")
    do_markdown = args.mode in ("markdown", "all")

    total = 0
    cleaned_prefix = 0
    stats_md = {"bold": 0, "italic": 0, "heading": 0, "list": 0}
    invalid = 0

    with (
        open(args.input, encoding="utf-8") as f_in,
        open(args.output, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                invalid += 1
                if invalid <= 3:
                    print(f"  Warning: invalid JSON at line {total}: {e}", file=sys.stderr)
                continue

            if "output" in item:
                output = item["output"]

                if do_prefix:
                    output, changed = clean_prefix(output)
                    if changed:
                        cleaned_prefix += 1

                if do_markdown:
                    output, md_changes = clean_markdown(output)
                    for k, v in md_changes.items():
                        if v:
                            stats_md[k] += 1

                item["output"] = output

            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Total: {total}")
    print(f"Invalid (skipped): {invalid}")
    if do_prefix:
        print(f"Prefix cleaned: {cleaned_prefix}")
    if do_markdown:
        print(
            f"Markdown — bold: {stats_md['bold']}, italic: {stats_md['italic']}, "
            f"heading: {stats_md['heading']}, list: {stats_md['list']}"
        )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
