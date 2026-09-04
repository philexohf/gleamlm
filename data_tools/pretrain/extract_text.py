"""数据格式转换：从各源原始格式提取纯文本。

支持 5 种格式:
  news  — zip/gz 归档中的 JSONL，拼接 title+desc+content
  baike — JSON 数组或 JSONL，拼接 title+summary+text
  qa    — zip/gz 归档中的 JSONL，转为 "问题：... 回答：..."
  wiki  — JSONL，提取 text 字段
  edu   — Parquet 目录，提取 text 列

用法:
    python data_tools/pretrain/extract_text.py --format news \\
        --input data/external/new2016zh.zip --output data/raw/news_raw.txt

    python data_tools/pretrain/extract_text.py --format wiki \\
        --input data/external/wikipedia-zh/wikipedia-zh-cn-20260501.json \\
        --output data/raw/wiki_raw.txt
"""

import argparse
import glob
import gzip
import io
import json
import os
import sys
import zipfile


def extract_news(input_path, output_path):
    """从 news2016zh zip/gz 提取纯文本（title + desc + content）"""
    archive = _open_archive(input_path)
    if archive is None:
        print(f"  无法打开: {input_path}")
        return

    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for line in archive:
            try:
                item = json.loads(line.strip())
                title = item.get("title", "") or ""
                desc = item.get("desc", "") or ""
                content = item.get("content", "") or ""
                parts = [p.strip() for p in [title, desc, content] if p.strip()]
                full_text = "。".join(parts) if parts else ""
                if len(full_text) > 30:
                    fout.write(full_text + "\n")
                    count += 1
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

    print(f"  news: {count:,} 篇 → {output_path}")


def extract_baike(input_path, output_path):
    """从百度百科 JSON/JSONL 提取纯文本（title + summary + text）"""
    # 自动解压 7z
    if input_path.endswith(".7z"):
        import py7zr

        json_path = input_path[:-3]
        if not os.path.exists(json_path):
            print(f"  解压 7z: {input_path} → {json_path}")
            with py7zr.SevenZipFile(input_path, "r") as archive:
                archive.extractall(os.path.dirname(json_path) or ".")
        input_path = json_path

    items = _load_json_or_jsonl(input_path)
    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in items:
            title = item.get("title", "") or ""
            summary = item.get("summary", "") or ""
            text = item.get("text", "") or ""
            parts = [p.strip() for p in [title, summary, text] if p.strip()]
            full_text = "。".join(parts) if parts else ""
            full_text = full_text.replace("。。", "。")
            if len(full_text) > 20:
                fout.write(full_text + "\n")
                count += 1

    print(f"  baike: {count:,} 条 → {output_path}")


def extract_qa(input_path, output_path):
    """从 webtext2019zh zip/gz 提取 QA 格式纯文本"""
    archive = _open_archive(input_path)
    if archive is None:
        print(f"  无法打开: {input_path}")
        return

    items = _parse_json_or_jsonl(archive)
    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in items:
            title = item.get("title", "") or ""
            content = item.get("content", "") or ""
            star = item.get("star", 1)
            if star == 0 and len(content) < 20:
                continue
            if len(title) > 3 and len(content) > 10:
                full_text = f"问题：{title} 回答：{content}"
                fout.write(full_text + "\n")
                count += 1

    print(f"  qa: {count:,} 条 → {output_path}")


def extract_wiki(input_path, output_path):
    """从维基百科 JSONL 提取纯文本（text 字段）"""
    count = 0
    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            try:
                item = json.loads(line.strip())
                text = item.get("text", "")
                if text and len(text.rstrip()) > 30:
                    fout.write(text.rstrip() + "\n")
                    count += 1
            except json.JSONDecodeError:
                continue

    print(f"  wiki: {count:,} 篇 → {output_path}")


def extract_edu(input_dir, output_path):
    """从 Parquet 目录提取 text 列 → 纯文本"""
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        print(f"  未找到 parquet 文件: {input_dir}")
        return

    total_size = sum(os.path.getsize(f) for f in files)
    print(f"  edu: {len(files)} 个 parquet, {total_size / 1e9:.2f} GB")

    try:
        import pyarrow.parquet as pq

        backend = "pyarrow"
    except ImportError:
        print("  pyarrow 未安装, 尝试 fastparquet/pandas...")
        try:
            import pandas as pd  # noqa: F401

            backend = "fastparquet"
        except ImportError:
            print("  错误: pip install pyarrow")
            sys.exit(1)

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for i, fpath in enumerate(files):
            if backend == "pyarrow":
                pf = pq.ParquetFile(fpath)
                for batch in pf.iter_batches(columns=["text"]):
                    col = batch.column("text")
                    for j in range(len(col)):
                        text = col[j].as_py()
                        if text and isinstance(text, str):
                            out.write(text.rstrip() + "\n")
                            count += 1
            else:
                df = pd.read_parquet(fpath, columns=["text"])  # type: ignore
                for text in df["text"].dropna():
                    out.write(str(text).rstrip() + "\n")
                    count += 1
            if (i + 1) % 5 == 0:
                print(f"    [{i + 1}/{len(files)}] {count:,} 行")

    print(f"  edu: {count:,} 行 → {output_path}")


# ──── Helpers ──────────────────────────────────────────────────────


def _open_archive(path):
    """打开 zip/gz/jsonl 归档, 返回逐行文本迭代器"""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    if path.endswith(".zip"):
        zf = zipfile.ZipFile(path, "r")
        json_files = [n for n in zf.namelist() if n.endswith(".json")]
        if not json_files:
            zf.close()
            return None
        train = [n for n in json_files if "train" in n.lower()]
        target = train[0] if train else max(json_files, key=lambda n: zf.getinfo(n).file_size)
        return io.TextIOWrapper(zf.open(target), encoding="utf-8")
    if path.endswith((".json", ".jsonl")):
        return open(path, "r", encoding="utf-8")
    return None


def _load_json_or_jsonl(path):
    """加载 JSON 数组或 JSONL 文件 → list[dict]"""
    with open(path, encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return items


def _parse_json_or_jsonl(iterator):
    """从迭代器解析 JSON 数组或 JSONL → list[dict]"""
    raw = iterator.read(1024) if hasattr(iterator, "read") else ""
    is_jsonl = b"qid" in (raw if isinstance(raw, bytes) else raw.encode()) if raw else False
    items = []
    if not is_jsonl:
        try:
            iterator.seek(0) if hasattr(iterator, "seek") else None
            data = json.load(iterator)
            if isinstance(data, dict) and "root" in data:
                items = data["root"]
            elif isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list) and len(v) > len(items):
                        items = v
            return items
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    iterator.seek(0) if hasattr(iterator, "seek") else None
    for line in iterator:
        if hasattr(line, "decode"):
            line = line.decode("utf-8")
        try:
            items.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            continue
    return items


# ──── Main ─────────────────────────────────────────────────────────

EXTRACTORS = {
    "news": extract_news,
    "baike": extract_baike,
    "qa": extract_qa,
    "wiki": extract_wiki,
    "edu": lambda p, o: extract_edu(p, o),
}


def main():
    parser = argparse.ArgumentParser(description="数据格式转换 → 纯文本")
    parser.add_argument(
        "--format", choices=["news", "baike", "qa", "wiki", "edu"], required=True, help="数据源格式"
    )
    parser.add_argument("--input", required=True, help="输入文件/目录")
    parser.add_argument("--output", required=True, help="输出纯文本文件")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    EXTRACTORS[args.format](args.input, args.output)


if __name__ == "__main__":
    main()
