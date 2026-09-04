"""多源数据下载 — datasets / HF Hub / 直链

输出文件名对齐数据 pipeline 约定（gleamlm/data/pipeline.py 的 `{name}_raw.txt`）：
  fineweb → data/raw/edu_raw.txt    （pipeline SOURCES 里 name=edu）
  wiki    → data/raw/wiki_raw.txt
pipeline 的 SOURCES 会消费这些 {name}_raw.txt 作为 step1 输入。
"""

import argparse
import os

from datasets import load_dataset


def download_chinese_fineweb_edu(output_dir: str, split: str = "train"):
    """HuggingFace 数据集，直接 stream 保存为 txt（pipeline 约定名 edu_raw.txt）"""
    ds = load_dataset("opencsg/chinese-fineweb-edu", split=split, streaming=True)
    out_path = os.path.join(output_dir, "edu_raw.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            f.write((row.get("text") or "").strip() + "\n")
            if i > 0 and i % 10000 == 0:
                print(f"  chinese-fineweb-edu: {i} lines")
    print(f"Saved: {out_path} ({i+1} lines)")


def download_wiki_zh(output_dir: str):
    """中文维基百科（pipeline 约定名 wiki_raw.txt）"""
    ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train", streaming=True)
    out_path = os.path.join(output_dir, "wiki_raw.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            f.write((row.get("text") or "").strip() + "\n")
    print(f"Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="data/raw")
    p.add_argument("--sources", type=str, nargs="+", default=["fineweb", "wiki"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for src in args.sources:
        if src == "fineweb":
            download_chinese_fineweb_edu(args.output_dir)
        elif src == "wiki":
            download_wiki_zh(args.output_dir)
