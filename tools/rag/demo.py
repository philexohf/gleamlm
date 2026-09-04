"""
RAG Demo — 检索增强生成演示脚本。

用法:
  # 先准备知识库文档 (每行一个文件路径)
  python -m tools.rag.demo \
    --model checkpoints/pro/final.pt \
    --docs docs.txt \
    --query "GleamLM 的特点是什么？"

  # 交互模式
  python -m tools.rag.demo \
    --model checkpoints/pro/final.pt \
    --docs docs.txt \
    --interactive
"""

import argparse
import os

import torch

from hf.hf_config import gleamlm_config_from_core
from hf.hf_model import GleamLMForCausalLM, load_from_checkpoint
from gleamlm.rag.retriever import SimpleRetriever
from tools.rag.pipeline import RAGPipeline
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, extract_checkpoint_config


def load_model(model_path: str, tokenizer_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BBPETokenizer.load(tokenizer_path or DEFAULT_TOKENIZER_PATH)

    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    cfg = extract_checkpoint_config(ckpt)
    hf_config = gleamlm_config_from_core(cfg)
    model = GleamLMForCausalLM(hf_config)
    missing, unexpected = load_from_checkpoint(model, ckpt, strict=True)
    if missing or unexpected:
        print(f"[warn] rag load — missing={missing} unexpected={unexpected}")
    model.eval()
    return model, tokenizer, device


def load_docs(paths: list[str]) -> list[str]:
    docs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            docs.append(f.read())
    return docs


def main():
    args = parse_args()

    if not os.path.exists(args.docs):
        print(f"Docs file not found: {args.docs}")
        return

    with open(args.docs, encoding="utf-8") as f:
        doc_files = [line.strip() for line in f if line.strip()]

    docs = load_docs(doc_files)
    print(f"Loaded {len(docs)} documents ({sum(len(d) for d in docs)} chars)")

    model, tokenizer, device = load_model(args.model, args.tokenizer_path)
    pipeline = RAGPipeline(model, tokenizer, device)
    pipeline.add_knowledge(docs, chunk_size=args.chunk_size)

    if args.interactive:
        print("\nRAG Interactive (type 'quit' to exit)\n")
        while True:
            q = input("Q: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            result = pipeline.query(q, top_k=args.top_k, max_new_tokens=args.max_new_tokens)
            print(f"A: {result['answer']}\n")
            print("--- Sources ---")
            for chunk, score in result["chunks"]:
                print(f"  [{score:.3f}] {chunk[:120]}...")
            print()
    else:
        result = pipeline.query(args.query, top_k=args.top_k, max_new_tokens=args.max_new_tokens)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")


def parse_args():
    p = argparse.ArgumentParser(description="GleamLM RAG Demo")
    p.add_argument("--model", required=True)
    p.add_argument("--docs", required=True, help="Text file listing document paths")
    p.add_argument("--query", default="", help="Question (omit for interactive)")
    p.add_argument("--tokenizer_path", default="")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--interactive", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
