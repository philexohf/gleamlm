"""GleamLM RAG — 检索增强生成核心组件（零外部依赖）。

retriever.py — BM25 手写检索器，纯 Python stdlib 实现。
"""

from gleamlm.rag.retriever import SimpleRetriever, chunk_text

__all__ = ["SimpleRetriever", "chunk_text"]
