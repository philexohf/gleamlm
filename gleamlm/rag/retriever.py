"""
SimpleRetriever — BM25 + 分块检索。RAG 链路: Chunking → Retrieval → Generation，
检索结果拼入 prompt 上下文。BM25 零成本、可解释，适合冷启动。

用法:"""

# BM25: 词袋 + TF-IDF 统计，零训练、可解释；Dense: 语义相似度，需训练；
# 生产系统两者融合 (RRF 排序融合)，BM25 擅长精确匹配，向量擅长语义近似。

import math
import re
from collections import Counter
from typing import Optional


# 按字符切分 (CJK 下 token ≈ 字符)，overlap 弥补横跨切口的信息丢失；
# 工业级按 ["\n\n", "\n", "。", " "] 优先级递归切。

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    tokens = list(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append("".join(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


class SimpleRetriever:
    """基于 BM25 的检索器，零外部依赖。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[str] = []
        self._idf: dict[str, float] = {}
        self._doc_freq: dict[str, int] = {}
        self._doc_lens: list[int] = []
        self._avg_len: float = 0.0
        self._tokenized: list[Counter] = []

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def add_documents(self, docs: list[str], chunk: bool = True, chunk_size: int = 512, chunk_overlap: int = 50) -> list[int]:
        if chunk:
            all_chunks = []
            for doc in docs:
                all_chunks.extend(chunk_text(doc, chunk_size, chunk_overlap))
        else:
            all_chunks = docs
        start_idx = len(self.documents)
        self.documents.extend(all_chunks)
        idx_range = list(range(start_idx, len(self.documents)))
        self._build_index()
        return idx_range

    # BM25 改进自 TF-IDF: TF 饱和 (k1)、长度归一化 (b)、IDF 平滑；
    # score(q,d) = Σ_t IDF(t)·tf(k1+1) / (tf + k1(1-b+b·|d|/avgdl))

    def _build_index(self):
        self._tokenized = [Counter(self._tokenize(d)) for d in self.documents]
        self._doc_lens = [sum(t.values()) for t in self._tokenized]
        self._avg_len = sum(self._doc_lens) / max(len(self._doc_lens), 1)

        all_terms: set[str] = set()
        for t in self._tokenized:
            all_terms.update(t.keys())

        N = len(self.documents)
        self._doc_freq = {}
        for term in all_terms:
            self._doc_freq[term] = sum(1 for t in self._tokenized if term in t)
        # IDF 平滑版: log(1 + (N - df + 0.5) / (df + 0.5) + 1)
        self._idf = {term: math.log(1 + (N - freq + 0.5) / (freq + 0.5) + 1) for term, freq in self._doc_freq.items()}
        for term in self._idf:
            if self._idf[term] < 0:
                self._idf[term] = 0

    # 对每个文档累加 query 词 BM25 分 (query 词去重防重复加分)；
    # 复杂度 O(N×|query|)，生产用倒排索引只对含 query 词的文档打分。

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        query_terms = self._tokenize(query)
        scores = []
        for i, tokenized in enumerate(self._tokenized):
            score = 0.0
            dl = self._doc_lens[i]
            for term in set(query_terms):
                if term not in self._idf:
                    continue
                tf = tokenized.get(term, 0)
                idf = self._idf[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_len)
                score += idf * numerator / denominator
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[i], s) for i, s in scores[:top_k]]
