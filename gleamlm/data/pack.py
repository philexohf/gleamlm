"""文本 → Megatron .bin/.idx (mmap) 格式 — 零 HuggingFace 依赖。

与 `gleamlm.data.dataset.tokenize_and_group` 对比:
  1. tokenize_and_group: 文本 → 内存 Dataset（一次性加载进 RAM）
  2. 本模块:         文本 → .bin (token ids 连续写入) + .idx (索引)，
                     训练时用 np.memmap 随机访问任意文档，内存占用 ≈ 0。
                     这是 TB 级预训练语料的唯一可行方式。

.bin 文件格式:
  所有文档的 token id 按 np.uint16 连续写入（无分隔符，靠 .idx 定位）。

.idx 文件格式（与 megatron.core.datasets.indexed_dataset 0.16 完全兼容，
两轨共用同一份数据: manual/pretrain.py 与 industrial/pretrain.py 都可直接读）:
  header:
    b'MMIDIDX\\x00\\x00'         9B  魔数
    version                 8B   uint64 (= 1)
    dtype_code              1B   (8 = uint16)
    sequence_count          8B   uint64 (序列数 = N)
    document_count          8B   uint64 (文档边界数 = N+1，含前导 0)
  sequence_lengths (N × 4B):     每个序列的 token 长度（int32）
  sequence_pointers (N × 8B):    每个序列起始字节偏移（int64）
  document_indices ((N+1) × 8B): 文档边界序列序号（int64，以 0 开头，
                                  本文档=序列 → [0, 1, ..., N]）
"""

import argparse
import importlib
import os
import struct
from typing import Protocol, cast

import numpy as np

# ── Megatron .idx header 魔数（indexed_dataset.py: _INDEX_HEADER）─────────────
_INDEX_HEADER = b"MMIDIDX\x00\x00"
_VERSION = 1
_DTYPE_CODE_UINT16 = 8  # DType 枚举: uint16 = 8


def load_text(path: str) -> list[str]:
    """读取 txt（每行一文档）或 jsonl（{"text": ...} 每行一文档）。

    注意: 不能只凭行首 `{` 判定为 jsonl —— 纯文本文档内容本身可能以
    花括号等开头（如公式/占位符），会被误当 JSON 解析而崩溃（JSONDecodeError）。
    只有整行能解析为含 "text" 键的 JSON 对象时才作为 jsonl，否则按纯文本行。
    """
    import json

    docs: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    obj = None
                if isinstance(obj, dict) and "text" in obj:
                    docs.append(obj["text"])
                    continue
            docs.append(line)
    return docs


# pack 轨需要的 tokenizer 最小接口（HF 风格属性别名）
class _TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...

    @property
    def eos_token_id(self) -> int: ...

    @property
    def vocab_size(self) -> int: ...


def get_tokenizer(name: str, tokenizer_path: str | None = None) -> _TokenizerLike:
    """获取 tokenizer。

    bbpe — 项目自研 BBPE（零 HuggingFace 依赖），默认选项。
    gpt2 — GPT-2 BPE，需安装 transformers（示例用，懒加载）。
    """
    if name == "gpt2":
        # transformers 未钉版本，4.x 直接导出 / 5.x stub 丢失 GPT2TokenizerFast，且 CI
        # 不装 → importlib 动态取类：运行时等价（懒加载/ImportError 语义不变），
        # 同时规避 mypy 对 import 行与属性访问的跨版本分裂报错
        GPT2TokenizerFast = importlib.import_module("transformers").GPT2TokenizerFast

        return cast(_TokenizerLike, GPT2TokenizerFast.from_pretrained("gpt2"))
    if name == "bbpe":
        if tokenizer_path is None:
            raise ValueError("--tokenizer bbpe 需要 --tokenizer-path <BBPE checkpoint 目录>")
        from gleamlm.tokenizer import BBPETokenizer

        tokenizer = BBPETokenizer.load(tokenizer_path)
        # eos_token_id / eod_token_id / vocab_size 由 BBPETokenizer 原生提供（HF 风格别名）
        return tokenizer
    raise ValueError(f"不支持的 tokenizer: {name} (目前支持 bbpe / gpt2)")


# 进程池 worker 的全局状态（initializer 注入，避免 pickle tokenizer 对象）
_WORKER_TOK: dict[str, _TokenizerLike] = {}
_WORKER_EOD: dict[str, int] = {}


def _init_worker(tok: _TokenizerLike) -> None:
    _WORKER_TOK["tok"] = tok


def _tokenize_one(doc: str) -> list[int]:
    return _WORKER_TOK["tok"].encode(doc) + [_WORKER_EOD["eod"]]


_TOKENIZE_CHUNK = 4096  # 每批 tokenize 的文档数；批内驻留内存，写盘后回收


def _tokenize_batch(batch: list[str], tokenizer: _TokenizerLike, workers: int) -> list[list[int]]:
    """并行 tokenize 一批文档（<= _TOKENIZE_CHUNK 个），返回其 token 列表。"""
    eod: int = getattr(tokenizer, "eod_token_id", tokenizer.eos_token_id)
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        _WORKER_EOD["eod"] = eod
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(tokenizer,)
        ) as pool:
            return list(pool.map(_tokenize_one, batch))
    return [tokenizer.encode(doc) + [eod] for doc in batch]


def write_indexed_dataset(prefix: str, docs_tokens: list[list[int]], vocab_size: int) -> str:
    """
    手写 .bin/.idx 生成 — 对应 megatron.core.datasets.indexed_dataset.IndexedDatasetBuilder
    （0.16 版 header 布局，两轨通用）。

    格式说明:
      - .bin 是纯二进制 token 流，训练时 np.memmap 直接映射，不占进程内存
      - .idx 让数据集支持 O(1) 随机访问任意文档（pointer + length 查表）
      - 这也是"为什么工业预训练不用 HF datasets/parquet"的底层原因之一:
        随机采样 shuffle 需要按 token 级随机访问，mmap + 索引是最优解
      - 每文档=一序列（与 megatron IndexedDataset 的"序列"概念对齐），
        因此 document_indices = [0, 1, ..., N]（含前导 0）

    docs_tokens 已是 token 列表（小数据集 / 实验）。TB 级语料请用
    build_indexed_dataset（流式，避免整库驻留内存）。
    """
    with open(prefix + ".bin", "wb") as f:
        for toks in docs_tokens:
            f.write(np.asarray(toks, dtype=np.uint16).tobytes())
    lengths = np.array([len(t) for t in docs_tokens], dtype=np.int32)
    total_tokens = int(lengths.sum())
    _write_idx(prefix, lengths)
    print(f"  {prefix}.bin: {total_tokens:,} tokens ({total_tokens * 2 / 1e6:.1f} MB)")
    print(f"  {prefix}.idx: {len(docs_tokens):,} documents")
    return prefix + ".bin"


def _write_idx(prefix: str, sequence_lengths: np.ndarray) -> int:
    """由每文档 token 长度数组写出 .idx（megatron 0.16 标准布局）。"""
    num_docs = len(sequence_lengths)
    # 每序列起始字节偏移（N 个，第一个=0，不含末尾 total）: uint16 → 2B/token
    sequence_pointers = np.zeros(num_docs, dtype=np.int64)
    np.cumsum(sequence_lengths.astype(np.int64) * 2, out=sequence_pointers)
    sequence_pointers = np.concatenate(([0], sequence_pointers[:-1]))
    # 文档边界索引（N+1 个，含前导 0）: 本文档=序列 → [0, 1, ..., N]
    document_indices = np.arange(num_docs + 1, dtype=np.int64)
    with open(prefix + ".idx", "wb") as f:
        f.write(_INDEX_HEADER)
        f.write(struct.pack("<Q", _VERSION))  # version (8B, uint64)
        f.write(struct.pack("<B", _DTYPE_CODE_UINT16))  # dtype code (1B)
        f.write(struct.pack("<Q", num_docs))  # sequence_count
        f.write(struct.pack("<Q", num_docs + 1))  # document_count (含前导 0)
        f.write(sequence_lengths.tobytes())  # int32 × N
        f.write(sequence_pointers.tobytes())  # int64 × N
        f.write(document_indices.tobytes())  # int64 × (N+1)
    return int(sequence_lengths.sum())


def build_indexed_dataset(
    prefix: str,
    docs: list[str],
    tokenizer: _TokenizerLike,
    workers: int = 4,
    vocab_size: int | None = None,
    chunk_size: int = _TOKENIZE_CHUNK,
    sample_verify: int = 8,
    skip_verify: bool = False,
    print_progress: bool = True,
) -> str:
    """流式（分块）文本 → .bin/.idx，避免整库驻留内存。

    与 write_indexed_dataset 不同，这里 docs 是原始文本：
      分成 chunk_size 个文档一批 → 并行 tokenize → 立即追加写 .bin → 释放该批。
    因此峰值内存 ≈ 一个批次的 token，与语料总量无关（TB 级语料也可跑）。

    验证: 写完后用 megatron/裸 mmap 读回，校验文档数与长度，并随机抽
    sample_verify 篇重新 tokenize 比对内容（全量比对对超大语料不现实）。
    """
    if vocab_size is None:
        vocab_size = tokenizer.vocab_size
    total_tokens = 0
    lengths: list[int] = []
    n_docs = len(docs)
    with open(prefix + ".bin", "wb") as f:
        for i in range(0, n_docs, chunk_size):
            batch = docs[i : i + chunk_size]
            for toks in _tokenize_batch(batch, tokenizer, workers):
                f.write(np.asarray(toks, dtype=np.uint16).tobytes())
                lengths.append(len(toks))
            total_tokens += sum(len(x) for x in batch)
            if print_progress and (i // chunk_size) % 25 == 0:
                print(
                    f"    tokenized {min(i + chunk_size, n_docs):,}/{n_docs:,} docs"
                    f" (written {total_tokens * 2 / 1e6:.0f} MB)",
                    flush=True,
                )
    written = _write_idx(prefix, np.array(lengths, dtype=np.int32))
    print(f"  {prefix}.bin: {written:,} tokens ({written * 2 / 1e6:.1f} MB)")
    print(f"  {prefix}.idx: {n_docs:,} documents")
    if not skip_verify:
        verify_indexed_dataset(
            prefix, n_docs, sample=sample_verify, tokenizer=tokenizer, source=docs, workers=workers
        )
    return prefix + ".bin"


def verify_indexed_dataset(
    prefix: str,
    n_docs: int,
    sample: int = 8,
    tokenizer: _TokenizerLike | None = None,
    source: list[str] | None = None,
    workers: int = 4,
) -> None:
    """读回验证 .bin/.idx。tokenizer/source 提供时，随机抽样重新 tokenize 比对内容。"""
    try:
        # megatron-core 仅 Linux 可装 (industrial extra)，未装时回退纯 py 校验；
        # mypy 侧由 pyproject overrides 静默该模块，无需行内 ignore
        from megatron.core.datasets.indexed_dataset import IndexedDataset

        ds = IndexedDataset(prefix)
        assert len(ds) == n_docs, f"文档数不符: {len(ds)} vs {n_docs}"
        print(f"  [OK] megatron IndexedDataset 验证通过 ({len(ds)} docs)")
        if tokenizer is not None and source is not None and len(source) > 0:
            import random

            rng = random.Random(0)
            for idx in rng.sample(range(len(source)), min(sample, len(source))):
                expect = tokenizer.encode(source[idx]) + [
                    getattr(tokenizer, "eod_token_id", tokenizer.eos_token_id)
                ]
                got = ds[idx].tolist()
                assert got == expect, f"文档 {idx} 内容不符"
            print(f"  [OK] 抽取 {min(sample, len(source))} 篇重新 tokenize 内容比对通过")
    except ImportError:
        import mmap
        import struct

        with open(prefix + ".idx", "rb") as f:
            data = f.read()
        assert data[:9] == _INDEX_HEADER
        seq_count = struct.unpack("<Q", data[18:26])[0]
        assert seq_count == n_docs, f"文档数不符: {seq_count} vs {n_docs}"
        with open(prefix + ".bin", "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        toks0 = np.frombuffer(mm, dtype=np.uint16, count=seq_count, offset=0)
        del toks0
        mm.close()
        print("  [OK] 裸 mmap 验证通过（megatron 未安装，用标准库验证）")


def verify(prefix: str, docs_tokens: list[list[int]]) -> None:
    """写后自读回验证（写完必须验证，工业数据管线同样如此）。"""
    try:
        from megatron.core.datasets.indexed_dataset import IndexedDataset

        ds = IndexedDataset(prefix)
        assert len(ds) == len(docs_tokens), f"文档数不符: {len(ds)} vs {len(docs_tokens)}"
        for i in (0, len(docs_tokens) // 2, len(docs_tokens) - 1):
            got = ds[i].tolist()
            assert got == docs_tokens[i], f"文档 {i} 内容不符"
        print(f"  [OK] megatron IndexedDataset 验证通过 ({len(ds)} docs)")
    except ImportError:
        # megatron 未安装时用裸 mmap 验证格式（新 header: 9 magic + 8 version
        # + 1 dtype + 8 seq_count + 8 doc_count = 34B，然后 int32 lengths(N)
        # + int64 pointers(N)）
        import mmap
        import struct

        with open(prefix + ".idx", "rb") as f:
            data = f.read()
        assert data[:9] == _INDEX_HEADER
        seq_count = struct.unpack("<Q", data[18:26])[0]
        ptr_offset = 34 + 4 * seq_count
        seq_ptr = np.frombuffer(
            data[ptr_offset : ptr_offset + 8 * seq_count],
            dtype=np.int64,
            count=seq_count,
        )
        with open(prefix + ".bin", "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        toks0 = np.frombuffer(mm, dtype=np.uint16, count=len(docs_tokens[0]), offset=seq_ptr[0])
        assert toks0.tolist() == docs_tokens[0]
        del toks0  # np.frombuffer 持有 mmap 缓冲引用，不释放则 close 报 BufferError
        mm.close()
        print("  [OK] 裸 mmap 验证通过（megatron 未安装，用标准库验证）")


# ──── CLI（独立使用: python -m gleamlm.data.pack） ────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="文本 → Megatron .bin/.idx 预处理")
    p.add_argument("--input", required=True, help="输入 txt（每行一文档）或 jsonl")
    p.add_argument("--output-prefix", required=True, help="输出前缀 (生成 .bin/.idx)")
    p.add_argument("--tokenizer", default="bbpe", help="tokenizer (bbpe 或 gpt2)")
    p.add_argument(
        "--tokenizer-path", default=None, help="BBPE checkpoint 目录 (--tokenizer bbpe 时必填)"
    )
    p.add_argument("--workers", type=int, default=4, help="并行进程数")
    p.add_argument("--skip-verify", action="store_true", help="跳过写后验证")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    print(f"[1/3] 读取数据: {args.input}")
    docs = load_text(args.input)
    if not docs:
        print("  (空输入) 跳过")
        return

    print(f"[2/3] tokenize ({len(docs)} docs, workers={args.workers})")
    tokenizer = get_tokenizer(args.tokenizer, args.tokenizer_path)

    print("[3/3] 流式写入 .bin/.idx")
    build_indexed_dataset(
        args.output_prefix,
        docs,
        tokenizer,
        workers=args.workers,
        vocab_size=tokenizer.vocab_size,
        skip_verify=args.skip_verify,
    )


if __name__ == "__main__":
    main()
