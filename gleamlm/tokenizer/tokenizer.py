"""
BBPE（Byte-Level BPE）分词器 — 纯 Python 自实现，零外部依赖。

纯 Python 实现，零外部依赖；对 CJK 做单字预分词。
"""

from __future__ import annotations

import heapq
import json
import os
import re
import time
from collections import defaultdict

# 预分词: 直接在字节上做 BPE 没有天然分隔符，合并不可控且词表膨胀。
# 本方案 CJK 单字独立 + 非 CJK 连续片段——中文每字 ≈ 一个 token 的信息密度，
# 比 GPT-2 的空白分隔 regex 对中文友好。


# CJK 预分词正则：匹配单个 CJK 汉字/符号，或连续非 CJK 片段

_PRE_TOKENIZE_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]|"
    r"[\u3000-\u303f\uff00-\uffef]|"
    r"[^\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f\uff00-\uffef]+"
)


def _pre_tokenize_re(text: str) -> list[str]:
    return [m.group(0) for m in _PRE_TOKENIZE_RE.finditer(text)]


# 特殊 token 布局: ID 0-2 ChatML 协议 (pad/bos/eos)，ID 3-12 buffer 位，
# ID 13-268 UTF-8 字节，269+ BPE merge。低 ID 便于 decode 快速跳过 (id < 13)。

_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|buffer1|>",
    "<|buffer2|>",
    "<|buffer3|>",
    "<|buffer4|>",
    "<|buffer5|>",
    "<|buffer6|>",
    "<|buffer7|>",
    "<|buffer8|>",
    "<|buffer9|>",
    "<|buffer10|>",
]
_NUM_SPECIAL_TOKENS = len(_SPECIAL_TOKENS)


class BBPETokenizer:
    """Byte-Level BPE tokenizer — CJK pre-tokenization + BPE merges."""

    def __init__(self) -> None:
        self.id_to_byte: dict[int, bytes] = {}

        self.merges: dict[tuple[int, int], int] = {}
        self.merge_pairs: dict[int, tuple[int, int]] = {}

        self.special_tokens: dict[str, int] = {}
        self.id_to_special: dict[int, str] = {}

        self._next_id = 0

        self._pre_tokenize_fn = _pre_tokenize_re
        self._special_regex: re.Pattern[str] | None = None

        self._add_special_tokens()

        for i in range(256):
            self.id_to_byte[self._next_id] = bytes([i])
            self._next_id += 1

    @property
    def _byte_offset(self) -> int:
        return len(self.special_tokens)

    def _offset_raw_bytes(self, raw: list[int]) -> list[int]:
        return [b + self._byte_offset for b in raw]

    @classmethod
    def train_from_files(
        cls,
        text_files: list[str],
        vocab_size: int = 12002,
        save_dir: str | None = None,
        max_train_chars: int = 500_000_000,
        ratios: list[float] | None = None,
    ) -> BBPETokenizer:
        """从多个文本文件训练 BBPE：预分词 → 字节化 → 倒排索引 → 最大堆 → 迭代合并。

        用堆而非全量扫描找最高频 pair：扫描是 O(V × L)，堆只需 O(log V) 弹顶
        + O(受影响位置数) 更新，可扩展到百 GB 级数据。
        """
        tokenizer = cls()
        if ratios is None:
            ratios = [1.0 / len(text_files)] * len(text_files)

        print(f"Training BBPE tokenizer (vocab_size={vocab_size})...")
        print(f"  Input files: {len(text_files)}, max_chars={max_train_chars / 1e6:.0f}M")
        for fp, r in zip(text_files, ratios, strict=False):
            print(f"    [{r * 100:.0f}%] {os.path.basename(fp)}")

        print("  Step 1/3: Pre-tokenizing and counting word frequencies...")
        byte_sequences, word_counts = tokenizer._pre_tokenize_files(
            text_files, max_chars=max_train_chars, ratios=ratios
        )
        total_pairs = sum(len(seq) - 1 for seq in byte_sequences if len(seq) > 1)
        print(f"  Collected {len(byte_sequences):,} unique words, {total_pairs:,} initial pairs")

        n_merges = vocab_size - tokenizer._next_id
        print(f"  Step 2/3: Training {n_merges} BPE merges...")

        print("    Building pair index...", end=" ", flush=True)
        t_idx = time.time()
        # 倒排索引: pair → {(word_idx, pos)}；word 已唯一化，
        # 实际出现频次 = 位置集合 × word_counts[word_idx]（加权）
        pair_to_positions: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
        for wid, seq in enumerate(byte_sequences):
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_to_positions[pair].add((wid, i))  # 记录pair出现的位置
        print(f"{len(pair_to_positions):,} unique pairs ({time.time() - t_idx:.1f}s)")

        print("    Building max-heap...", end=" ", flush=True)
        t_heap = time.time()

        def pair_freq(positions: set[tuple[int, int]]) -> int:
            # 加权频次: 每个唯一 word 出现 word_counts[wid] 次
            return sum(word_counts[wid] for wid, _ in positions)

        # heapq 是最小堆，用负频次取堆顶 = 频次最高的 pair
        heap: list[tuple[int, tuple[int, int]]] = []
        for pair, positions in pair_to_positions.items():
            heapq.heappush(heap, (-pair_freq(positions), pair))  # 负号实现最大堆
        print(f"done ({time.time() - t_heap:.1f}s)")

        t_start = time.time()
        pbar_interval = max(1, n_merges // 200)
        print(f"    Merging (pbar every {pbar_interval} steps)...", flush=True)

        for merge_step in range(n_merges):
            if not pair_to_positions:
                print(f"\n  No more pairs to merge at step {merge_step}")
                break

            # 惰性校验: pop 后检查 pair 是否仍在索引中、count 是否匹配，否则重入堆
            while heap:
                neg_count, best_pair = heapq.heappop(heap)  # 弹出堆顶
                if best_pair not in pair_to_positions:  # pair已被合并掉
                    continue
                actual_count = pair_freq(pair_to_positions[best_pair])  # 实际频次
                if actual_count != -neg_count:  # 频次不匹配，脏数据
                    heapq.heappush(heap, (-actual_count, best_pair))  # 用正确频次重入
                    continue
                break  # 有效pair，退出循环
            else:
                print(f"\n  Heap empty at step {merge_step}")
                break

            best_count = pair_freq(pair_to_positions[best_pair])

            if best_count < 2:
                print(f"\n  All pairs have count=1 at step {merge_step}, stopping")
                break

            # merges 供编码查表；merge_pairs/id_to_byte 让解码直接拼接字节，无需还原 merge 树
            new_id = tokenizer._next_id
            tokenizer.merges[best_pair] = new_id
            tokenizer.merge_pairs[new_id] = best_pair
            tokenizer.id_to_byte[new_id] = (
                tokenizer.id_to_byte[best_pair[0]] + tokenizer.id_to_byte[best_pair[1]]
            )
            tokenizer._next_id += 1

            # 受影响位置按 word 分组；从后往前合并避免位置偏移
            affected = pair_to_positions.pop(best_pair)  # 取出所有出现位置，同时从索引删除
            by_word: dict[int, list[int]] = defaultdict(list)
            for wid, pos in affected:
                by_word[wid].append(pos)  # 按word分组

            for wid, pos_list in by_word.items():
                seq = byte_sequences[wid]
                pos_list.sort(reverse=True)  # 从后往前合并，避免位置偏移
                for pos in pos_list:
                    if pos >= len(seq) - 1:  # 越界检查
                        continue
                    if (seq[pos], seq[pos + 1]) != best_pair:  # 已被前面合并改变
                        continue

                    # 删除旧邻居 pair

                    if pos > 0:
                        old_left = (seq[pos - 1], seq[pos])  # 左边的pair
                        if old_left in pair_to_positions:
                            pair_to_positions[old_left].discard((wid, pos - 1))
                            if not pair_to_positions[old_left]:
                                del pair_to_positions[old_left]

                    if pos + 2 < len(seq):
                        old_right = (seq[pos + 1], seq[pos + 2])  # 右边的pair
                        if old_right in pair_to_positions:
                            pair_to_positions[old_right].discard((wid, pos + 1))
                            if not pair_to_positions[old_right]:
                                del pair_to_positions[old_right]

                    # 执行合并: seq[pos] = new_id, 删除 seq[pos+1]

                    seq[pos] = new_id  # 合并为新token
                    del seq[pos + 1]  # 删除被合并的token

                    # 注册新邻居 pair

                    if pos > 0:
                        new_pair = (seq[pos - 1], new_id)  # 左边新pair
                        pair_to_positions[new_pair].add((wid, pos - 1))
                        heapq.heappush(heap, (-pair_freq(pair_to_positions[new_pair]), new_pair))
                    if pos < len(seq) - 1:
                        new_pair = (new_id, seq[pos + 1])  # 右边新pair
                        pair_to_positions[new_pair].add((wid, pos))
                        heapq.heappush(heap, (-pair_freq(pair_to_positions[new_pair]), new_pair))

            step = merge_step + 1
            if step % pbar_interval == 0 or step == n_merges:
                pct = step / n_merges * 100
                bar_width = 30
                filled = int(bar_width * step / n_merges)
                bar = "#" * filled + "-" * (bar_width - filled)
                elapsed = time.time() - t_start
                eta = elapsed / step * (n_merges - step)
                eta_str = (
                    f"{eta / 60:.0f}m{eta % 60:02.0f}s" if eta < 3600 else f"{eta / 3600:.1f}h"
                )
                print(
                    f"\r  [{bar}] {pct:5.1f}% ({step}/{n_merges}) | "
                    f"pair=({best_pair[0]},{best_pair[1]}) cnt={best_count} | "
                    f"ETA {eta_str}",
                    end="",
                    flush=True,
                )

            if step % 1000 == 0:
                print()

        print(f"\n  Trained {len(tokenizer.merges)} merges, vocab_size={tokenizer._next_id}")

        if save_dir:
            tokenizer.save(save_dir)

        return tokenizer

    def _pre_tokenize_files(
        self, text_files: list[str], max_chars: int = 500_000_000, ratios: list[float] | None = None
    ) -> tuple[list[list[int]], list[int]]:
        """聚合词频统计：返回唯一 word 的字节序列 + 出现次数。

        相比旧的"全量 word 序列列表"方案，这里对重复 word 只保留一份
        (bytes → count)，内存从 O(总 words) 降到 O(唯一 words)。
        """
        from collections import Counter

        word_counts: Counter = Counter()
        if ratios is None:
            ratios = [1.0 / len(text_files)] * len(text_files)

        quotas = [int(max_chars * r) for r in ratios]
        total_words = 0
        chunk_size = 5_000_000

        for i, fpath in enumerate(text_files):
            if not os.path.exists(fpath):
                print(f"    Skip: {fpath} (not found)")
                continue
            if quotas[i] <= 0:
                continue

            quota_mb = quotas[i] / 1e6
            print(
                f"    [{ratios[i] * 100:.0f}%] {os.path.basename(fpath)}: "
                f"quota={quota_mb:.1f}M chars",
                flush=True,
            )

            file_words = 0
            with open(fpath, encoding="utf-8") as f:
                text_remaining = quotas[i]
                while text_remaining > 0:
                    chunk = f.read(min(chunk_size, text_remaining))
                    if not chunk:
                        break
                    text_remaining -= len(chunk)

                    words = self._pre_tokenize(chunk)
                    for word in words:
                        byte_seq = self._offset_raw_bytes(list(word.encode("utf-8")))
                        if byte_seq:
                            word_counts[tuple(byte_seq)] += 1
                            file_words += 1

                    pct = 100 * (quotas[i] - text_remaining) / quotas[i]
                    print(f"\r      {pct:.0f}% ({file_words:,} words)", end="", flush=True)

            total_words += file_words
            print(f" → {file_words:,} words")

        # 展开为 (唯一字节序列, 频次) 列表，供 train_from_files 使用
        sequences = [list(seq) for seq in word_counts]
        counts = [word_counts[tuple(seq)] for seq in sequences]
        print(
            f"    Total: {total_words:,} words → {len(sequences):,} unique "
            f"({sum(quotas) / 1e6:.1f}M chars from {len(text_files)} files)"
        )
        return sequences, counts

    def _pre_tokenize(self, text: str) -> list[str]:
        return self._pre_tokenize_fn(text)

    def _add_special_tokens(self) -> None:
        if self.special_tokens:
            return

        for token in _SPECIAL_TOKENS:
            if token not in self.special_tokens:
                tid = self._next_id
                self.special_tokens[token] = tid
                self.id_to_special[tid] = token
                self.id_to_byte[tid] = token.encode("utf-8")
                self._next_id += 1

        self._set_aliases()
        self._build_special_regex()

    def _set_aliases(self) -> None:
        self.pad_token = "<|endoftext|>"
        self.unk_token = "<|endoftext|>"
        self.bos_token = "<|im_start|>"
        self.eos_token = "<|im_end|>"

        self.pad_id = self.special_tokens.get("<|endoftext|>", 0)
        self.unk_id = self.special_tokens.get("<|endoftext|>", 0)
        self.bos_id = self.special_tokens.get("<|im_start|>", 1)
        self.eos_id = self.special_tokens.get("<|im_end|>", 2)

        self.im_start_id = self.special_tokens.get("<|im_start|>", 1)
        self.im_end_id = self.special_tokens.get("<|im_end|>", 2)

    def _build_special_regex(self) -> None:
        if not self.special_tokens:
            self._special_regex = None
            return
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self._special_regex = re.compile("(" + "|".join(escaped) + ")")

    # 编码: 先用特殊 token regex 把它们当硬边界保留为独立 ID，普通文本再做 BPE
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:

        ids = []
        if add_bos:
            ids.append(self.bos_id)

        parts = self._special_regex.split(text) if self._special_regex is not None else [text]

        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                words = self._pre_tokenize(part)
                for word in words:
                    byte_seq = list(word.encode("utf-8"))
                    ids.extend(self._apply_bpe_to_bytes(self._offset_raw_bytes(byte_seq)))

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def _apply_bpe_to_bytes(self, byte_seq: list[int]) -> list[int]:
        """BPE 贪心编码：每次找 rank 最小（训练时最早合并）的 pair 并合并。"""
        if not byte_seq:
            return []

        seq = list(byte_seq)
        while len(seq) > 1:
            best_rank = float("inf")
            best_pos = -1
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                if pair in self.merges:  # rank 越小 = 训练时越早合并 = 优先级越高
                    rank = self.merges[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pos = i

            if best_pos == -1:
                break

            pair = (seq[best_pos], seq[best_pos + 1])
            merged = self.merges[pair]
            seq[best_pos] = merged
            del seq[best_pos + 1]

        return seq

    # 解码: ID → 字节拼接 → UTF-8。训练时已记录每个 merge token 的完整字节序列，
    # 无需还原 merge 树。
    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        byte_buffer = bytearray()

        for tid in ids:
            if skip_special and tid in self.id_to_special:
                continue
            if tid in self.id_to_byte:
                byte_buffer.extend(self.id_to_byte[tid])
            else:
                byte_buffer.extend(b"?")

        return byte_buffer.decode("utf-8", errors="replace")

    def encode_batch(
        self, texts: list[str], add_bos: bool = False, add_eos: bool = False
    ) -> list[list[int]]:
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def token_to_id(self, token: str) -> int:
        return self.special_tokens.get(token, self.unk_id)

    def get_vocab_size(self) -> int:
        return self._next_id

    def __len__(self) -> int:
        return self.get_vocab_size()

    @property
    def eos_token_id(self) -> int:
        """HF 风格别名: ChatML 终止符 <|im_end|>（SFT 轨 eos）。"""
        return self.eos_id

    @property
    def eod_token_id(self) -> int:
        """预训练文档边界符 <|endoftext|>（= pad id，与 eos 区分）。"""
        return self.pad_id

    @property
    def vocab_size(self) -> int:
        """HF 风格别名，与 get_vocab_size() 等价。"""
        return self.get_vocab_size()

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)

        data = {
            "merges": {f"{a} {b}": mid for (a, b), mid in self.merges.items()},
            "merge_pairs": {str(mid): list(pair) for mid, pair in self.merge_pairs.items()},
            "special_tokens": self.special_tokens,
            "id_to_special": {str(k): v for k, v in self.id_to_special.items()},
            "_next_id": self._next_id,
        }

        path = os.path.join(save_dir, "bbpe_tokenizer.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"BBPE tokenizer saved: {path} (vocab_size={self._next_id})")

    @classmethod
    def load(cls, save_dir: str) -> BBPETokenizer:
        path = os.path.join(save_dir, "bbpe_tokenizer.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer not found: {path}")

        # utf-8-sig: 兼容 Windows 写入的 UTF-8 BOM 头（json.load 会拒绝 BOM）
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)

        tokenizer = cls()

        for pair_str, mid in data["merges"].items():
            a, b = pair_str.split()
            pair = (int(a), int(b))
            tokenizer.merges[pair] = int(mid)

        for mid_str, pair_list in sorted(data["merge_pairs"].items(), key=lambda x: int(x[0])):
            mid = int(mid_str)
            tokenizer.merge_pairs[mid] = tuple(pair_list)
            a, b = pair_list
            tokenizer.id_to_byte[mid] = tokenizer.id_to_byte[a] + tokenizer.id_to_byte[b]

        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.id_to_special = {int(k): v for k, v in data["id_to_special"].items()}
        for tid, token in tokenizer.id_to_special.items():
            tokenizer.id_to_byte[tid] = token.encode("utf-8")
        tokenizer._next_id = data["_next_id"]

        tokenizer._set_aliases()
        tokenizer._build_special_regex()

        print(f"BBPE tokenizer loaded: {path} (vocab_size={tokenizer._next_id})")
        return tokenizer

    # HF 导出: vLLM/transformers 通过 tokenizer.json 加载。
    # vocab 字节用 _bytes_to_unicode() 符号 (HF byte-fallback 内部按字节符号合并)，
    # <0xXX> 只用于兜底输出；pre-tokenizer 复刻 _PRE_TOKENIZE_RE。

    def to_hf_tokenizer_json(self) -> dict:
        """导出 HF tokenizer.json 格式的 dict，供 vLLM / AutoTokenizer 加载。"""
        byte_symbols = _bytes_to_unicode()

        # merge token 用子 token 字符串拼接: 贪心 BPE 靠字符串匹配，
        # 合并后的 token 才能匹配原始字节序列；merge 只引用更小 id。

        vocab: dict[str, int] = {}
        token_strings: dict[int, str] = {}
        for tid in range(self._next_id):
            if tid in self.id_to_special:
                s = self.id_to_special[tid]
            elif tid < self._byte_offset + 256:
                s = byte_symbols[tid - self._byte_offset]
            else:
                a, b = self.merge_pairs[tid]
                s = token_strings[a] + token_strings[b]
            token_strings[tid] = s
            vocab[s] = tid

        # merges 按训练顺序输出: id 升序 = 合并先后 = rank

        merges = [
            f"{token_strings[a]} {token_strings[b]}"
            for (a, b), _ in sorted(self.merges.items(), key=lambda kv: kv[1])
        ]

        # added_tokens: special=True，编码时自动隔离

        added_tokens = [
            {
                "id": tid,
                "content": tok,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for tok, tid in sorted(self.special_tokens.items(), key=lambda kv: kv[1])
        ]

        return {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": added_tokens,
            "normalizer": None,
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {"Regex": _PRE_TOKENIZE_RE.pattern},
                        "behavior": "Isolated",
                        "invert": False,
                    },
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": False,
                        "trim_offsets": True,
                        "use_regex": False,
                    },
                ],
            },
            "post_processor": None,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            },
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": None,
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "byte_fallback": True,
                "vocab": vocab,
                "merges": merges,
            },
        }

    def export_to_hf_format(self, save_dir: str) -> str:
        """写出 HF tokenizer.json + tokenizer_config.json。"""
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_hf_tokenizer_json(), f, ensure_ascii=False, indent=2)

        additional_special = [
            t for t in _SPECIAL_TOKENS if t not in (self.bos_token, self.eos_token, self.pad_token)
        ]
        config = {
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": None,
            "additional_special_tokens": additional_special,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": 4096,
            "chat_template": _CHATML_TEMPLATE,
        }
        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(
            f"HF tokenizer exported: {os.path.join(save_dir, 'tokenizer.json')} (vocab={self._next_id})"
        )
        return save_dir


# HF byte-fallback BPE 内部把文本逐字符转成"字节符号"再做合并:
# 可打印字节保持原样，其余映射到 U+0100 私有区（空格 → "Ġ"）。


def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs], strict=False))


# ChatML 模板: 与 gleamlm/utils/chatml.py 的协议一致，Jinja2 语法 (HF serve 栈通用)。

_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


# 纯 Python 实现: SentencePiece 训练不可控难 debug，tiktoken 不可自定义训练；
# 牺牲 10-100× 速度换完全可控与可读性。
#    编码（贪心合并）两个核心函数，50 行就够了。"
