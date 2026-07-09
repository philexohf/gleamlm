"""GleamLM Byte-Level BPE Tokenizer. 基于 GPT-2 Byte-Level BPE 方案，256 字节基座 + BPE 合并规则。"""

from __future__ import annotations

import heapq
import json
import os
import re
import time
from collections import defaultdict

# CJK 逐字 + 非 CJK 连续段预分词正则
_PRE_TOKENIZE_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]|"
    r"[\u3000-\u303f\uff00-\uffef]|"
    r"[^\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f\uff00-\uffef]+"
)


def _pre_tokenize_re(text: str) -> list[str]:
    """正则预分词：CJK 逐字切分 + 非 CJK 连续段保持"""
    return [m.group(0) for m in _PRE_TOKENIZE_RE.finditer(text)]


# （当前预分词完全由正则 _PRE_TOKENIZE_RE 完成）


# BBPE Tokenizer
class BBPETokenizer:
    """Byte-Level BPE 分词器"""

    def __init__(self) -> None:
        # 256 字节基座
        self.id_to_byte: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        # BPE 合并规则: (id_a, id_b) -> merged_id
        self.merges: dict[tuple[int, int], int] = {}
        # 反向: merged_id -> (id_a, id_b)
        self.merge_pairs: dict[int, tuple[int, int]] = {}

        # 特殊 token 映射
        self.special_tokens: dict[str, int] = {}
        self.id_to_special: dict[int, str] = {}

        # 下一个可用 token ID（256 之后）
        self._next_id = 256

        # 预分词方法（C 级字符类正则）
        self._pre_tokenize_fn = _pre_tokenize_re

        # 特殊 token 切分正则（_add_special_tokens 或 load 后构建）
        self._special_regex = None

    # 训练

    @classmethod
    def train_from_files(
        cls,
        text_files: list[str],
        vocab_size: int = 12002,
        save_dir: str | None = None,
        max_train_chars: int = 500_000_000,
        ratios: list[float] | None = None,
    ) -> BBPETokenizer:
        """从文本文件训练 BBPE tokenizer"""
        tokenizer = cls()
        if ratios is None:
            ratios = [1.0 / len(text_files)] * len(text_files)

        print(f"Training BBPE tokenizer (vocab_size={vocab_size})...")
        print(f"  Input files: {len(text_files)}, max_chars={max_train_chars / 1e6:.0f}M")
        for fp, r in zip(text_files, ratios, strict=False):
            print(f"    [{r * 100:.0f}%] {os.path.basename(fp)}")

        print("  Step 1/3: Pre-tokenizing and converting to byte sequences...")
        byte_sequences = tokenizer._pre_tokenize_files(
            text_files, max_chars=max_train_chars, ratios=ratios
        )
        total_pairs = sum(len(seq) - 1 for seq in byte_sequences if len(seq) > 1)
        print(f"  Collected {len(byte_sequences):,} words, {total_pairs:,} initial pairs")

        n_merges = vocab_size - 256 - 10  # 预留 10 个特殊 token 位置
        print(f"  Step 2/3: Training {n_merges} BPE merges...")

        # 构建 pair → 出现位置的索引（用 dict/set 实现 O(1) 增删）
        print("    Building pair index...", end=" ", flush=True)
        t_idx = time.time()
        pair_to_positions = defaultdict(dict)  # pair -> {(wid,pos): True}
        for wid, seq in enumerate(byte_sequences):
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_to_positions[pair][(wid, i)] = True
        print(f"{len(pair_to_positions):,} unique pairs ({time.time() - t_idx:.1f}s)")

        # 初始化最大堆：(-count, pair)，用于 O(log P) 找最频繁 pair
        print("    Building max-heap...", end=" ", flush=True)
        t_heap = time.time()
        heap = []
        for pair, positions in pair_to_positions.items():
            heapq.heappush(heap, (-len(positions), pair))
        print(f"done ({time.time() - t_heap:.1f}s)")

        t_start = time.time()
        pbar_interval = max(1, n_merges // 200)  # 每 0.5% 刷新进度条
        print(f"    Merging (pbar every {pbar_interval} steps)...", flush=True)

        for merge_step in range(n_merges):
            if not pair_to_positions:
                print(f"\n  No more pairs to merge at step {merge_step}")
                break

            # 惰性删除堆：跳过已失效或计数变化的条目
            while heap:
                neg_count, best_pair = heapq.heappop(heap)
                if best_pair not in pair_to_positions:
                    continue  # 已移除的 pair
                actual_count = len(pair_to_positions[best_pair])
                if actual_count != -neg_count:
                    # 计数过时，压入新值
                    heapq.heappush(heap, (-actual_count, best_pair))
                    continue
                break
            else:
                print(f"\n  Heap empty at step {merge_step}")
                break

            best_count = len(pair_to_positions[best_pair])

            if best_count < 2:
                print(f"\n  All pairs have count=1 at step {merge_step}, stopping")
                break

            # 创建新 token
            new_id = tokenizer._next_id
            tokenizer.merges[best_pair] = new_id
            tokenizer.merge_pairs[new_id] = best_pair
            tokenizer.id_to_byte[new_id] = (
                tokenizer.id_to_byte[best_pair[0]] + tokenizer.id_to_byte[best_pair[1]]
            )
            tokenizer._next_id += 1

            # 更新受影响的序列
            affected = pair_to_positions.pop(best_pair)
            # 按 word_idx 分组，从后往前更新避免索引偏移
            by_word = defaultdict(list)
            for wid, pos in affected:
                by_word[wid].append(pos)

            for wid, positions in by_word.items():
                seq = byte_sequences[wid]
                positions.sort(reverse=True)  # 从后往前
                for pos in positions:
                    if pos >= len(seq) - 1:
                        continue
                    if (seq[pos], seq[pos + 1]) != best_pair:
                        continue

                    # 移除旧 pair 索引（O(1) dict pop）
                    if pos > 0:
                        old_left = (seq[pos - 1], seq[pos])
                        if old_left in pair_to_positions:
                            pair_to_positions[old_left].pop((wid, pos - 1), None)
                            if not pair_to_positions[old_left]:
                                del pair_to_positions[old_left]

                    if pos + 2 < len(seq):
                        old_right = (seq[pos + 1], seq[pos + 2])
                        if old_right in pair_to_positions:
                            pair_to_positions[old_right].pop((wid, pos + 1), None)
                            if not pair_to_positions[old_right]:
                                del pair_to_positions[old_right]

                    # 合并: seq[pos], seq[pos+1] → new_id
                    seq[pos] = new_id
                    del seq[pos + 1]

                    # 新 pair: (seq[pos-1], new_id)（O(1) dict insert）
                    if pos > 0:
                        new_pair = (seq[pos - 1], new_id)
                        pair_to_positions[new_pair][(wid, pos - 1)] = True
                        heapq.heappush(heap, (-len(pair_to_positions[new_pair]), new_pair))
                    # 新 pair: (new_id, seq[pos+1])
                    if pos < len(seq) - 1:
                        new_pair = (new_id, seq[pos + 1])
                        pair_to_positions[new_pair][(wid, pos)] = True
                        heapq.heappush(heap, (-len(pair_to_positions[new_pair]), new_pair))

            # 进度条
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
                # 写临时文件避免 Windows GBK 编码问题
                print(
                    f"\r  [{bar}] {pct:5.1f}% ({step}/{n_merges}) | "
                    f"pair=({best_pair[0]},{best_pair[1]}) cnt={best_count} | "
                    f"ETA {eta_str}",
                    end="",
                    flush=True,
                )

            # 里程碑日志（换行打印，不覆盖进度条）
            if step % 1000 == 0:
                print()  # 换行

        print(f"\n  Trained {len(tokenizer.merges)} merges, vocab_size={tokenizer._next_id}")

        tokenizer._add_special_tokens()

        print(f"  Step 3/3: Injected special tokens, final vocab={tokenizer._next_id}")

        if save_dir:
            tokenizer.save(save_dir)

        return tokenizer

    def _pre_tokenize_files(
        self, text_files: list[str], max_chars: int = 500_000_000, ratios: list[float] | None = None
    ) -> list[list[int]]:
        """按配比从多文件中采样读取 → 预分词 → 转字节序列"""
        byte_sequences: list[list[int]] = []
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
                        byte_seq = list(word.encode("utf-8"))
                        if byte_seq:
                            byte_sequences.append(byte_seq)
                            file_words += 1

                    pct = 100 * (quotas[i] - text_remaining) / quotas[i]
                    print(f"\r      {pct:.0f}% ({file_words:,} words)", end="", flush=True)

            total_words += file_words
            print(f" → {file_words:,} words")

        print(
            f"    Total: {total_words:,} words, "
            f"{sum(quotas) / 1e6:.1f}M chars from {len(text_files)} files"
        )
        return byte_sequences

    def _pre_tokenize(self, text: str) -> list[str]:
        """将文本拆分为 word 列表（C 级字符类正则）"""
        return self._pre_tokenize_fn(text)

    def _add_special_tokens(self) -> None:
        """注入特殊 token 到词表末尾"""
        specials = [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<pad>",
            "<unk>",
            "<s>",
            "</s>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
        ]
        for token in specials:
            if token not in self.special_tokens:
                tid = self._next_id
                self.special_tokens[token] = tid
                self.id_to_special[tid] = token
                # 写入 id_to_byte 供 decode() 还原
                self.id_to_byte[tid] = token.encode("utf-8")
                self._next_id += 1

        # 设置常用别名
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"

        self.pad_id = self.special_tokens["<pad>"]
        self.unk_id = self.special_tokens["<unk>"]
        self.bos_id = self.special_tokens["<s>"]
        self.eos_id = self.special_tokens["</s>"]

        # 构建特殊 token 切分正则（按长度降序，确保最长匹配优先）
        self._build_special_regex()

    # 编码 / 解码

    def _build_special_regex(self) -> None:
        """构建用于切分特殊 token 的正则（按长度降序，最长匹配优先）"""
        if not self.special_tokens:
            self._special_regex = None
            return
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self._special_regex = re.compile("(" + "|".join(escaped) + ")")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """文本 → token ID 序列

        先用正则按特殊 token 边界切分文本，再对每段普通文本
        独立预分词 + BPE 编码。这确保特殊 token 无论在 CJK
        还是非 CJK 上下文中都能被正确识别。
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")

        ids = []
        if add_bos:
            ids.append(self.bos_id)

        parts = self._special_regex.split(text) if self._special_regex is not None else [text]

        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                # 特殊 token → 直接映射 ID
                ids.append(self.special_tokens[part])
            else:
                # 普通文本 → 预分词 → 逐词 BPE 编码
                words = self._pre_tokenize(part)
                for word in words:
                    byte_seq = list(word.encode("utf-8"))
                    ids.extend(self._apply_bpe_to_bytes(byte_seq))

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def _apply_bpe_to_bytes(self, byte_seq: list[int]) -> list[int]:
        """对字节序列应用 BPE 合并"""
        if not byte_seq:
            return []

        seq = list(byte_seq)
        while len(seq) > 1:
            best_rank = float("inf")
            best_pos = -1
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                if pair in self.merges:
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

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Token ID 序列 → 文本"""
        byte_buffer = bytearray()

        for tid in ids:
            if skip_special and tid in self.id_to_special:
                continue
            if tid in self.id_to_byte:
                byte_buffer.extend(self.id_to_byte[tid])
            else:
                # 回退到 <unk> 的字节表示
                byte_buffer.extend(b"?")

        return byte_buffer.decode("utf-8", errors="replace")

    # 便捷方法

    def encode_batch(
        self, texts: list[str], add_bos: bool = False, add_eos: bool = False
    ) -> list[list[int]]:
        """批量编码"""
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def token_to_id(self, token: str) -> int:
        """特殊 token 字符串 → ID"""
        return self.special_tokens.get(token, self.unk_id)

    def get_vocab_size(self) -> int:
        """返回词表大小"""
        return self._next_id

    def __len__(self) -> int:
        return self.get_vocab_size()

    # 持久化

    def save(self, save_dir: str) -> None:
        """保存 tokenizer 到目录"""
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
        """从目录加载 tokenizer"""
        path = os.path.join(save_dir, "bbpe_tokenizer.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer not found: {path}")

        with open(path, encoding="utf-8") as f:
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

        tokenizer.pad_token = "<pad>"
        tokenizer.unk_token = "<unk>"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.pad_id = tokenizer.special_tokens.get("<pad>", 0)
        tokenizer.unk_id = tokenizer.special_tokens.get("<unk>", 1)
        tokenizer.bos_id = tokenizer.special_tokens.get("<s>", 2)
        tokenizer.eos_id = tokenizer.special_tokens.get("</s>", 3)

        tokenizer._build_special_regex()

        print(f"BBPE tokenizer loaded: {path} (vocab_size={tokenizer._next_id})")
        return tokenizer
