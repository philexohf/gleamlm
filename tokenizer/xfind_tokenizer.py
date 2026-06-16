"""
Xfind-Mini 分词器

基于 SentencePiece BPE，适配纯文本续写任务。
"""

import sentencepiece as spm
import os
import tempfile


class XfindTokenizer:
    """
    Xfind BPE 分词器

    32K 词表 BPE 分词器，中英混合，统一 encode/decode 接口。
    """

    def __init__(self, model_prefix=None):
        self.sp = None
        self.model_prefix = model_prefix

        # 特殊 Token (SentencePiece 默认）
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"

        # Token ID
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

        # 如果有已训练模型，加载
        if model_prefix and os.path.exists(model_prefix + ".model"):
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(model_prefix + ".model")

    def train(self, text_files, vocab_size=32000, model_prefix="bpe_32k"):
        """
        训练 BPE 分词模型

        Args:
            text_files: 文本文件列表（已清洗的纯文本）
            vocab_size: 词表大小（默认 32000）
            model_prefix: 模型保存路径前缀
        """
        self.model_prefix = model_prefix

        # 已有模型直接加载
        if os.path.exists(model_prefix + ".model"):
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(model_prefix + ".model")
            print(f"Loaded existing BPE model: {model_prefix}")
            return

        print(f"Training BPE tokenizer (vocab_size={vocab_size})...")

        # 将所有文本合并到一个临时文件
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, encoding='utf-8', suffix='.txt'
        )

        try:
            for file_path in text_files:
                if not os.path.exists(file_path):
                    print(f"  Warning: {file_path} not found, skipping")
                    continue
                print(f"  Processing: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        text = line.strip()
                        if text:
                            temp_file.write(text + "\n")
            temp_file.close()

            # 训练 BPE 模型
            spm.SentencePieceTrainer.train(
                input=temp_file.name,
                vocab_size=vocab_size,
                model_prefix=model_prefix,
                character_coverage=1.0,       # 覆盖所有字符（中英混合需要）
                model_type='bpe',
                pad_id=self.pad_id,
                unk_id=self.unk_id,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                normalization_rule_name='nmt_nfkc',
                input_sentence_size=2000000,   # 最多使用 200 万句训练
                max_sentence_length=16384,     # 允许长行
                shuffle_input_sentence=True,
            )

            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(model_prefix + ".model")
            print(f"BPE model trained: vocab_size={self.sp.get_piece_size()}")

        finally:
            os.unlink(temp_file.name)

    def encode(self, text, add_bos=False, add_eos=True):
        """
        文本 → Token ID 序列

        Args:
            text: 输入文本
            add_bos: 是否添加 <s>
            add_eos: 是否添加 </s>

        Returns:
            ID 列表，如 [1234, 5678, 3]
        """
        if self.sp is None:
            raise ValueError("BPE model not loaded. Call train() first.")

        pieces = self.sp.encode(text, out_type=int)

        ids = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(pieces)
        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids, skip_special=True):
        """
        Token ID 序列 → 文本

        Args:
            ids: ID 列表
            skip_special: 是否跳过特殊 token

        Returns:
            还原后的文本
        """
        if self.sp is None:
            raise ValueError("BPE model not loaded. Call train() first.")

        if skip_special:
            # 过滤掉 pad, bos, eos, unk
            special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
            ids = [i for i in ids if i not in special_ids]

        text = self.sp.decode(ids)
        return text

    def get_vocab_size(self):
        """返回词表大小"""
        return self.sp.get_piece_size() if self.sp else 0

    def __len__(self):
        return self.get_vocab_size()


def build_tokenizer(text_files, vocab_size=32000, model_prefix="./bpe_32k"):
    """
    构建 Xfind 分词器

    首次运行会自动训练 BPE 模型，后续复用。

    Args:
        text_files: 训练文本文件列表
        vocab_size: 词表大小
        model_prefix: 模型保存路径

    Returns:
        XfindTokenizer 实例
    """
    tokenizer = XfindTokenizer(model_prefix)
    tokenizer.train(text_files, vocab_size, model_prefix)
    return tokenizer


# ============================================================
# 快速验证
# ============================================================

if __name__ == "__main__":
    import tempfile

    # 创建临时测试文本
    test_text = tempfile.NamedTemporaryFile(
        mode='w', delete=False, encoding='utf-8', suffix='.txt'
    )
    test_text.write("这是中文测试文本用于训练BPE分词器\n")
    test_text.write("this is english test text for tokenizer training\n")
    test_text.write("人工智能正在改变世界Artificial intelligence is changing the world\n")
    test_text.close()

    # 训练分词器（小词表做快速测试）
    tokenizer = build_tokenizer(
        [test_text.name],
        vocab_size=200,
        model_prefix="./tokenizer/checkpoints/test_bpe"
    )

    print(f"\n词表大小: {len(tokenizer)}")

    # 测试编码/解码
    text = "人工智能改变世界"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    print(f"原文: {text}")
    print(f"IDs:  {ids}")
    print(f"解码: {decoded}")

    # 清理
    os.unlink(test_text.name)
    print("\n分词器测试通过")
