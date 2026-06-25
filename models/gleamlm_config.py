"""烁珑GleamLM 全局配置"""

import argparse
import os

# V4 默认路径常量，集中管理避免硬编码散落
DEFAULT_TOKENIZER_PATH = "./tokenizer/checkpoints/bbpe_12k"
DEFAULT_CHECKPOINT_DIR = "./checkpoints"
DEFAULT_DATA_DIR = "./data/splits"


def get_args():
    """获取命令行参数，可通过 python gleamlm_train.py --d_model 512 覆盖默认值"""
    parser = argparse.ArgumentParser(description='烁珑GleamLM 大模型训练配置')

    # 随机种子
    parser.add_argument("--seed", type=int, default=42, help='随机种子')

    # 训练参数
    parser.add_argument("--epochs", type=int, default=5, help='训练轮数')
    parser.add_argument("--batch_size", type=int, default=8, help='Micro-batch 大小')
    parser.add_argument("--accumulate_grad", type=int, default=8,
                        help='梯度累积步数，有效batch=batch_size×accumulate_grad')

    # 学习率
    parser.add_argument("--lr", type=float, default=3e-4, help='峰值学习率')
    parser.add_argument("--warmup_ratio", type=float, default=0.01,
                        help='Warmup 占总步数比例')

    # 模型结构（V4 Deep-Narrow：12层×512dim + BBPE 12K）
    parser.add_argument("--vocab_size", type=int, default=12001,
                        help='词表大小')
    parser.add_argument("--d_model", type=int, default=512,
                        help='模型隐藏层维度')
    parser.add_argument("--num_layers", type=int, default=12,
                        help='Decoder 层数')
    parser.add_argument("--num_heads", type=int, default=8,
                        help='查询注意力头数')
    parser.add_argument("--num_kv_heads", type=int, default=4,
                        help='KV 注意力头数（GQA）')
    parser.add_argument("--d_ff", type=int, default=1365,
                        help='SwiGLU 中间层维度')
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help='最大序列长度（上下文窗口）')
    parser.add_argument("--dropout", type=float, default=0.1,
                        help='Dropout 比例')

    # 训练控制
    parser.add_argument("--clip_grad", type=float, default=1.0,
                        help='梯度裁剪阈值')
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help='AdamW 权重衰减')
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help='标签平滑')

    parser.add_argument("--log_interval", type=int, default=50,
                        help='日志打印间隔步数')
    parser.add_argument("--eval_interval", type=int, default=500,
                        help='验证间隔步数')
    parser.add_argument("--save_interval", type=int, default=2000,
                        help='模型保存间隔步数')

    # 路径
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help='数据目录')
    parser.add_argument("--tokenizer_path", type=str,
                        default=DEFAULT_TOKENIZER_PATH,
                        help='分词器目录路径')
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help='检查点保存目录')
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help='加载检查点路径（断点续训）')
    parser.add_argument("--max_train_chars", type=int, default=1_200_000_000,
                        help='训练数据字符上限（约 1.2B chars → 1.2B tokens）')

    return parser.parse_args()
