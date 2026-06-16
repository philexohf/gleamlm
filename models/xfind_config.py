"""
Xfind-Mini 现代 Decoder-only LLM - 全局配置

参考 Llama/Qwen 架构，专为 RTX 4070 Ti 12GB 优化
"""

import argparse
import os


def get_args():
    """获取命令行参数，可通过 python xfind_train.py --d_model 512 覆盖默认值"""
    parser = argparse.ArgumentParser(description='Xfind-Mini 大模型训练配置')

    # 随机种子
    parser.add_argument("--seed", type=int, default=42, help='随机种子')

    # 训练参数
    parser.add_argument("--epochs", type=int, default=8, help='训练轮数')
    parser.add_argument("--batch_size", type=int, default=8, help='Micro-batch 大小（显存安全值 8）')
    parser.add_argument("--accumulate_grad", type=int, default=8,
                        help='梯度累积步数（有效batch=batch_size*accumulate_grad）')

    # 学习率
    parser.add_argument("--lr", type=float, default=3e-4, help='峰值学习率')
    parser.add_argument("--warmup_ratio", type=float, default=0.01,
                        help='Warmup 占总步数比例（1%）')

    # 模型结构（固定为 Xfind-Mini 约 35M 参数规格）
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help='词表大小')
    parser.add_argument("--d_model", type=int, default=512,
                        help='模型隐藏层维度')
    parser.add_argument("--num_layers", type=int, default=8,
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
                        help='Dropout 比例（39M 小模型建议 0.1-0.12）')

    # 训练控制
    parser.add_argument("--clip_grad", type=float, default=1.0,
                        help='梯度裁剪阈值')
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help='AdamW 权重衰减')
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help='标签平滑（打破 softmax 尖锐分布，39M 建议 0.1）')

    # 日志与保存
    parser.add_argument("--log_interval", type=int, default=50,
                        help='日志打印间隔（步）')
    parser.add_argument("--eval_interval", type=int, default=500,
                        help='验证间隔（步）')
    parser.add_argument("--save_interval", type=int, default=2000,
                        help='模型保存间隔（步）')

    # 路径
    parser.add_argument("--data_dir", type=str, default="./data/splits",
                        help='数据目录')
    parser.add_argument("--tokenizer_path", type=str,
                        default="./tokenizer/checkpoints/bpe_32k",
                        help='分词器模型前缀')
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help='检查点保存目录')
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help='加载检查点路径（断点续训）')

    return parser.parse_args()


class XfindConfig:
    """
    默认配置类 - 用于推理时加载模型参数和快速查阅默认值
    Xfind-Mini: 约 39M 参数
    """
    seed = 42
    epochs = 5
    batch_size = 16
    accumulate_grad = 4

    lr = 3e-4
    warmup_ratio = 0.01

    vocab_size = 32000
    d_model = 512
    num_layers = 8
    num_heads = 8
    num_kv_heads = 4
    d_ff = 1365
    max_seq_len = 1024
    dropout = 0.1

    clip_grad = 1.0
    weight_decay = 0.01
    label_smoothing = 0.1

    log_interval = 50
    eval_interval = 500
    save_interval = 2000

    data_dir = "./data/splits"
    tokenizer_path = "./tokenizer/checkpoints/bpe_32k"
    checkpoint_dir = "./checkpoints"
    load_checkpoint = None

    device = "cuda"
    pad_token_id = 0


# 显存适配指南（RTX 4070 Ti 12GB）
#
# 注意力权重 O(seq²) 是显存大户：seq_len=1024→268MB/层，2048→1.07GB/层
# v2 推荐使用 batch_size=8, accumulate_grad=8（batch_size=16 会 OOM）
#
# | batch_size | max_seq_len | accumulate_grad | 有效batch | 显存实测 |
# |------------|-------------|-----------------|-----------|----------|
# | 8          | 1024        | 8               | 64        | ~5 GB    |
# | 16 ❌      | 1024        | 4               | 64        | >12 GB（OOM）|
# | 8          | 2048        | 8               | 64        | ~7 GB    |
