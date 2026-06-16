"""
构建训练数据集

将清洗后的文本文件格式化为 LM 训练所需的格式，
并切分为训练集/验证集/测试集。
"""

import os
import argparse
import random


def build_dataset(input_paths, output_dir, train_ratio=0.9, valid_ratio=0.05,
                  ratios=None, total_tokens=None):
    """
    构建训练数据集

    将多个清洗后的文本文件合并，用 <|endoftext|> 分隔文档，
    然后切分为 train/valid/test。

    Args:
        input_paths: 输入文本文件路径列表
        output_dir: 输出目录
        train_ratio: 训练集比例
        valid_ratio: 验证集比例（测试集 = 1 - train - valid）
        ratios: 每个数据源的目标占比列表（与 input_paths 对应），None 则不限制
        total_tokens: 目标总 tokens（B），仅 ratios 非 None 时生效
    """
    os.makedirs(output_dir, exist_ok=True)

    # 字符到 token 的估算系数（中文 BPE 约 1 字符 ≈ 1.4 tokens）
    CHAR_TO_TOKEN = 1.4

    all_lines = []
    source_names = []

    # 计算每个源的目标字符数
    targets = None
    if ratios is not None and total_tokens is not None:
        if len(ratios) != len(input_paths):
            raise ValueError(f"ratios 数量 ({len(ratios)}) 与输入文件数 ({len(input_paths)}) 不匹配")
        total_target_chars = total_tokens * 1e9 * CHAR_TO_TOKEN
        targets = [total_target_chars * r for r in ratios]
        print(f"配比控制: 总目标 ~{total_tokens:.2f}B tokens ({total_target_chars/1e9:.2f}B 字符)")
        for p, r, t in zip(input_paths, ratios, targets):
            print(f"  {os.path.basename(p)}: {r*100:.0f}% → ~{t/1e6:.0f}M 字符")
        print()

    # 读取所有文本
    for idx, path in enumerate(input_paths):
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue

        source_name = os.path.basename(path).replace('_clean.txt', '').replace('_clean_v3.txt', '')
        source_names.append(source_name)

        print(f"Reading: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        source_chars = sum(len(l) for l in lines)

        # 如果指定了配比，按目标裁剪
        if targets is not None:
            target_chars = targets[idx]
            if source_chars > target_chars:
                # 按字符累计，取到刚好超出目标的 cut 点
                random.seed(42)
                indices = list(range(len(lines)))
                random.shuffle(indices)
                cum = 0
                cut = len(indices)
                for i, j in enumerate(indices):
                    cum += len(lines[j])
                    if cum >= target_chars:
                        cut = i + 1
                        break
                kept_indices = sorted(indices[:cut])
                lines = [lines[i] for i in kept_indices]
                kept_chars = sum(len(l) for l in lines)
                print(f"  {source_name}: {len(indices):,} → {cut:,} 行 ({source_chars/1e6:.0f}M → {kept_chars/1e6:.0f}M 字符)")
            else:
                print(f"  {source_name}: {len(lines):,} 行 ({source_chars/1e6:.0f}M 字符, 未达上限, 全部保留)")

        all_lines.extend(lines)
        print(f"  Loaded {len(lines)} lines")

    print(f"\nTotal: {len(all_lines)} lines")

    # 打乱顺序
    random.seed(42)
    random.shuffle(all_lines)

    # 计算切分点（以行数为单位，避免创建超大字符串）
    train_end_line = int(len(all_lines) * train_ratio)
    valid_end_line = int(len(all_lines) * (train_ratio + valid_ratio))

    train_lines = all_lines[:train_end_line]
    valid_lines = all_lines[train_end_line:valid_end_line]
    test_lines = all_lines[valid_end_line:]

    separator = "\n<|endoftext|>\n"

    splits = {
        'train.txt': train_lines,
        'valid.txt': valid_lines,
        'test.txt': test_lines,
    }

    for filename, lines in splits.items():
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, line in enumerate(lines):
                if i > 0:
                    f.write(separator)
                f.write(line)
        output_size = os.path.getsize(output_path)
        print(f"Saved: {output_path} ({output_size} bytes, {len(lines)} lines)")

    print("\nDataset building completed!")
    print(f"  Train: {len(train_lines)} lines ({train_ratio*100:.0f}%)")
    print(f"  Valid: {len(valid_lines)} lines ({valid_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_lines)} lines ({(1-train_ratio-valid_ratio)*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description='构建训练数据集')
    parser.add_argument('--input', type=str, nargs='+', required=True,
                        help='输入文本文件路径（可多个）')
    parser.add_argument('--output_dir', type=str, default='./data/splits',
                        help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='训练集比例')
    parser.add_argument('--valid_ratio', type=float, default=0.05,
                        help='验证集比例')
    parser.add_argument('--ratios', type=float, nargs='+', default=None,
                        help='数据源配比（与 --input 顺序对应）')
    parser.add_argument('--total_tokens', type=float, default=None,
                        help='目标总 tokens（B），配合 --ratios 使用')
    args = parser.parse_args()

    build_dataset(args.input, args.output_dir, args.train_ratio, args.valid_ratio,
                  ratios=args.ratios, total_tokens=args.total_tokens)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        # 无参数时生成 demo 数据
        print("Generating demo dataset for quick testing...\n")

        demo_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
        os.makedirs(demo_dir, exist_ok=True)

        demo_file = os.path.join(demo_dir, 'demo_text.txt')
        with open(demo_file, 'w', encoding='utf-8') as f:
            demo_lines = [
                "人工智能是计算机科学的一个重要分支，旨在创造能够模拟人类智能的系统。",
                "深度学习通过多层神经网络来学习数据的层次化表示，在图像识别、自然语言处理等领域取得了突破性进展。",
                "Transformer架构引入了自注意力机制，使得模型能够并行处理序列数据，大大提高了训练效率。",
                "预训练语言模型如BERT和GPT系列，通过在大规模无标注文本上进行自监督学习，获得了强大的语言理解能力。",
                "强化学习是一种通过与环境交互来学习最优策略的机器学习方法，在游戏、机器人控制等领域有广泛应用。",
                "自然语言处理是人工智能的核心领域之一，包括机器翻译、文本摘要、情感分析、问答系统等任务。",
                "计算机视觉技术让机器能够理解和分析图像和视频内容，广泛应用于自动驾驶、医疗诊断等领域。",
                "大语言模型通过海量数据训练，展现了涌现能力，能够完成推理、编程、创作等多种复杂任务。",
                "模型压缩技术如量化、剪枝、蒸馏，可以在保持模型性能的同时显著减小模型体积和推理延迟。",
                "知识图谱以结构化的方式表示实体间的关系，为智能问答和推荐系统提供了重要的知识基础。",
            ]
            for _ in range(50):
                for line in demo_lines:
                    f.write(line + '\n')

        print(f"Demo data saved: {demo_file}")

        # 构建数据集
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'splits')
        build_dataset([demo_file], output_dir, train_ratio=0.8, valid_ratio=0.1)
