"""Long-form SFT data generation.

Generates detailed, multi-paragraph Chinese answers from seed topics. The long-form
responses (300-800 tokens) help the model learn sustained coherent generation beyond
the typical 50-150 token QA window.

Usage:
    $env:DEEPSEEK_API_KEY = "sk-xxxx"
    python data_tools/gen_longform_sft.py --count 500 --output data/longform_sft.jsonl
    python data_tools/gen_longform_sft.py --input data/topics.txt --output data/longform_sft.jsonl
"""

import argparse
import json
import os
import sys
import time

from data_tools._api_client import DEFAULT_MODEL, chat_completion, get_client

LONG_FORM_TOPICS = [
    "人工智能的发展历史",
    "深度学习的基本原理",
    "中国历史朝代更迭",
    "光合作用的详细过程",
    "市场经济与计划经济",
    "太阳系的形成与演化",
    "DNA的结构与功能",
    "计算机网络的七层模型",
    "中国茶文化的起源与发展",
    "全球气候变暖的原因与影响",
    "量子力学的基本概念",
    "中国古代四大发明",
    "人体免疫系统如何工作",
    "黑洞的形成与特性",
    "新能源汽车的技术路线",
    "唐诗宋词的代表人物与风格",
    "蛋白质的合成过程",
    "地震的成因与预防",
    "中国书法的发展历程",
    "宇宙大爆炸理论",
    "认知心理学的核心概念",
    "中国古典园林的设计理念",
    "可再生能源的种类与应用",
    "机器学习中的监督与无监督学习",
    "中国饮食文化的地域差异",
    "相对论的基本思想",
    "心血管疾病的预防",
    "中国戏曲的主要剧种",
    "互联网的发展阶段",
    "基因编辑技术CRISPR",
    "中国陶瓷艺术的历史",
    "板块构造学说",
    "睡眠的科学",
    "中国古代哲学流派",
    "电池技术的发展",
    "人类大脑的结构与功能",
    "中国传统节日的起源",
    "半导体物理基础",
    "环境保护的重要性与方法",
    "中国古代建筑的特点",
    "电磁波的种类与应用",
    "细胞的有丝分裂过程",
    "中国古典音乐的发展",
    "大数据技术的发展与应用",
    "材料的力学性能",
    "情绪管理的科学方法",
    "中国现代文学的流变",
    "化学键的类型与特点",
    "世界主要宗教概述",
    "纳米技术的应用前景",
]

SYSTEM_PROMPT = (
    "你是一位知识渊博的中文写作助手。请对每个问题给出详细、结构化的长文回答。"
    "要求：\n"
    "- 字数不少于300字，鼓励500-800字的详细回答\n"
    "- 使用分段结构：引言 → 主体（2-4个段落） → 总结\n"
    "- 提供具体例子、数据或典故来支撑论点\n"
    "- 语言简洁清晰，避免空洞的套话\n"
    '- 直接输出回答内容，不要加"回答："之类的标记'
)


def main():
    parser = argparse.ArgumentParser(description="Generate long-form SFT data via DeepSeek API")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Topic list file (one per line). Uses built-in topics if omitted.",
    )
    parser.add_argument("--count", type=int, default=0, help="Number of topics to use (0 = all)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="DeepSeek model name")
    parser.add_argument(
        "--api_key", type=str, default=None, help="API key (default: $env:DEEPSEEK_API_KEY)"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Set DEEPSEEK_API_KEY environment variable or pass --api_key", file=sys.stderr)
        sys.exit(1)

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(topics)} topics from {args.input}")
    else:
        topics = LONG_FORM_TOPICS[: args.count] if args.count else LONG_FORM_TOPICS
        print(f"Using {len(topics)} built-in topics")

    client = get_client(api_key)

    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for i, topic in enumerate(topics):
            topic = topic.strip()
            if not topic:
                continue
            print(f"[{i + 1}/{len(topics)}] {topic[:60]}...", end=" ", flush=True)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请详细介绍：{topic}"},
            ]
            answer = chat_completion(client, args.model, messages, temperature=0.8, max_tokens=2048)
            if answer and len(answer) >= 150:
                item = {"instruction": topic, "output": answer}
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1
                print(f"OK ({len(answer)} chars)")
            elif answer:
                print(f"SKIP (too short: {len(answer)} chars)")
            else:
                print("FAILED")
            time.sleep(0.3)

    print(f"\nDone. {written}/{len(topics)} samples written to {args.output}")


if __name__ == "__main__":
    main()
