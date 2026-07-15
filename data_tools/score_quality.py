"""数据质量评分脚本。
用 DeepSeek API 标注样本 → 训练 fastText 分类器 → 全量数据评分。

3 分类：低质(1-2分) / 中等(3分) / 高质(4-5分)

用法：
    set DEEPSEEK_API_KEY=sk-xxxx
    pip install fasttext-wheel openai

    # 首次标注
    python data_tools/score_quality.py --input data/lite_data/train.txt --sample 5000

    # 断点续标（追加到已有标注，跳过已标注的文本）
    python data_tools/score_quality.py --input data/lite_data/train.txt --sample 5000 --resume

    # 训分类器 + 全量评分
    python data_tools/score_quality.py --input data/lite_data/train.txt --train --score
"""

import argparse
import json
import os
import random
import sys
import time

from data_tools._api_client import DEFAULT_BASE_URL, DEFAULT_MODEL, chat_completion, get_client

SCORING_SYSTEM_PROMPT = """你是一个专业的中文文本质量评估器。请根据以下标准对文本评分（1-5 分）：

1 分 — 无信息/垃圾：纯乱码、广告、导航菜单、纯链接列表、无完整句子
2 分 — 低质：流水账、纯数字/符号列表、严重重复内容、不成段的碎片
3 分 — 可接受：信息基本完整、有简单逻辑结构、可作为训练填充数据
4 分 — 高质量：内容扎实、逻辑清晰、有一定知识密度和教育价值
5 分 — 优秀：结构完整、知识密度高、内容准确、适合作为教科书/百科条目

只返回一个整数（1-5），不要任何其他内容。"""


def map_score(score: int) -> int:
    """5 阶 → 3 类：低(0) / 中(1) / 高(2)"""
    if score <= 2:
        return 0
    if score == 3:
        return 1
    return 2


def score_text(client, model: str, text: str, max_retries: int = 3) -> int | None:
    """用 API 对单条文本评分，返回 1-5 或 None。"""
    messages = [
        {"role": "system", "content": SCORING_SYSTEM_PROMPT},
        {"role": "user", "content": text[:3000]},
    ]
    for _ in range(max_retries):
        result = chat_completion(client, model, messages, temperature=0.0, max_tokens=4)
        if result is None:
            return None
        try:
            score = int("".join(c for c in result if c.isdigit()))
            if 1 <= score <= 5:
                return score
            print(f"  Warning: invalid score '{result}', retrying...")
        except ValueError:
            print(f"  Warning: non-numeric result '{result}', retrying...")
    return None


def load_labeled_texts(label_path: str) -> set[str]:
    """读取已标注的文本 set（用于去重）"""
    if not os.path.exists(label_path):
        return set()
    with open(label_path, encoding="utf-8") as f:
        data = json.load(f)
    return {item["text"] for item in data}


def sample_lines(
    input_path: str, n: int, exclude: set[str] | None = None, min_len: int = 20
) -> list[str]:
    """从文本文件中随机采样 n 行（排除已标注文本）"""
    exclude = exclude or set()
    print(f"Reading {input_path}...")
    with open(input_path, encoding="utf-8") as f:
        total = 0
        pool: list[str] = []
        for line in f:
            stripped = line.strip()
            total += 1
            if len(stripped) >= min_len and stripped not in exclude:
                pool.append(stripped)
        if total == 0:
            print("Error: empty file")
            sys.exit(1)

    print(f"  {total:,} total lines, {len(pool):,} >= {min_len} chars")
    n = min(n, len(pool))
    random.seed(42)
    samples = random.sample(pool, n)
    print(f"  Sampled {n} lines")
    return samples


def label_samples(
    samples: list[str],
    client,
    model: str,
    output_path: str,
    delay: float = 0.3,
    resume: bool = False,
) -> None:
    """用 DeepSeek API 标注样本质量分"""
    results: list[dict] = []
    if resume and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing labels")

    start_idx = len(results)
    fail_count = 0
    for i, text in enumerate(samples):
        print(f"[{start_idx + i + 1}] ", end="", flush=True)
        score = score_text(client, model, text)
        if score is not None:
            results.append({"score": score, "text": text})
            print(f"score={score}")
        else:
            fail_count += 1
            print("FAIL")
        time.sleep(delay)
        if (i + 1) % 200 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    new = len(results) - start_idx
    print(f"\nDone: {new} newly labeled (total {len(results)}), {fail_count} failed")
    print(f"Saved: {output_path}")


def convert_to_fasttext(json_path: str, ft_path: str) -> int:
    """将标注结果转为 3 类 fastText 格式"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    with open(ft_path, "w", encoding="utf-8") as out:
        for item in data:
            text = item["text"].replace("\n", " ")
            label = map_score(item["score"])
            out.write(f"__label__{label} {text}\n")
    print(f"Converted {len(data)} samples → {ft_path} (3-class)")
    return len(data)


def train_fasttext(ft_path: str, model_path: str) -> None:
    """训练 fastText 分类器"""
    try:
        import fasttext
    except ImportError:
        print("请先安装 fasttext-wheel: pip install fasttext-wheel")
        sys.exit(1)

    print(f"Training fastText on {ft_path}...")
    model = fasttext.train_supervised(
        input=ft_path,
        epoch=25,
        lr=0.5,
        wordNgrams=2,
        dim=100,
        loss="softmax",
        verbose=2,
    )
    model.save_model(model_path)
    print(f"Saved: {model_path}")

    n, p, r = model.test(ft_path)
    print(f"  Samples:     {n}")
    print(f"  Precision@1: {p:.3f}")
    print(f"  Recall@1:    {r:.3f}")


def score_all(input_path: str, model_path: str, output_path: str, min_len: int = 20) -> None:
    """用训练好的 fastText 分类器对全量数据评分"""
    try:
        import fasttext
    except ImportError:
        print("请先安装 fasttext-wheel: pip install fasttext-wheel")
        sys.exit(1)

    model = fasttext.load_model(model_path)
    total = 0
    scored = 0

    def _predict(text: str) -> str | None:
        """fasttext C++ predict，绕过 numpy 2.x 兼容问题"""
        results = model.f.predict(text, 1, 0.0, "")
        if not results:
            return None
        return results[0][1]

    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            stripped = line.strip()
            total += 1
            if len(stripped) < min_len:
                fout.write("0\n")
                continue
            pred = _predict(stripped[:3000])
            label = 0 if pred is None else int(pred.replace("__label__", ""))
            fout.write(f"{label}\n")
            scored += 1
            if total % 100000 == 0:
                print(f"  {total:,} lines processed")

    print(f"Done: {total:,} total, {scored:,} scored")
    print(f"Saved: {output_path}")


def print_distribution(json_path: str) -> None:
    """打印标注分布（原始 5 阶 + 3 类合并）"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    counts_5: dict[int, int] = {}
    counts_3: dict[int, int] = {0: 0, 1: 0, 2: 0}
    for item in data:
        s = item["score"]
        counts_5[s] = counts_5.get(s, 0) + 1
        counts_3[map_score(s)] += 1
    total = len(data)
    labels = {0: "低质", 1: "中等", 2: "高质"}
    print(f"\n标注分布（{total} 条）:")
    print("  原始 5 阶:")
    for k in range(1, 6):
        c = counts_5.get(k, 0)
        print(f"    {k}: {c:5d}  {c / total * 100:5.1f}%")
    print("  合并 3 类:")
    for k in range(3):
        c = counts_3[k]
        print(f"    {labels[k]}: {c:5d}  {c / total * 100:5.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="数据质量评分")
    parser.add_argument("--input", type=str, required=True, help="输入文本文件路径")
    parser.add_argument("--sample", type=int, default=0, help="采样数量（0=跳过标注）")
    parser.add_argument("--resume", action="store_true", help="追加标注（跳过已标注文本）")
    parser.add_argument("--output", type=str, default=None, help="标注输出 JSON 路径")
    parser.add_argument("--api_key", type=str, default=None, help="DeepSeek API Key")
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--delay", type=float, default=0.3, help="API 调用间隔秒数")
    parser.add_argument("--train", action="store_true", help="训 fastText 分类器")
    parser.add_argument("--score", action="store_true", help="全量数据评分")
    parser.add_argument(
        "--ft_model", type=str, default="data/quality_model.bin", help="分类器模型路径"
    )
    parser.add_argument("--score_out", type=str, default="data/train.score", help="评分输出路径")
    args = parser.parse_args()

    base = os.path.splitext(os.path.basename(args.input))[0]
    label_path = args.output or f"data/{base}_labeled.json"

    # Step 1: 采样 + API 标注
    if args.sample > 0:
        api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("Error: DEEPSEEK_API_KEY 未设置")
            sys.exit(1)

        exclude = load_labeled_texts(label_path) if args.resume else None
        if exclude:
            print(f"Resume: {len(exclude)} previously labeled lines will be skipped")
        samples = sample_lines(args.input, args.sample, exclude=exclude)
        client = get_client(api_key, args.base_url)
        print(f"API: {args.base_url}, model: {args.model}")
        label_samples(samples, client, args.model, label_path, args.delay, resume=args.resume)

    # Step 2: 转 fastText 格式 + 分布
    if os.path.exists(label_path):
        print_distribution(label_path)
        ft_path = os.path.splitext(label_path)[0] + ".ft.txt"
        convert_to_fasttext(label_path, ft_path)
    else:
        print(f"Note: {label_path} 不存在，跳过")

    # Step 3: 训练分类器
    if args.train:
        ft_path = os.path.splitext(label_path)[0] + ".ft.txt"
        if not os.path.exists(ft_path):
            print(f"Error: {ft_path} 不存在，请先运行 --sample")
            sys.exit(1)
        train_fasttext(ft_path, args.ft_model)

    # Step 4: 全量评分
    if args.score:
        if not os.path.exists(args.ft_model):
            print(f"Error: {args.ft_model} 不存在，请先运行 --train")
            sys.exit(1)
        score_all(args.input, args.ft_model, args.score_out)


if __name__ == "__main__":
    main()
