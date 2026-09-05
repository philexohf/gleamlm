"""用本地 Qwen3-0.6B 教师对 QA→SFT 候选做语义分级（形态规则过滤的天花板补充）。

背景：qa_to_sft.py 的关键词规则能清零"形态垃圾"（成人/自述/八卦/引流），
但滤不掉"话题垃圾"——知乎历史政治高争议区（军队经商/地方政府/保蒙古…）词表
不可穷尽，只能靠语义判断。本项目 OPD 教师（Qwen3-0.6B 本地 HF）零外部依赖，
对 2 万条做四分类生成约几分钟。

分类体系（四选一，贪心生成后正则匹配）:
  知识 = 客观事实/原理/方法/清单/教程
  议论 = 观点/评价/吐槽/历史政治评论
  叙事 = 个人经历/情绪倾诉
  噪音 = 广告/乱码/无关/无法归类

用法:
  python data_tools/sft/qa_grade.py \
    --input data/nano/sft/qa_sft.jsonl \
    --labeled data/nano/sft/qa_sft_labeled.jsonl \
    --clean  data/nano/sft/qa_sft_clean.jsonl \
    --teacher_model_path checkpoints/Qwen3-0.6B
（--clean 输出 label=知识 的子集，供训练接入）
"""

import argparse
import json
import re
import time

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

_CLASS_SYSTEM = (
    "你是文本分类器。判断用户提供的“回答”属于哪种类型，只输出一个词：\n"
    "知识：客观事实、原理、方法、清单、教程；\n"
    "议论：观点、评价、吐槽、历史政治评论；\n"
    "叙事：个人经历、情绪倾诉；\n"
    "噪音：广告、乱码、无关、无法归类。"
)
_LABELS = ("知识", "议论", "叙事", "噪音")


def format_chatml(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def classify(model, tok, prompts: list[str], device: torch.device) -> list[str]:
    """贪心生成 → 剥 <think> 思考块 → 答案区匹配四类标签。

    Qwen3 系默认 thinking 模式：先出 <think>…</think> 再回答（0.6B 思考块可达
    200+ token）；本机 transformers 5.x 不支持 enable_thinking=False，故生成
    足量 token 让思考走完，剥离闭合思考块后在答案区匹配。思考块未闭合（384
    token 仍没走完）→ 标“噪音”(保守，不猜)。"""
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=1024).to(device)
    input_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=256,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    labels = []
    for seq in out:
        raw = tok.decode(seq[input_len:], skip_special_tokens=False)
        answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
        hit = next((w for w in _LABELS if w in answer), None)
        labels.append(hit if hit else "噪音")
    return labels


def main():
    p = argparse.ArgumentParser(description="QA→SFT 候选语义分级（本地教师）")
    p.add_argument("--input", default="data/nano/sft/qa_sft.jsonl")
    p.add_argument("--labeled", default="data/nano/sft/qa_sft_labeled.jsonl")
    p.add_argument("--clean", default="data/nano/sft/qa_sft_clean.jsonl")
    p.add_argument("--teacher_model_path", default="checkpoints/Qwen3-0.6B")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 条（0=全部，试跑用）")
    p.add_argument("--no_save", action="store_true", help="不落盘，只打印统计（试跑）")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.teacher_model_path)
    tok.padding_side = "left"  # decoder-only：生成需左侧 padding
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    print(f"Teacher loaded: {args.teacher_model_path} on {device}")

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Rows: {len(rows)}")

    prompts = [
        format_chatml(_CLASS_SYSTEM, f"问题：{r['instruction']}\n回答：{r['output'][:400]}\n这段回答的类型是：")
        for r in rows
    ]

    counts: dict[str, int] = {k: 0 for k in _LABELS}
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start: start + args.batch_size]
            labels = classify(model, tok, batch, device)
            for r, lab in zip(rows[start: start + args.batch_size], labels):
                r["label"] = lab
                counts[lab] = counts.get(lab, 0) + 1
            if (start // args.batch_size) % 25 == 0:
                done = start + len(batch)
                el = time.time() - t0
                print(f"  {done}/{len(rows)}  {counts}  ({el:.0f}s)", flush=True)

    print(f"Final: {counts}  ({(time.time() - t0):.0f}s)")
    if args.no_save:
        return
    with open(args.labeled, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    clean = [r for r in rows if r["label"] == "知识"]
    with open(args.clean, "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Labeled: {args.labeled} ({len(rows)}) | Clean(知识): {args.clean} ({len(clean)})")


if __name__ == "__main__":
    main()
