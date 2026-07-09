"""A1: 40M 事实知识评估 — 填空测试 + A2: 实体知识探针"""

import os
import sys

import torch

from gleamlm import load_model_for_inference
from gleamlm.inference.sampler import sample_token
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DEFAULT_CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "gleamlm-nano", "checkpoints")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = f"{DEFAULT_CHECKPOINT_DIR}/sft/sft_best.pt"
TOKENIZER_PATH = DEFAULT_TOKENIZER_PATH
OUTPUT_FILE = "scripts/eval_knowledge_result.txt"

# A1: 50 fact fill-in prompts
FACT_PROMPTS = [
    ("世界上最高的山峰是", "珠穆朗玛峰"),
    ("水的化学式是", "H2O"),
    ("中国的首都是", "北京"),
    ("爱因斯坦提出了", "相对论"),
    ("光的速度大约是每秒", "30万公里"),
    ("地球绕着什么转", "太阳"),
    ("人类登上月球是在", "1969"),
    ("DNA的全称是", "脱氧核糖核酸"),
    ("第一个进入太空的人是", "加加林"),
    ("万里长城始建于", "秦朝"),
    ("世界上最大的海洋是", "太平洋"),
    ("Python编程语言的创建者是", "Guido"),
    ("一年有多少天", "365"),
    ("人体最大的器官是", "皮肤"),
    ("太阳系有几大行星", "八"),
    ("诺贝尔奖的创立者是", "诺贝尔"),
    ("马可波罗来自哪个国家", "意大利"),
    ("青霉素的发现者是", "弗莱明"),
    ("莎士比亚写了", "哈姆雷特"),
    ("光合作用需要", "阳光"),
    ("地球的卫星是", "月球"),
    ("世界上最长的河流是", "尼罗河"),
    ("元素周期表第一个元素是", "氢"),
    ("达尔文提出了", "进化论"),
    ("日本的货币是", "日元"),
    ("奥运会的发源地是", "希腊"),
    ("莫扎特是哪国人", "奥地利"),
    ("圆周率的前三位是", "3.14"),
    ("世界上人口最多的国家是", "中国"),
    ("人的正常体温大约是", "37度"),
    ("互联网的发明者是", "蒂姆"),
    ("大熊猫主要生活在", "四川"),
    ("飞机的发明者是", "莱特兄弟"),
    ("金字塔位于哪个国家", "埃及"),
    ("贝多芬是哪国人", "德国"),
    ("世界上最大的沙漠是", "撒哈拉"),
    ("咖啡原产于", "埃塞俄比亚"),
    ("诸葛亮是哪个朝代的人", "三国"),
    ("梵高画了", "向日葵"),
    ("牛顿发现了", "万有引力"),
    ("电灯的发明者是", "爱迪生"),
    ("喜马拉雅山脉位于", "亚洲"),
    ("亚马逊河流经", "南美洲"),
    ("世界上最小的国家是", "梵蒂冈"),
    ("月球上有什么", "环形山"),
    ("人体有多少块骨头", "206"),
    ("太阳从哪边升起", "东方"),
    ("企鹅生活在", "南极"),
    ("鲸鱼是哺乳动物吗", "是"),
    ("地球的年龄大约是", "46亿年"),
]

# A2: Entity probe — 20 entities, 3 question variations
ENTITY_PROBES = {
    "北京": ["中国的首都是哪里", "天安门在哪个城市", "故宫位于"],
    "光速": ["光的速度是多少", "真空中最快的是什么", "光每秒跑多远"],
    "DNA": ["DNA是什么", "遗传物质叫什么", "脱氧核糖核酸的简称"],
    "爱因斯坦": ["爱因斯坦提出了什么", "相对论是谁提出的", "E=mc^2是谁的公式"],
    "太阳": ["太阳是什么", "地球绕着什么转", "离我们最近的恒星是"],
    "珠峰": ["世界最高峰是", "珠穆朗玛峰有多高", "喜马拉雅山最高峰是"],
    "太平洋": ["最大的海洋", "太平洋在哪", "地球上最大的水体"],
    "月球": ["月球的别称", "地球的卫星是什么", "人类登陆过哪个星球"],
    "诺贝尔": ["诺贝尔奖是谁设立的", "诺贝尔发明了什么", "最高科学奖项"],
    "金字塔": ["金字塔在哪个国家", "埃及有什么著名建筑", "法老的陵墓"],
    "钢琴": ["贝多芬弹什么乐器", "88键的乐器", "莫扎特用的乐器"],
    "计算机": ["计算机的核心部件", "CPU是什么", "电脑的中央处理器"],
    "奥运会": ["奥运会起源", "奥运五环代表什么", "奥林匹克在哪里"],
    "癌症": ["癌症是什么病", "恶性肿瘤的俗称", "最难治的疾病之一"],
    "互联网": ["互联网是什么", "万维网简称", "www是什么意思"],
    "恐龙": ["恐龙是什么动物", "史前最大的爬行动物", "侏罗纪的主角"],
    "火山": ["火山为什么会喷发", "熔岩从哪里出来", "富士山是什么"],
    "维生素": ["维生素C的作用", "水果含有什么营养", "人体必需的微量有机物"],
    "黑洞": ["黑洞是什么", "能吞噬光的星体", "爱因斯坦理论预言的"],
    "病毒": ["病毒是什么", "比细菌还小的微生物", "新冠的是什么"],
}


def load_model_and_tokenizer():
    print(f"Loading SFT model: {MODEL_PATH}")
    tok = BBPETokenizer.load(TOKENIZER_PATH)
    model, config = load_model_for_inference(MODEL_PATH, DEVICE)
    model.eval()
    total, _ = model.get_num_params()
    print(f"  Model: {total / 1e6:.2f}M params, Vocab: {tok.get_vocab_size()}")
    return model, tok, config


def generate(model, tokenizer, prompt, max_new_tokens=64):
    """Generate text using KV cache + shared sampler"""
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], device=DEVICE)
    generated_ids = prompt_ids.copy()
    past_kv = None

    with torch.no_grad():
        for _i in range(max_new_tokens):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, past_kv = model(input_ids, past_kv_list=past_kv)
            next_token = sample_token(
                logits[:, -1, :],
                temperature=0.7,
                top_k=50,
                generated_ids=generated_ids,
            )
            token_id = next_token.item()
            if token_id == tokenizer.eos_id:
                break
            generated_ids.append(token_id)
            input_ids = torch.tensor([[token_id]], device=DEVICE)

    full = tokenizer.decode(generated_ids)
    generated = full[len(prompt):].strip() if full.startswith(prompt) else full.strip()
    return generated[:200]


def check_knowledge(generated, expected):
    """Check if generated text contains expected answer"""
    generated_lower = generated.lower().replace(" ", "")
    for kw in expected.split(","):
        kw = kw.strip().lower()
        if kw in generated_lower:
            return "CORRECT"
    # Check for hallucination indicators
    hallucination_keywords = ["保护区", "氧化物", "太阳质量", "热带"]
    for hw in hallucination_keywords:
        if hw in generated:
            return "HALLUCINATION"
    return "WRONG"


def run_a1(model, tok):
    print("\n" + "=" * 70)
    print("A1: FACT FILL-IN TEST (50 prompts)")
    print("=" * 70)

    results = {"CORRECT": 0, "WRONG": 0, "HALLUCINATION": 0}
    details = []

    for prompt, expected in FACT_PROMPTS:
        generated = generate(model, tok, prompt, max_new_tokens=64)
        result = check_knowledge(generated, expected)
        results[result] = results.get(result, 0) + 1
        details.append((prompt, expected, generated[:100], result))

        if result == "CORRECT":
            print(f"  [OK]    {prompt} -> {generated[:60]}...")
        elif result == "HALLUCINATION":
            print(f"  [HALU]  {prompt} -> {generated[:60]}...")
        else:
            print(f"  [WRONG] {prompt} -> {generated[:60]}...")

    total = sum(results.values())
    print(
        f"\n  Results: {results['CORRECT']}/{total} correct ({100 * results['CORRECT'] / total:.0f}%), "
        f"{results['HALLUCINATION']} hallucinations"
    )
    return {"results": results, "details": details}


def run_a2(model, tok):
    print("\n" + "=" * 70)
    print("A2: ENTITY PROBE (20 entities x 3 questions = 60")
    print("=" * 70)

    entity_scores = {}
    for entity, questions in ENTITY_PROBES.items():
        correct = 0
        for q in questions:
            generated = generate(model, tok, q, max_new_tokens=64)
            if entity in generated:
                correct += 1
        entity_scores[entity] = correct
        icon = "***" if correct >= 2 else ("*" if correct == 1 else " ")
        print(f"  [{icon}] {entity}: {correct}/3")

    consistent = sum(1 for v in entity_scores.values() if v >= 2)
    print(f"\n  Consistent entities (>=2/3): {consistent}/20 ({100 * consistent / 20:.0f}%)")
    return entity_scores


def main():
    f = open(OUTPUT_FILE, "w", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = f
    try:
        model, tok, config = load_model_and_tokenizer()
        a1 = run_a1(model, tok)
        a2 = run_a2(model, tok)

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        total_q = sum(a1["results"].values())
        print(
            f"A1 Fact Accuracy: {a1['results']['CORRECT']}/{total_q} "
            f"({100 * a1['results']['CORRECT'] / total_q:.0f}%)"
        )
        print(
            f"A1 Hallucination Rate: {a1['results']['HALLUCINATION']}/{total_q} "
            f"({100 * a1['results']['HALLUCINATION'] / total_q:.0f}%)"
        )
        consistent = sum(1 for v in a2.values() if v >= 2)
        print(f"A2 Entity Consistency: {consistent}/20 ({100 * consistent / 20:.0f}%)")
    finally:
        sys.stdout = orig_stdout
        f.close()
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
