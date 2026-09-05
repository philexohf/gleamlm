"""从 QA 语料（知乎问答）提取 SFT 指令数据。

背景：qa_dedup.txt（filter_qa 后 916,914 条）知识型问题占 60.8%，但混有
观点评价（如何评价/什么体验 9.3%）、人物八卦（2.7%）、知乎腔（谢邀/利益相关 9.0%），
直接全量作 SFT 会引入主观噪音 —— 此处按规则过滤后产出 instruction/output。

过滤规则（白名单优先，命中即知识型候选；黑名单命中即弃）:
  知识型问题: 什么是/为什么/如何/怎么/怎样/有哪些/区别/原理/步骤/方法/技巧/
              建议/介绍/解释/适合/推荐/需要/学习/能否/能不能/多少/几个
  黑名单问题（观点/八卦/情感）: 如何评价/如何看待/什么体验/什么感觉/哪些瞬间/
              你见过/大家怎么/有人知道/怎么看待/是不是只有/算不算/暗恋/约会/
              表白/告白/相亲/女友/男友/异地恋/室友/宿舍关系/分手/出轨/前任/
              婆婆/丈母娘/老公/老婆/明星/偶像/粉丝/爱豆/网红/主播
  内容安全（问题+答案双查，命中即弃）: 成人内容与敏感词表，绝不进训练数据
   红线（问题/答案双查）: 心理健康自述（抑郁/焦虑症/自杀/想死/不想活）、站内互撕
               （撕逼/勃学/大V）、营销引流（点赞评论/公众号/私信我/关注我）
   问题向（观点/人物评价/八卦/情感）: 怎样的人/是什么样的人 + 八卦情感词表
   敏感话题（问题/答案双查）: 死刑/配枪/文革/台湾/香港/共产党/国民党等（宁多勿漏）
   自述锚点（答案命中即弃）: 我现在/我今年/我也有/我自己/我朋友/我觉得/我认为…
   自述密度: 答案中“我”出现 ≥6 次即弃（个人叙事浓度过高）
  黑名单答案: 谢邀/泻药/利益相关/不请自来/先说结论/题主/楼主/本人男/本人女/
              再更新/票圈/附上地址/视频链接/纯手打/哈哈哈哈哈/知乎
  URL 变体残留: http/www/.com/.cn/.org/youku/优酷/bilibili（filter_qa 漏网的变形链接）
  长度:       问题 10~80 字，答案 40~400 字（过滤碎片答与超长口语流）

用法:
  # 预览（随机抽 20 条给人审质量）
  python data_tools/sft/qa_to_sft.py --preview --max 20 --output /tmp/qa_preview.jsonl

  # 正式抽取：默认取合格候选池的 10%（约 3.8 万条），确定性 seed
  python data_tools/sft/qa_to_sft.py --output data/nano/sft/qa_candidates.jsonl

  # 追加合并到 SFT 训练集（先自动备份）
  python data_tools/sft/qa_to_sft.py --append-to data/nano/sft/sft_data.jsonl
"""

import argparse
import json
import os
import random
import re
import sys

_KNOW_Q = re.compile(
    r"什么是|什么是|为什么|如何|怎么|怎样|有哪些|区别|原理|步骤|方法|技巧|"
    r"建议|介绍|解释|适合|推荐|需要|学习|能否|能不能|多少|几个"
)
_OPINION_Q = re.compile(
    r"如何评价|如何看待|什么体验|什么感觉|哪些瞬间|你见过|大家怎么|"
    r"有人知道|怎么看待|是不是只有|算不算|你如何看|你觉得|被低估|是怎样的人|是什么样的人|暗恋|约会|"
    r"表白|告白|相亲|女友|男友|异地恋|室友|宿舍关系|分手|出轨|前任|婆婆|"
    r"丈母娘|老公|老婆|明星|偶像|粉丝|爱豆|网红|主播"
)
# 内容安全词表：问题与答案均检查，命中即弃（教培项目红线，宁多勿漏）
_NSFW = re.compile(
    r"娇喘|啪啪啪|上床|做爱|性生活|约炮|一夜情|自慰|援交|嫖娼|口交|肛交|"
    r"a片|小黄片|裸体|裸照|丝袜|情趣内衣|黄色网站|情色|色情"
)
# 红线：心理健康自述 / 站内互撕 / 营销引流
_REDLINE = re.compile(
    r"抑郁|焦虑症|自杀|想死|不想活|撕逼|勃学|大V|公众号|私信我|关注我|求关注|"
    r"点赞评论|收藏转发"
)
# 敏感话题档（教培项目宁多勿漏）
_POLITICAL = re.compile(
    r"死刑|配枪|持枪|文革|六四|台湾|香港|共产党|国民党|政治犯|中南海|反腐|"
    r"枪杀|枪击|港独|台独"
)
# 自述锚点：个人经验/经历开场或主观表达（命中即弃）
_SELF_OPEN = re.compile(
    r"我现在|我今年|我去年|我也有|我自己|我身边|我认识|我表哥|我朋友|我同学|"
    r"我觉得|我认为|我个人|说来话长|本人|我本科|我硕士|我工作|我毕业"
)
# 答案中“我”的高频出现 = 个人叙事浓度过高（>170 字平均答案里 6 次以上）
_MAX_FIRST_PERSON = 6
# 知乎自指/平台腔（答案里出现说明是答主个人叙事或站内操作）
_CHATTY_A = re.compile(
    r"谢邀|泻药|利益相关|不请自来|先说结论|题主|楼主|本人男|本人女|再更新|"
    r"票圈|附上地址|视频链接|纯手打|哈哈哈哈哈|知乎|更新于|补充于|分割线|谢\\s*\S*\\s*邀"
)
# filter_qa 漏网的变形链接（如 httpv.youku.com 清洗后残留）
_URL_GARBAGE = re.compile(r"http|www\.|\.com|\.cn|\.org|youku|优酷|bilibili")


def parse_qa(line: str):
    """切分 '问题：... 回答：...' 单行。返回 (q, a) 或 (None, None)。"""
    m = re.match(r"^问题[:：](.*?)回答[:：](.*)$", line, re.S)
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()


def eligible(q: str, a: str) -> bool:
    if not q or not a:
        return False
    if not (10 <= len(q) <= 80) or not (40 <= len(a) <= 400):
        return False
    if not _KNOW_Q.search(q):
        return False
    if _OPINION_Q.search(q):
        return False
    if _NSFW.search(q) or _NSFW.search(a):
        return False
    if _REDLINE.search(q) or _REDLINE.search(a):
        return False
    if _POLITICAL.search(q) or _POLITICAL.search(a):
        return False
    if _SELF_OPEN.search(a):
        return False
    if a.count("我") >= _MAX_FIRST_PERSON:
        return False
    if _CHATTY_A.search(a):
        return False
    if _URL_GARBAGE.search(q) or _URL_GARBAGE.search(a):
        return False
    return True


def _load_rows(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    p = argparse.ArgumentParser(description="QA 语料 → SFT 指令提取")
    p.add_argument("--input", default="data/raw/qa_dedup.txt")
    p.add_argument("--output", default="data/nano/sft/qa_sft.jsonl",
                   help="输出 JSONL（{instruction, output}）")
    p.add_argument("--max", type=int, default=20000, help="抽取条数上限（默认按 --ratio 取候选池比例）")
    p.add_argument("--ratio", type=float, default=0.06, help="抽取比例（默认 6%≈2 万条）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--append-to", default=None,
                   help="把抽取结果追加合并到该 JSONL（自动备份 *.bak_preqa）")
    p.add_argument("--preview", action="store_true", help="预览模式：打印抽样前 5 条到 stderr")
    args = p.parse_args()

    hits = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            q, a = parse_qa(line.rstrip("\n"))
            if eligible(q or "", a or ""):
                hits.append((q, a))

    print(f"Candidates: {len(hits):,}", file=sys.stderr)
    rng = random.Random(args.seed)
    rng.shuffle(hits)
    n = args.max if args.max else int(len(hits) * args.ratio)
    picked = hits[:n]
    if args.preview:
        for q, a in picked[:5]:
            print(f"---\nQ: {q[:80]}\nA: {a[:200]}", file=sys.stderr)

    if args.append_to:
        backup = args.append_to.replace(".jsonl", ".bak_preqa.jsonl")
        if not os.path.exists(backup):
            existing = _load_rows(args.append_to)
            with open(backup, "w", encoding="utf-8") as fb:
                for r in existing:
                    fb.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Backup: {backup}")
        with open(args.append_to, "a", encoding="utf-8") as f:
            for q, a in picked:
                f.write(json.dumps({"instruction": q, "output": a}, ensure_ascii=False) + "\n")
        print(f"Appended {len(picked)} rows -> {args.append_to}")
        return

    with open(args.output, "w", encoding="utf-8") as f:
        for q, a in picked:
            f.write(json.dumps({"instruction": q, "output": a}, ensure_ascii=False) + "\n")
    print(f"Written: {args.output} ({len(picked)} rows)")


if __name__ == "__main__":
    main()
