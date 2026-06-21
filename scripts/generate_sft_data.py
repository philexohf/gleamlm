"""SFT 数据蒸馏脚本。
用 DeepSeek API 对 200 条种子问题生成高质量回答 + 同义变体 → 800-1000 条。

用法：
    set DEEPSEEK_API_KEY=sk-xxxx
    python scripts/generate_sft_data.py --output data/sft_data.jsonl

依赖：pip install openai
"""

import argparse
import json
import os
import re
import sys
import time

from openai import OpenAI

# ============================================================
# 200 种子问题（通用问答80 + 百科知识70 + 中文创作50）
# ============================================================

SEEDS = [
    # ========================
    # 通用问答 (80)
    # ========================
    # 自我介绍 / 能力边界
    "介绍一下你自己。",
    "你能做什么？不能做什么？",
    "你是谁创造的？",
    "你和一个人类有什么区别？",
    "你有什么优点和缺点？",
    # 日常对话
    "今天天气真好，适合做什么？",
    "推荐几道简单好做的家常菜。",
    "如何缓解工作压力？",
    "怎么培养一个好习惯？",
    "如何提高睡眠质量？",
    "怎样保持积极的心态？",
    "独处的时候可以做什么？",
    "如何高效利用碎片时间？",
    # 常识解释
    "什么是人工智能？",
    "什么是机器学习？和传统编程有什么区别？",
    "什么是大数据？",
    "什么是5G技术？",
    "什么是物联网？",
    "云计算是什么？通俗一点解释。",
    "区块链是什么？它的核心特点是什么？",
    "什么是元宇宙？",
    # 学习与教育
    "如何高效地学习一门新知识？",
    "记忆有什么技巧？",
    "读书有什么好处？",
    "怎么提高写作能力？",
    "为什么学英语很重要？",
    "如何培养孩子的阅读兴趣？",
    "大学生如何规划自己的职业生涯？",
    "终身学习为什么重要？",
    # 生活实用
    "如何做一道西红柿炒鸡蛋？",
    "第一次去北京旅游，有什么推荐？",
    "怎么做一锅好喝的鸡汤？",
    "如何挑选新鲜的水果？",
    "搬家时有哪些注意事项？",
    "如何写一封正式的邮件？",
    "面试时需要注意什么？",
    "租房要注意哪些问题？",
    # 人际关系
    "如何与性格不合的人相处？",
    "朋友之间产生矛盾怎么办？",
    "如何做一个好的倾听者？",
    "怎么拒绝别人又不伤感情？",
    "如何建立良好的职场人际关系？",
    "家人之间有分歧时怎么办？",
    # 社会话题
    "环境保护为什么重要？",
    "垃圾分类的意义是什么？",
    "为什么要提倡节约用水？",
    "公共交通和私家车各有什么优缺点？",
    "网络时代如何保护个人隐私？",
    "为什么说诚信很重要？",
    # 文化与艺术
    "中国书法有什么魅力？",
    "中国画和西方油画有什么区别？",
    "为什么说音乐是无国界的？",
    "电影和小说在叙事上有什么不同？",
    "什么样的照片算是一张好照片？",
    "茶文化在中国有什么地位？",
    # 哲学与思考
    "什么是幸福？",
    "人为什么要工作？",
    "时间为什么宝贵？",
    "什么是真正的自由？",
    "失败真的是一件坏事吗？",
    "善良和聪明的优先关系是什么？",
    # 趣味问答
    "如果时间旅行成为可能，你最想去哪个时代？",
    "如果有一天你变成一只猫，你会做什么？",
    "为什么猫咪这么喜欢纸箱？",
    "如果地球突然停止自转会怎样？",
    "人类为什么会有好奇心？",
    "为什么我们会做梦？",
    # 建议类
    "给二十岁的年轻人一些人生建议。",
    "如何应对人生中的不确定性？",
    "怎样找到自己真正热爱的事情？",
    "自律和自由的关系是什么？",
    "如何面对别人的批评？",
    "选择比努力更重要吗？",
    "养宠物有什么好处和坏处？",
    "如何快速适应一个新环境？",
    "什么是好的领导力？",
    "为什么团队合作很重要？",
    "如何看待成功和失败？",
    "你相信星座吗？为什么？",
    "怎样做一个受欢迎的人？",

    # ========================
    # 百科知识 (70)
    # ========================
    # 历史
    "唐朝是中国历史上哪个朝代？它有什么特点？",
    "秦始皇统一六国后做了哪些重要的事？",
    "第一次世界大战爆发的主要原因是什么？",
    "工业革命对世界产生了什么影响？",
    "二战结束后世界格局发生了怎样的变化？",
    "中国的四大发明是什么？它们如何影响了世界？",
    "丝绸之路有什么历史意义？",
    "文艺复兴是什么？它对欧洲有什么影响？",
    # 地理
    "地球上最大的海洋是哪个？",
    "珠穆朗玛峰有多高？位于哪个国家？",
    "长江和黄河各有什么特点？",
    "为什么会有四季变化？",
    "赤道附近的国家为什么比较热？",
    "地球上最深的海洋在哪里？有多深？",
    "中国的五个自治区分别是哪些？",
    "世界上最长的河流是什么？",
    # 科学
    "光合作用的基本过程是什么？",
    "DNA 和 RNA 有什么区别？",
    "为什么天空是蓝色的？",
    "黑洞是什么？它是怎么形成的？",
    "什么是相对论？简单解释一下。",
    "牛顿三大定律分别是什么？",
    "为什么物体会往下掉？",
    "电流是什么？它和电压有什么关系？",
    "基因编辑技术 CRISPR 是什么？",
    "为什么铁会生锈？",
    # 生物
    "人体有多少块骨头？",
    "恐龙为什么会灭绝？",
    "蜜蜂是怎么采蜜的？",
    "为什么说熊猫是国宝？",
    "人的大脑如何存储记忆？",
    "植物是怎么呼吸的？",
    "为什么有些动物会冬眠？",
    "人体的免疫系统是如何工作的？",
    # 天文
    "太阳系有哪八大行星？",
    "月亮为什么会有阴晴圆缺？",
    "什么是日食和月食？",
    "银河系是什么形状的？",
    "什么是暗物质？",
    "流星雨是怎样形成的？",
    # 文化
    "中国春节的由来是什么？有哪些习俗？",
    "端午节为什么要吃粽子和划龙舟？",
    "中秋节有什么传说和习俗？",
    "京剧的特点是什么？有哪些经典剧目？",
    "什么是太极？它有什么好处？",
    "中国有哪些世界文化遗产？举几个例子。",
    # 科技
    "互联网是怎么工作的？",
    "手机信号是怎么传输的？",
    "什么是半导体？它为什么重要？",
    "人造卫星有什么用途？",
    "核能发电的原理是什么？",
    "锂电池为什么能反复充电？",
    # 医学健康
    "人体正常体温是多少？",
    "感冒和流感有什么区别？",
    "维生素对人体有什么作用？",
    "为什么要打疫苗？",
    "什么是心理健康？如何保持？",
    "大熊猫为什么被称为活化石？",
    "潮汐是怎么形成的？",
    "地震是如何产生的？",
    "为什么海水是咸的？",
    "彩虹是怎么形成的？",
    "人类的祖先是谁？",
    "为什么要保护生物多样性？",
    "什么是碳中和？",
    # 经济
    "什么是通货膨胀？",
    "为什么会有贸易？",
    "股票和债券有什么区别？",
    "什么是GDP？它用来衡量什么？",

    # ========================
    # 中文创作 (50)
    # ========================
    # 诗歌
    "写一首关于春天的五言诗。",
    "以月亮为题写一首诗。",
    "写一首赞美母爱的短诗。",
    "以秋天的落叶为题写一首诗。",
    "写一首关于友谊的诗。",
    "以大海为题写一首诗。",
    "写一首关于雪的诗。",
    "以故乡为题写一首诗。",
    # 景物描写
    "请用一段话描述秋天北京的景色。",
    "描写一场夏日的暴雨。",
    "描述一个日落的场景。",
    "描写清晨的公园。",
    "描绘一片寂静的森林。",
    "描述一个热闹的菜市场。",
    "描写冬天的第一场雪。",
    "描述一条老街上黄昏时的景象。",
    # 故事续写
    "续写这个故事：从前有座山，山上有座庙……",
    "续写：一个少年在图书馆发现了一本神秘的书……",
    "续写：一个下雨的傍晚，有人在门口放了一把陌生的伞……",
    "续写：她打开了那扇从来没人打开过的门……",
    "续写：那封信上的邮戳是二十年前的日期……",
    # 短文
    "写一篇关于'家'的短文。",
    "写一篇关于'时间'的感悟文。",
    "写一篇关于'坚持'的短文，用一个小故事说明。",
    "以'我眼中的美'为题写一段文字。",
    "写一个关于勇气的小故事。",
    "用300字左右写一篇《我的老师》。",
    "写一篇关于成长感悟的短文。",
    # 应用文
    "写一封感谢信给帮助过你的人。",
    "写一封给远方朋友的信。",
    "写一段自我介绍，用于新环境的破冰。",
    "写一段给新同事的欢迎词。",
    "写一篇关于保护环境的倡议短文。",
    # 想象与创意
    "如果你能飞，你会飞去哪里？写一段描写。",
    "想象你是一朵云，写一段独白。",
    "假如人类可以生活在海底，世界会是什么样？",
    "写一个发生在未来一百年的小故事。",
    "用一个比喻来形容你理解的'时间'。",
    "如果动物能说话，你觉得它们会说什么？写一小段对话。",
    "描写一个夜晚星空下的场景。",
    "用一段话描述你心中的春天。",
    "写一篇关于友谊价值的短文。",
    "写一小段关于孤独的抒情文字。",
    "以'路'为题写一段富有哲理的话。",
    # 议论思考
    "谈谈你对'知识改变命运'的理解。",
    "简单谈一下科技发展对人际关系的影响。",
    "谈谈你对'书中自有黄金屋'的理解。",
    "简单论述一下读书的意义。",
    "谈谈你对传统文化传承的看法。",
    "聊聊现代人为什么越来越孤独。",
]

# 变体生成 Prompt 模板
VARIANT_PROMPT = """请为以下问题生成 {n} 个意思相近但表达不同的变体。
要求：
- 变体必须是中文
- 保持原问题的核心意图不变
- 可以改变句式、用词、角度
- 每条变体单独一行，不要编号

原问题：{seed}

{example_text}

请直接输出变体："""

VARIANT_EXAMPLE = """示例：
原问题：介绍一下你自己。

输出：
请做一个自我介绍。
你是谁？可以介绍一下吗？
能跟我聊聊你自己吗？
简单说说你的情况吧。"""


def get_client(api_key, base_url):
    """创建 OpenAI 兼容客户端"""
    return OpenAI(api_key=api_key, base_url=base_url)


def call_api(client, model, system_prompt, user_prompt,
             temperature=0.7, max_tokens=1024, max_retries=3):
    """调用 API，带重试"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                err_msg = str(e).lower()
                if "429" in err_msg or "rate" in err_msg:
                    time.sleep(5 * (2 ** attempt))
                else:
                    time.sleep(2 ** attempt)
            else:
                return None


def generate_variants(client, model, seed, n_variants=4):
    """为一个种子生成 n 个同义变体"""
    prompt = VARIANT_PROMPT.format(
        n=n_variants,
        seed=seed,
        example_text=VARIANT_EXAMPLE,
    )
    result = call_api(
        client, model,
        system_prompt="你是一个中文语言专家，擅长改写问题。",
        user_prompt=prompt,
        temperature=0.8,
        max_tokens=512,
    )
    if result is None:
        return []
    variants = [v.strip() for v in result.split("\n") if v.strip()]
    # 去重、过滤与原问题完全相同的内容
    unique = []
    for v in variants:
        v = re.sub(r'^[\d\s.\-、)）*#]+\s*', '', v).strip()
        if v and v != seed and len(v) >= 5 and v not in unique:
            unique.append(v)
    return unique[:n_variants]


def generate_answer(client, model, instruction):
    """用 DeepSeek 生成高质量回答"""
    return call_api(
        client, model,
        system_prompt=(
            "你是XFIND-LLM，一个面向教育和研究的轻量级开源对话模型（约39M参数），"
            "由个人开发者基于PyTorch从零实现，参考了LLaMA3和Qwen3架构。"
            "请用中文回答问题。"
            "回答要准确、简洁、有条理，长度适中。"
            "注意：不要提及任何具体公司、产品名称或模型来源，"
            "不要说你来自哪个公司或由谁开发。像一个通用的AI助手一样回答。"
        ),
        user_prompt=instruction,
        temperature=0.7,
        max_tokens=1024,
    )


def main():
    parser = argparse.ArgumentParser(description="SFT 数据蒸馏")
    parser.add_argument("--output", type=str, default="data/sft_data.jsonl",
                        help="输出 JSONL 路径")
    parser.add_argument("--api_key", type=str, default=None,
                        help="DeepSeek API Key（默认从环境变量 DEEPSEEK_API_KEY 读取）")
    parser.add_argument("--base_url", type=str,
                        default="https://api.deepseek.com",
                        help="API Base URL")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                        help="模型名称")
    parser.add_argument("--variants_per_seed", type=int, default=4,
                        help="每个种子生成的变体数（目标：200×4=800）")
    parser.add_argument("--skip_variants", action="store_true",
                        help="跳过变体生成，只对种子生成答案")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="API 调用间隔秒数")
    parser.add_argument("--dry_run", type=int, default=0,
                        help="只运行前 N 条（用于测试）")
    args = parser.parse_args()

    # API Key
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: 请设置 DEEPSEEK_API_KEY 环境变量或通过 --api_key 传入")
        sys.exit(1)

    client = get_client(api_key, args.base_url)
    print(f"API: {args.base_url}, model: {args.model}")

    # 准备种子（dry_run 截断）
    seeds = SEEDS[:args.dry_run] if args.dry_run > 0 else SEEDS
    print(f"种子问题数: {len(seeds)}")

    # Step 1: 变体生成（可选跳过）
    all_instructions = []
    if args.skip_variants:
        all_instructions = list(seeds)
        print("跳过变体生成，仅使用种子问题")
    else:
        print(f"\n=== Step 1: 生成变体（每个种子 {args.variants_per_seed} 个）===")
        for i, seed in enumerate(seeds):
            print(f"[{i+1}/{len(seeds)}] {seed[:40]}...", end=" ", flush=True)
            variants = generate_variants(client, args.model, seed, args.variants_per_seed)
            all_instructions.append(seed)
            all_instructions.extend(variants)
            print(f"→ {1 + len(variants)} 条")
            time.sleep(args.delay)

    print(f"\n总共将生成 {len(all_instructions)} 条问答")

    # Step 2: 生成回答
    print(f"\n=== Step 2: 生成高质量回答 ===")
    results = []
    fail_count = 0

    for i, instruction in enumerate(all_instructions):
        print(f"[{i+1}/{len(all_instructions)}] {instruction[:50]}...", end=" ", flush=True)
        answer = generate_answer(client, args.model, instruction)
        if answer:
            results.append({"instruction": instruction, "output": answer})
            print("OK")
        else:
            fail_count += 1
            print("FAIL")

        # 每 50 条保存一次（防止中断丢失）
        if (i + 1) % 50 == 0 and results:
            save_partial(results, args.output + ".partial")
            print(f"  --- Partial save: {len(results)} entries ---")

        time.sleep(args.delay)

    # Step 3: 最终保存
    if results:
        save_final(results, args.output)
        print(f"\n完成！共生成 {len(results)} 条数据")
        print(f"失败: {fail_count} 条")
        print(f"输出: {os.path.abspath(args.output)}")

        # 统计类别分布
        print_distribution(results)
    else:
        print("\nError: 没有成功生成任何数据")
        sys.exit(1)


def save_partial(results, path):
    """部分保存"""
    with open(path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_final(results, path):
    """最终保存为 JSONL + 备份"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # 备份旧文件
    if os.path.exists(path):
        backup = path + ".bak"
        os.rename(path, backup)
        print(f"Backed up old file to: {backup}")
    save_partial(results, path)
    # 清除 partial
    partial = path + ".partial"
    if os.path.exists(partial):
        os.remove(partial)


def print_distribution(results):
    """统计并打印类别分布"""
    creation_kw = ["诗", "描写", "描述", "短文", "故事", "续写", "信", "想象",
                   "比喻", "独白", "议论", "谈谈", "论述", "倡议"]
    general_kw = ["介绍", "你", "能做什么", "优点", "缺点", "天气", "推荐", "缓解", "习惯",
                  "睡眠", "心态", "独处", "碎片", "人工智能", "机器学习", "大数据",
                  "5G", "物联网", "云计算", "区块链", "元宇宙", "学习", "记忆",
                  "读书", "写作", "英语", "孩子", "大学", "终身", "面试", "租房",
                  "邮件", "朋友", "拒绝", "家庭", "工作压力", "健身", "时间",
                  "幸福", "自由", "失败", "善良", "猫", "人生"]

    general = 0
    creation = 0
    knowledge = 0
    for r in results:
        if any(kw in r["instruction"] for kw in creation_kw):
            creation += 1
        elif any(kw in r["instruction"] for kw in general_kw):
            general += 1
        else:
            knowledge += 1

    print(f"\n类别分布（近似）:")
    print(f"  通用问答: {general}")
    print(f"  百科知识: {knowledge}")
    print(f"  中文创作: {creation}")


if __name__ == "__main__":
    main()
