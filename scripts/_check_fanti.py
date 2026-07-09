"""检查真正的繁简异体字（简/繁体形式不同的字）"""

# 繁简异体对照 - 只包含形体不同的字
fanti_to_jianti = {
    "國": "国",
    "體": "体",
    "時": "时",
    "對": "对",
    "發": "发",
    "學": "学",
    "關": "关",
    "係": "系",
    "無": "无",
    "後": "后",
    "會": "会",
    "來": "来",
    "裏": "里",
    "麼": "么",
    "爲": "为",
    "衛": "卫",
    "軍": "军",
    "際": "际",
    "動": "动",
    "進": "进",
    "門": "门",
    "開": "开",
    "東": "东",
    "車": "车",
    "長": "长",
    "間": "间",
    "過": "过",
    "個": "个",
    "實": "实",
    "從": "从",
    "現": "现",
    "當": "当",
    "還": "还",
    "問": "问",
    "題": "题",
    "點": "点",
    "業": "业",
    "總": "总",
    "設": "设",
    "與": "与",
    "並": "并",
    "產": "产",
    "於": "于",
    "層": "层",
    "戰": "战",
    "數": "数",
    "區": "区",
    "處": "处",
    "萬": "万",
    "書": "书",
    "兒": "儿",
    "頭": "头",
    "電": "电",
    "網": "网",
    "馬": "马",
    "縣": "县",
    "報": "报",
    "愛": "爱",
    "聲": "声",
    "聽": "听",
    "證": "证",
    "飛": "飞",
    "風": "风",
    "氣": "气",
    "連": "连",
    "遠": "远",
    "華": "华",
    "語": "语",
    "說": "说",
    "員": "员",
    "義": "义",
    "龍": "龙",
    "亞": "亚",
    "號": "号",
    "場": "场",
    "圖": "图",
    "達": "达",
    "變": "变",
    "見": "见",
    "樣": "样",
    "代": "代",
    "曆": "历",
    "險": "险",
    "轉": "转",
    "響": "响",
    "辦": "办",
    "爭": "争",
    "節": "节",
    "機": "机",
    "漢": "汉",
    "殺": "杀",
    "勞": "劳",
    "單": "单",
    "難": "难",
    "買": "买",
    "賣": "卖",
    "養": "养",
    "鐵": "铁",
    "歐": "欧",
    "賽": "赛",
    "輪": "轮",
    "質": "质",
    "權": "权",
    "醫": "医",
    "標": "标",
    "術": "术",
}
fanti_chars = set(fanti_to_jianti.keys())

files = {
    "wiki": "data/raw/wiki_clean.txt",
    "news": "data/raw/news_clean.txt",
    "baike": "data/raw/baike_clean.txt",
    "qa": "data/raw/qa_clean.txt",
}
all_clean = True
for name, path in files.items():
    with open(path, encoding="utf-8") as f:
        text = f.read(2000000)
    found_chars = set()
    found_positions = []
    for i, c in enumerate(text):
        if c in fanti_chars:
            found_chars.add(c)
            if len(found_positions) < 5:
                ctx = text[max(0, i - 8) : i + 8]
                found_positions.append((c, fanti_to_jianti.get(c, "?"), ctx))
    if found_chars:
        print(f"{name}: {len(found_chars)} types of traditional chars found:")
        for c, j, ctx in found_positions:
            print(f'  {c} -> {j}  "...{ctx}..."')
        all_clean = False
    else:
        print(f"{name}: clean, no traditional chars")
if all_clean:
    print("\nAll 4 sources pure simplified. zhconv worked correctly.")
