"""清理 SFT 数据中的 A:/B:/C: 字母前缀（DeepSeek API 格式化污染）"""
import json
import re
import os

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sft_data.jsonl")
out_path = os.path.join(os.path.dirname(__file__), "..", "data", "sft_data_clean.jsonl")

cleaned = 0
total = 0
with open(data_path, 'r', encoding='utf-8') as f_in:
    with open(out_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            total += 1
            item = json.loads(line)
            old = item['output']
            # 去掉 A: B: C: D: E: F: 等字母前缀（带空格直接跟汉字或英文）
            new = re.sub(r'(?<![a-zA-Z])[A-F]:\s?(?=[\u4e00-\u9fffA-Z])', '', old)
            # 去掉连续空格
            new = re.sub(r' {2,}', ' ', new).strip()
            if new != old:
                cleaned += 1
            item['output'] = new
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Cleaned {cleaned}/{total} entries -> {out_path}")
