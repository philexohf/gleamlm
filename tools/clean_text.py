"""文本清洗。去 HTML、URL、空白，过滤短/纯符号行，可选简繁转换、中文占比过滤、广告过滤"""

import re
import os
import argparse

# 简繁转换
try:
    import zhconv
    HAS_ZhCONV = True
except ImportError:
    HAS_ZhCONV = False
    print("提示: pip install zhconv 可启用简繁转换")

# 广告/垃圾文本特征
AD_PATTERNS = [
    re.compile(r'咨询.*[热热线电].*[：:]?\s*\d{3,}'),    # 咨询热线/电话
    re.compile(r'(活动|加盟|招商|订[购车]).*[热热线电].*[：:]?\s*\d{3,}'),  # 活动热线/加盟电话
    re.compile(r'[热线电话][：:]\s*\d{3,}'),               # 热线/电话：xxx
    re.compile(r'[Qq]{2}[：:]\s*\d{5,}'),                 # QQ号
    re.compile(r'(微信|加微信|V信|vx)[：:：]\s*\S+'),       # 微信推广
    re.compile(r'(扫码|扫一扫|关注公众号|添加客服)'),        # 扫码引流
    re.compile(r'(点击.*链接|立即.*下载|限时.*抢购|免费.*领取)'), # 营销话术
    re.compile(r'\d{3,}[-—]\d{3,}[-—]\d{3,}'),            # 电话号码
    re.compile(r'(直营店|加盟店|连锁店|分店).*(覆盖|遍布|全国)'),  # 连锁加盟软文
    re.compile(r'(特价|优惠|折扣|促销|限时|团购).*(活动|进行|开启)'), # 促销
    re.compile(r'(史上最低|年终大促|亏本甩卖|跳楼价)'),      # 极端营销词
    re.compile(r'(名额有限|先到先得|抢购|火爆.*中)'),        # 饥饿营销
]

# Wiki 无价值模板（美国小镇人口普查/坐标/非建制地区）
WIKI_JUNK_PATTERNS = [
    re.compile(r'镇区人口有'),                        # 美国人口普查模板
    re.compile(r'涵盖总面积为'),                      # 同上
    re.compile(r'(美国|United States).*人口普查'),    # 人口普查描述
    re.compile(r'座标为'),                            # 纯坐标条目
    re.compile(r'非建制地区'),                        # 无价值地理条目
    re.compile(r'海拔高度为.*(米|英尺)'),             # 纯地理数据
]


def clean_text(text, min_len=10, max_len=2000, convert_zh=False, min_zh_ratio=0.0,
               filter_ads=False, filter_wiki_junk=False):
    """
    清洗单条文本

    清洗规则：
        - 去除 HTML 标签
        - 去除多余空白
        - 过滤过短/过长文本
        - 过滤纯数字/符号行
        - 可选：中文占比不足（针对 Wiki 人口普查等非中文内容）
        - 可选：广告/营销软文过滤
        - 统一标点符号
        - 简繁转换（可选）
    """
    if not text or not text.strip():
        return None

    # 简繁转换（需 pip install zhconv）
    if convert_zh and HAS_ZhCONV:
        text = zhconv.convert(text, 'zh-cn')

    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)

    # 去除 URL
    text = re.sub(r'https?://\S+', '', text)

    # 统一空白字符
    text = re.sub(r'\s+', ' ', text).strip()

    # 过滤长度
    if len(text) < min_len or len(text) > max_len:
        return None

    # 中文占比过滤（针对 Wiki 等混入大量英文/拉丁字符的内容）
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    if min_zh_ratio > 0 and len(text) > 0:
        if chinese_chars / len(text) < min_zh_ratio:
            return None

    # 过滤纯数字/符号行（中文/英文占比过低）
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    if chinese_chars + english_chars < len(text) * 0.3:
        return None

    # 广告过滤
    if filter_ads:
        for pattern in AD_PATTERNS:
            if pattern.search(text):
                return None

    # Wiki 垃圾过滤（美国小镇人口普查等）
    if filter_wiki_junk:
        for pattern in WIKI_JUNK_PATTERNS:
            if pattern.search(text):
                return None

    return text


def clean_file(input_path, output_path, min_len=10, max_len=2000, convert_zh=False,
               min_zh_ratio=0.0, filter_ads=False, filter_wiki_junk=False):
    """
    清洗整个文本文件
    """
    total = 0
    kept = 0

    if convert_zh and not HAS_ZhCONV:
        print("WARNING: zhconv 未安装，简繁转换已跳过。pip install zhconv")

    print(f"Cleaning: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as fin:
        with open(output_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                total += 1
                cleaned = clean_text(line, min_len, max_len, convert_zh, min_zh_ratio, filter_ads, filter_wiki_junk)
                if cleaned:
                    fout.write(cleaned + '\n')
                    kept += 1

                if total % 100000 == 0:
                    print(f"  Processed {total} lines, kept {kept} ({100*kept/max(1,total):.1f}%)")

    print(f"Done: {total} lines processed, {kept} kept ({100*kept/max(1,total):.1f}%)")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='文本清洗工具')
    parser.add_argument('--input', type=str, required=True, help='输入文件')
    parser.add_argument('--output', type=str, required=True, help='输出文件')
    parser.add_argument('--min_len', type=int, default=10, help='最小文本长度')
    parser.add_argument('--max_len', type=int, default=2000, help='最大文本长度')
    parser.add_argument('--convert_zh', action='store_true', default=False,
                        help='繁体转简体 (需 pip install zhconv)')
    parser.add_argument('--min_zh_ratio', type=float, default=0.0,
                        help='最小中文占比 (0.0=关闭, 建议 Wiki 设 0.15)')
    parser.add_argument('--filter_ads', action='store_true', default=False,
                        help='过滤广告/软文')
    parser.add_argument('--filter_wiki_junk', action='store_true', default=False,
                        help='过滤 Wiki 无价值模板（人口普查/坐标/非建制地区）')
    args = parser.parse_args()

    clean_file(args.input, args.output, args.min_len, args.max_len, args.convert_zh,
               args.min_zh_ratio, args.filter_ads, args.filter_wiki_junk)


if __name__ == '__main__':
    main()
