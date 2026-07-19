"""文本清洗。去 HTML、URL、空白，过滤短/纯符号行，可选简繁转换、中文占比过滤、广告过滤"""

from __future__ import annotations

import argparse
import re

try:
    import zhconv

    HAS_ZhCONV = True
except ImportError:
    HAS_ZhCONV = False
    print("提示: pip install zhconv 可启用简繁转换")

_HTML_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_URL_RE = re.compile(r"https?://\S+")
_SPACE_RE = re.compile(r"\s+")
_ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_CHAR_RE = re.compile(r"[a-zA-Z]")

AD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"咨询.*[热热线电].*[：:]?\s*\d{3,}"),
    re.compile(r"(活动|加盟|招商|订[购车]).*[热热线电].*[：:]?\s*\d{3,}"),
    re.compile(r"[热线电话][：:]\s*\d{3,}"),
    re.compile(r"[Qq]{2}[：:]\s*\d{5,}"),
    re.compile(r"(微信|加微信|V信|vx)[：:：]\s*\S+"),
    re.compile(r"(扫码|扫一扫|关注公众号|添加客服)"),
    re.compile(r"(点击.*链接|立即.*下载|限时.*抢购|免费.*领取)"),
    re.compile(r"\d{3,}[-—]\d{3,}[-—]\d{3,}"),
    re.compile(r"(直营店|加盟店|连锁店|分店).*(覆盖|遍布|全国)"),
    re.compile(r"(特价|优惠|折扣|促销|限时|团购).*(活动|进行|开启)"),
    re.compile(r"(史上最低|年终大促|亏本甩卖|跳楼价)"),
    re.compile(r"(名额有限|先到先得|抢购|火爆.*中)"),
]

WIKI_JUNK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"镇区人口有"),
    re.compile(r"涵盖总面积为"),
    re.compile(r"(美国|United States).*人口普查"),
    re.compile(r"座标为"),
    re.compile(r"非建制地区"),
    re.compile(r"海拔高度为.*(米|英尺)"),
]


def clean_text(
    text: str,
    min_len: int = 10,
    max_len: int = 2000,
    convert_zh: bool = False,
    min_zh_ratio: float = 0.0,
    filter_ads: bool = False,
    filter_wiki_junk: bool = False,
) -> str | None:
    if not text or not text.strip():
        return None

    if convert_zh and HAS_ZhCONV:
        text = zhconv.convert(text, "zh-cn")

    text = _HTML_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text).strip()

    if len(text) < min_len or len(text) > max_len:
        return None

    chinese_chars = len(_ZH_CHAR_RE.findall(text))
    if min_zh_ratio > 0 and len(text) > 0 and chinese_chars / len(text) < min_zh_ratio:
        return None

    english_chars = len(_EN_CHAR_RE.findall(text))
    if chinese_chars + english_chars < len(text) * 0.3:
        return None

    if filter_ads:
        for pattern in AD_PATTERNS:
            if pattern.search(text):
                return None

    if filter_wiki_junk:
        for pattern in WIKI_JUNK_PATTERNS:
            if pattern.search(text):
                return None

    return text


def clean_file(
    input_path: str,
    output_path: str,
    min_len: int = 10,
    max_len: int = 2000,
    convert_zh: bool = False,
    min_zh_ratio: float = 0.0,
    filter_ads: bool = False,
    filter_wiki_junk: bool = False,
) -> None:
    total = 0
    kept = 0

    if convert_zh and not HAS_ZhCONV:
        print("WARNING: zhconv 未安装，简繁转换已跳过。pip install zhconv")

    print(f"Cleaning: {input_path}")

    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            total += 1
            cleaned = clean_text(
                line, min_len, max_len, convert_zh, min_zh_ratio, filter_ads, filter_wiki_junk
            )
            if cleaned:
                fout.write(cleaned + "\n")
                kept += 1

            if total % 100000 == 0:
                print(
                    f"  Processed {total} lines, kept {kept} ({100 * kept / max(1, total):.1f}%)",
                    flush=True,
                )

    print(f"Done: {total} lines processed, {kept} kept ({100 * kept / max(1, total):.1f}%)")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="文本清洗工具")
    parser.add_argument("--input", type=str, required=True, help="输入文件")
    parser.add_argument("--output", type=str, required=True, help="输出文件")
    parser.add_argument("--min_len", type=int, default=10, help="最小文本长度")
    parser.add_argument("--max_len", type=int, default=2000, help="最大文本长度")
    parser.add_argument(
        "--convert_zh",
        action="store_true",
        default=False,
        help="繁体转简体 (需 pip install zhconv)",
    )
    parser.add_argument(
        "--min_zh_ratio", type=float, default=0.0, help="最小中文占比 (0.0=关闭, 建议 Wiki 设 0.15)"
    )
    parser.add_argument("--filter_ads", action="store_true", default=False, help="过滤广告/软文")
    parser.add_argument(
        "--filter_wiki_junk",
        action="store_true",
        default=False,
        help="过滤 Wiki 无价值模板（人口普查/坐标/非建制地区）",
    )
    args = parser.parse_args()

    clean_file(
        args.input,
        args.output,
        args.min_len,
        args.max_len,
        args.convert_zh,
        args.min_zh_ratio,
        args.filter_ads,
        args.filter_wiki_junk,
    )


if __name__ == "__main__":
    main()
