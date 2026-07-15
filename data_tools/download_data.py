"""
多源数据下载脚本

下载三个数据源并提取为纯文本（每行一篇文档）：
1. news2016zh - 中文新闻（250万篇，~0.35B tokens）
2. BaiduBaike - 百度百科（563万条，~0.25B tokens）
3. webtext2019zh - 社区问答（410万条，~0.14B tokens）

数据来源：
- nlp_chinese_corpus (brightmart/nlp_chinese_corpus)
- baby-llama2-chinese (DLLXW/baby-llama2-chinese)
- Kaggle (webtext2019zh)

下载方式：优先 kagglehub 自动下载，失败则打印手动搜索指引。

输出目录：
    data/raw/news_raw.txt
    data/raw/baike_raw.txt
    data/raw/qa_raw.txt

用法：
    python data_tools/download_data.py              # 下载全部
    python data_tools/download_data.py --source news   # 只下新闻
    python data_tools/download_data.py --source baike  # 只下百科
    python data_tools/download_data.py --source qa     # 只下问答
"""

import argparse
import gzip
import json
import os
import shutil
import subprocess

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(RAW_DIR, exist_ok=True)


def download_news(output_txt):
    """下载 news2016zh 新闻数据集，提取 title+desc+content"""
    print("\n" + "=" * 60)
    print("1/3  中文新闻 (news2016zh)")
    print("=" * 60)

    if os.path.exists(output_txt):
        print(f"  新闻文本已存在: {output_txt}")
        return True

    archive_file = os.path.join(RAW_DIR, "news2016zh.json.gz")

    # 兼容 zip 格式
    zip_file = os.path.join(RAW_DIR, "new2016zh.zip")
    if os.path.exists(zip_file):
        print("  检测到 zip 格式，解压中...")
        import zipfile

        with zipfile.ZipFile(zip_file, "r") as zf:
            namelist = zf.namelist()
            json_files = [n for n in namelist if n.endswith(".json")]
            if json_files:
                train_files = [n for n in json_files if "train" in n.lower()]
                target = (
                    train_files[0]
                    if train_files
                    else max(json_files, key=lambda n: zf.getinfo(n).file_size)
                )
                print(f"  解压: {target}")
                zf.extract(target, RAW_DIR)
                extracted = os.path.join(RAW_DIR, target)
                if extracted.endswith(".json"):
                    print("  压缩为 gz...")
                    with open(extracted, "rb") as fin, gzip.open(archive_file, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                    os.remove(extracted)
            else:
                print("  zip 内未找到 JSON 文件")
                return False
        os.remove(zip_file)
        print(f"  解压完成: {archive_file}")
        return _extract_news(archive_file, output_txt)

    # 该数据集版权归各新闻媒体所有，不提供直链
    print("\n  === 手动获取指引 ===")
    print("  请自行搜索 news2016zh 数据集（nlp_chinese_corpus）")
    print(f"  下载后将 news2016zh.json.gz 放到: {RAW_DIR}")
    print("  然后重新运行: python data_tools/download_data.py --source news")
    return False


def _extract_news(archive_file, output_txt):
    """从 news2016zh.json.gz 提取纯文本"""
    print("  提取新闻文本...")
    count = 0
    with (
        gzip.open(archive_file, "rt", encoding="utf-8") as fin,
        open(output_txt, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            try:
                item = json.loads(line.strip())
                title = item.get("title", "") or ""
                desc = item.get("desc", "") or ""
                content = item.get("content", "") or ""
                # 合并: 标题 + 描述 + 正文
                parts = [p.strip() for p in [title, desc, content] if p.strip()]
                full_text = "。".join(parts) if parts else ""
                if len(full_text) > 30:
                    fout.write(full_text + "\n")
                    count += 1
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

    print(f"  新闻文本提取完成: {count} 篇, 保存到 {output_txt}")
    return True


def download_baike(output_txt):
    """下载百度百科数据集，563万词条，提取 title+summary+text"""
    print("\n" + "=" * 60)
    print("2/3  百度百科 (baby-llama2-chinese)")
    print("=" * 60)

    if os.path.exists(output_txt):
        print(f"  百科文本已存在: {output_txt}")
        return True

    archive_file = os.path.join(RAW_DIR, "563w_baidubaike.json.7z")
    json_file = os.path.join(RAW_DIR, "563w_baidubaike.json")

    # 该数据集原始文件已被作者从 GitHub 删除，目前仅百度网盘可用
    # Google Drive 无此文件

    if not os.path.exists(json_file) and not os.path.exists(archive_file):
        print("  === 手动获取指引 ===")
        print("  请自行搜索 baidubaike 563w 数据集（baby-llama2-chinese）")
        print(f"  下载后放到: {RAW_DIR}")
        print("  然后重新运行: python data_tools/download_data.py --source baike")
        return False

    # 解压 7z
    if not os.path.exists(json_file):
        print("  解压 7z 文件...")
        try:
            import py7zr

            with py7zr.SevenZipFile(archive_file, "r") as archive:
                archive.extractall(RAW_DIR)
            print(f"  解压完成: {json_file}")
        except ImportError:
            try:
                subprocess.run(["7z", "x", archive_file, f"-o{RAW_DIR}"], check=True)
                print(f"  解压完成: {json_file}")
            except Exception:
                print("  请安装 py7zr: pip install py7zr")
                print(f"  或手动解压 {archive_file} 到 {RAW_DIR}")
                return False

    return _extract_baike(json_file, output_txt)


def _extract_baike(json_file, output_txt):
    """从 563w_baidubaike.json 提取纯文本"""
    print("  提取百科文本（title + summary + text）...")

    # 读取 JSON（整个文件是一个大的 JSON 对象或 JSONL）
    with open(json_file, encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            data = json.load(f)
            items = data if isinstance(data, list) else [data]
        else:
            # JSONL 格式
            items = []
            for line in f:
                try:
                    items.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    count = 0
    with open(output_txt, "w", encoding="utf-8") as fout:
        for item in items:
            title = item.get("title", "") or ""
            summary = item.get("summary", "") or ""
            text = item.get("text", "") or ""
            parts = [p.strip() for p in [title, summary, text] if p.strip()]
            full_text = "。".join(parts) if parts else ""
            full_text = full_text.replace("。。", "。")
            if len(full_text) > 20:
                fout.write(full_text + "\n")
                count += 1

    print(f"  百科文本提取完成: {count} 条, 保存到 {output_txt}")
    return True


def download_qa(output_txt):
    """下载 webtext2019zh 社区问答，410万条，提取 title+content"""
    print("\n" + "=" * 60)
    print("3/3  社区问答 (webtext2019zh)")
    print("=" * 60)

    if os.path.exists(output_txt):
        print(f"  问答文本已存在: {output_txt}")
        return True

    archive_file = os.path.join(RAW_DIR, "webtext2019zh.json.gz")

    # 兼容 zip 格式（从 data/ 根目录移到 raw/）
    zip_file = os.path.join(DATA_DIR, "webtext2019zh.zip")
    if os.path.exists(zip_file) and not os.path.exists(archive_file):
        print("  [webtext2019zh] 检测到 zip 格式，解压中...")
        import zipfile

        os.makedirs(RAW_DIR, exist_ok=True)
        with zipfile.ZipFile(zip_file, "r") as zf:
            namelist = zf.namelist()
            json_files = [n for n in namelist if n.endswith(".json")]
            if json_files:
                # 优先 train，否则取最大的
                train_files = [n for n in json_files if "train" in n.lower()]
                target = (
                    train_files[0]
                    if train_files
                    else max(json_files, key=lambda n: zf.getinfo(n).file_size)
                )
                print(f"  解压: {target}")
                zf.extract(target, RAW_DIR)
                # 如果解压出来的是 .json，gzip 一下统一格式
                extracted = os.path.join(RAW_DIR, target)
                if extracted.endswith(".json"):
                    print("  压缩中...")
                    with open(extracted, "rb") as fin, gzip.open(archive_file, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                    os.remove(extracted)
                    print(f"  压缩完成: {archive_file}")
            else:
                print("  zip 内未找到 JSON 文件，尝试全部解压...")
                zf.extractall(RAW_DIR)
                for n in namelist:
                    if n.endswith(".json"):
                        extracted = os.path.join(RAW_DIR, n)
                        with open(extracted, "rb") as fin, gzip.open(archive_file, "wb") as fout:
                            shutil.copyfileobj(fin, fout)
                        os.remove(extracted)
                        break
        os.remove(zip_file)  # 清理 zip
        print(f"  解压完成: {archive_file}")

    # 方法1: kagglehub（国内可直连）
    if not os.path.exists(archive_file):
        try:
            import kagglehub

            print("  [webtext2019zh] Kaggle 下载中...")
            kaggle_path = kagglehub.dataset_download("terrychanorg/webtext2019zhjsonwebtext2019zh")
            # 查找 gz 或 json 文件，复制到 raw 目录
            for f in os.listdir(kaggle_path):
                if f.endswith(".gz") or f.endswith(".json"):
                    src = os.path.join(kaggle_path, f)
                    if f.endswith(".gz"):
                        shutil.copy2(src, archive_file)
                        print(f"  [webtext2019zh] 下载完成: {archive_file}")
                        break
                    elif f.endswith(".json") and not f.startswith("web_text_zh_test"):
                        # 非 test 文件，可能是 train，gzip 后复制
                        with open(src, "rb") as fin, gzip.open(archive_file, "wb") as fout:
                            shutil.copyfileobj(fin, fout)
                        print(f"  [webtext2019zh] 下载完成 (gzip'd): {archive_file}")
                        break
            else:
                # 没找到合适文件，直接拿第一个大的 json
                for f in sorted(os.listdir(kaggle_path)):
                    if f.endswith(".json"):
                        src = os.path.join(kaggle_path, f)
                        with open(src, "rb") as fin, gzip.open(archive_file, "wb") as fout:
                            shutil.copyfileobj(fin, fout)
                        print(f"  [webtext2019zh] 下载完成 (fallback, gzip'd): {archive_file}")
                        break
        except Exception as e:
            print(f"  [webtext2019zh] Kaggle 下载失败: {e}")

    if not os.path.exists(archive_file):
        print("\n  === 手动获取指引 ===")
        print(
            "  Kaggle: https://www.kaggle.com/datasets/terrychanorg/webtext2019zhjsonwebtext2019zh"
        )
        print(f"  下载后将 webtext2019zh.json.gz 放到: {RAW_DIR}")
        print("  然后重新运行: python data_tools/download_data.py --source qa")
        print()
        print("  备选数据源 - baike2018qa (150万百科问答):")
        print(f"  请自行搜索下载，放到 {RAW_DIR}")
        print("  然后用 --source qa_alt 运行本脚本")
        return False

    return _extract_qa(archive_file, output_txt)


def _extract_qa(archive_file, output_txt):
    """从 webtext2019zh 提取纯文本（支持 JSONL 和 JSON 数组两种格式）"""
    print("  提取问答文本（title + content）...")

    # 先尝试全文加载（可能是 {"root": [...]} 或纯数组格式）
    with gzip.open(archive_file, "rt", encoding="utf-8") as fin:
        raw = fin.read(1024)
    is_jsonl = raw.strip().startswith("{") and "qid" in raw[:200]

    items = []
    if not is_jsonl:
        # 尝试整体 JSON 解析
        print("  检测为 JSON 数组格式，整体加载...")
        with gzip.open(archive_file, "rt", encoding="utf-8") as fin:
            data = json.load(fin)
        if isinstance(data, dict) and "root" in data:
            items = data["root"]
        elif isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # 可能是带顶层 key 的对象，取最大的列表字段
            for v in data.values():
                if isinstance(v, list) and len(v) > len(items):
                    items = v
    else:
        # JSONL 逐行解析
        print("  检测为 JSONL 格式，逐行解析...")
        with gzip.open(archive_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                try:
                    items.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    count = 0
    with open(output_txt, "w", encoding="utf-8") as fout:
        for item in items:
            title = item.get("title", "") or ""
            content = item.get("content", "") or ""
            # 过滤低质量: star=0 且内容太短跳过
            star = item.get("star", 1)
            if star == 0 and len(content) < 20:
                continue
            if len(title) > 3 and len(content) > 10:
                # 合并问题+回答，保留多样句式
                full_text = f"问题：{title} 回答：{content}"
                fout.write(full_text + "\n")
                count += 1

    print(f"  问答文本提取完成: {count} 条, 保存到 {output_txt}")
    return True


def main():
    parser = argparse.ArgumentParser(description="多源数据下载与提取")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "news", "baike", "qa"],
        help="下载指定数据源",
    )
    args = parser.parse_args()

    sources = {
        "news": (download_news, os.path.join(RAW_DIR, "news_raw.txt")),
        "baike": (download_baike, os.path.join(RAW_DIR, "baike_raw.txt")),
        "qa": (download_qa, os.path.join(RAW_DIR, "qa_raw.txt")),
    }

    if args.source == "all":
        for key in ["news", "baike", "qa"]:
            func, outpath = sources[key]
            func(outpath)
    else:
        func, outpath = sources[args.source]
        func(outpath)

    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"输出目录: {RAW_DIR}")
    print()
    print("后续步骤：")
    print("  # 一键管道（清洗 → 去重 → 混合 → 切分）")
    print("  python data_tools/prepare_data.py")


if __name__ == "__main__":
    main()
