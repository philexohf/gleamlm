# XFIND-LLM —— 面向教育和科研的自研语言模型

<img src="./assets/Xfind-logo.png" style="zoom: 25%;" />

> **从零手写、每行可讲的教学级 LLM 平台**——纯 PyTorch 实现，零 HuggingFace 依赖，XFIND-Nano 双平台兼容、XFIND-Lite 科研进阶、XFIND-Pro 前沿探索。降低 LLM 学习与研究的门槛。

## 核心教学优势

| # | 优势 | 对比其他开源项目 |
|---|------|------------------|
| 1 | **零 HuggingFace 依赖** — 不继承 `PreTrainedModel`，所有 LLM 组件从零手写 | 多数项目依赖 HuggingFace `transformers` 生态 |
| 2 | **单文件可读完** — 模型定义 ~300 行，训练脚本 ~200 行，全局代码 ~2000 行 | 大多数教学项目 > 1 万行，分散在数十个文件中 |
| 3 | **中文优先** — 代码注释、课程文档、API 说明全中文 | 主流开源项目以英文为主 |
| 4 | **双平台兼容** — Windows + Linux 体验完全一致，学生端无需服务器 | 多数项目以 Linux 为官方开发环境 |
| 5 | **从零手写、每行可讲** — 纯 PyTorch，不隐藏任何技术细节，适合课堂逐行解析 | 高层框架封装了大量实现细节 |

## 项目简介

**独立实现，从头写起。** 全部约 1350 行纯 PyTorch 代码，不依赖任何预训练模型或高层框架。从 BPE 分词器训练到模型搭建到训练脚本，全部手写。因为目标不是调参跑通一个 demo，而是理解每一行代码在做什么。

**处理真实世界的中文数据。** 四个数据源（维基百科、新闻、百度百科、社区问答）自行下载、提取、清洗、配比。繁体自动转简体，噪声过滤，四源按字符占比精确混合。没有用任何现成的训练数据集——数据管线本身就是一个独立的工程挑战。

**32K BPE 词表，自己训练的分词器。** 不是随便设的数字——词表太小会严重影响中文编码效率和生成质量，32K 是在覆盖度和模型效率之间精心选取的平衡点。

**做了真正的工程打磨。** numpy memmap 让几十 G 数据只占 1MB 内存，多线程分词把启动时间压缩到几分之一，断点续训保存优化器 + 调度器 + 混合精度缩放器全量状态。这些不是"高级功能"，是真正把模型训起来的过程中必然遇到的问题和必须做的选择。

**全链路可跑通。** 数据下载 → 清洗 → 配比 → 分词 → 训练 → 推理，一条命令搞定。训练监控接入 TensorBoard，推理端支持交互对话和多种采样策略。

**定位**：教育科研平台，用纯 PyTorch 从零实现 SwiGLU / GQA / RoPE / RMSNorm 等工业级组件，完整覆盖数据预处理、分词器训练、混合精度训练和KV Cache 推理全流程。最低仅需单张 12GB 消费级显卡，支持任何语言数据集训练模型。

**意义**：代码全栈开源、训练流程一键复现。让算力匮乏的研究者也能参与 LLM 架构研究，每一条训练曲线、每一步配置均可检验。

---

## 与工业级大模型的距离

XFIND-Nano 在**架构设计**和**工程实现**上与 Qwen3 / LLaMA 3 等工业级模型站在同一条地基上：

### 架构一致性

| 组件 | XFIND-Nano | LLaMA 3 | Qwen3 |
|------|------------|---------|-------|
| 范式 | Decoder-only | Decoder-only | Decoder-only |
| 归一化 | Pre-Norm + RMSNorm | Pre-Norm + RMSNorm | Pre-Norm + RMSNorm |
| 位置编码 | RoPE | RoPE | RoPE |
| 注意力 | GQA | GQA | GQA |
| 激活函数 | SwiGLU | SwiGLU | SwiGLU |
| Tokenizer | BPE（自训练） | BPE（tiktoken） | BPE（BBPE） |

### 工程能力（已实现）

| 能力 | 状态 | 说明 |
|------|------|------|
| 混合精度训练 (AMP) | 已实现 | `torch.cuda.amp`，GradScaler |
| 分布式训练 (DDP) | 已实现 | `DistributedSampler` + `DistributedDataParallel`，`torchrun` 一行启动 |
| 梯度累积 | 已实现 | `accumulate_grad`，支持小显存等效大 batch |
| 断点续训 | 已实现 | 保存 optimizer/scheduler/scaler 状态 |
| Cosine Warmup + Decay | 已实现 | LambdaLR，warmup_ratio 可配 |
| Tokenizer 训练 | 已实现 | BPE 自训练，SentencePiece 兼容 |
| FP16 量化推理 | 已实现 | `torch.float16` 导出，大小减半 |

### 差距仅在规模

| 维度 | XFIND-Nano | Qwen3-0.6B | 差距 |
|------|------------|------------|------|
| 参数量 | 39M | 0.6B | 15x |
| 训练数据 | 0.46B tokens (V2) | 18T tokens | 39,000x |
| V3 训练数据 | 1.2B tokens（4 源混合） | — | — |
| GPU | 1 × 4070 Ti | 数千 GPU·月 | 1,000x+ |
| 人类对齐 | 未应用 | SFT + RLHF | 在 XFIND-Pro 应用 |

### Remark

> 缩小规模差距所需的**工程能力已经具备**——分布式训练（DDP 代码已就绪，`torchrun` 一行启动）、
> 大规模数据过滤、梯度稳定性控制、SFT/DPO 对齐等技术在原理层面已经掌握，只差算力资源。
>
> **XFIND-LLM 的定位不是"替代大模型"，而是"证明大模型不再神秘"。**

---

## 核心技术栈

| 技术 | 说明 |
|------|------|
| 主干架构 | Pre-Norm Decoder-only（Llama/Qwen 标准） |
| 位置编码 | RoPE 旋转位置编码（支持长度外推） |
| 归一化 | RMSNorm（替代 LayerNorm） |
| 注意力 | GQA 分组查询注意力（8 查询头 / 4 KV 头） |
| 激活函数 | SwiGLU（替代 ReLU/GELU） |
| 训练精度 | BF16/FP16 AMP 混合精度 |
| 分布式 | DDP 多卡数据并行 |
| 学习率 | Warmup + CosineAnnealing |
| 推理加速 | KV Cache + 流式生成 + 多种采样策略 |

> **为什么不用 MoE（混合专家）？**  
> MoE 是面向千亿参数模型的扩展性技术（用稀疏激活换取更大参数规模），而 XFIND-Nano 只有 39M 参数，全部激活也没有任何瓶颈。在小模型上强行拆分 expert 会导致：（1）每个 expert 参数量过小（~10M），表达能力严重不足；（2）gate 路由网络引入额外训练开销和负载均衡损失；（3）expert collapse 等训练不稳定问题在小模型上更容易发生。对于 39M 规模，**密集模型（Dense Model）就是最优解**。

---

## 参数规格（XFIND-Nano ~39M）

| 参数 | 值 |
|------|-----|
| 上下文窗口 | 1024（RoPE 支持外推至 2048/4096） |
| 词表大小 | 32,000 |
| 网络层数 | 8 |
| 模型维度 | 512 |
| 查询注意力头 | 8 |
| KV 注意力头 | 4（GQA） |
| SwiGLU 中间维度 | 1365 |
| Dropout | 0.1 |
| Bias | 关闭 |
| 参数量 | **约 39M**（含 Embedding/lm_head 权重绑定） |

---

## 项目结构

```
Xfind-LLM/
├── xfind_train.py           # 训练脚本（AMP + DDP + 余弦退火 + 梯度累积）
├── xfind_infer.py           # 推理脚本（KV Cache + 交互式对话）
├── xfind_dataset.py         # 数据集（滑动窗口 + memmap 预分词）
├── xfind_quantize.py        # 量化导出（FP16）
│
├── models/
│   ├── __init__.py
│   ├── xfind_config.py      # 全局配置 + 命令行参数解析 + 显存适配指南
│   └── xfind_model.py       # 模型定义（RMSNorm/RoPE/GQA/SwiGLU/Decoder）
│
├── tokenizer/
│   ├── xfind_tokenizer.py   # BPE 分词器（SentencePiece 32K）
│   └── checkpoints/         # 分词器模型（.model / .vocab）
│
├── inference/
│   ├── sampler.py           # Temperature / TopK / TopP / RepetitionPenalty 采样
│   └── streamer.py          # 流式生成器（KV Cache 增量推理）
│
├── evaluation/
│   ├── perplexity.py        # PPL 评估工具
│   └── generate_samples.py  # 生成样例评测（多 prompt / 多温度）
│
├── tools/
│   ├── build_dataset.py     # 数据集构建（EOS 分隔 + 切分 train/valid/test）
│   ├── clean_text.py        # 文本清洗（长度过滤 + 语言占比过滤）
│   ├── download_v3_data.py  # V3 多源数据下载（新闻/百科/问答）
│   ├── quick_run.py         # 快速验证脚本（Level 1 冒烟 / 2 小规模 / 3 全量）
│   ├── eval_ppl.py          # 命令行 PPL 评估（可指定模型/批次）
│   ├── check_ckpt.py        # checkpoint 检查工具
│   ├── make_small_data.py   # 迷你数据集生成
│   └── _read_tb.py          # TensorBoard 数据读取
│
├── assets/
│   └── Xfind-logo.png       # XFIND 项目图标
│
├── data/
│   ├── raw/                 # 原始语料 + 清洗后文本
│   └── splits/              # 训练/验证/测试集（含 .npy 预分词缓存）
│
├── checkpoints/
│   ├── best_model.pt        # 最佳验证模型
│   ├── checkpoint_epoch_N.pt # 各 epoch 检查点
│   └── runs/                # TensorBoard 日志
│
├── README.md                # 本文件
├── requirements.txt         # Python 依赖
```

---

## 数据集

### 数据来源

#### V1/V2：中文维基百科

从 [modelscope.cn/datasets/caoaolong/zhwiki](https://www.modelscope.cn/datasets/caoaolong/zhwiki) 下载，包含 1388 万条中文维基百科条目（JSONL 格式）。

#### V3 新增：多源混合

| 数据源 | 来源 | 下载方式 |
|--------|------|----------|
| 中文新闻 (news2016zh) | [百度网盘](https://pan.baidu.com/s/1LJeq1dkA0wmYd9ZGZw72Xg) | 提取码 `film` |
| 百度百科 (563w_baidubaike) | 百度网盘 | 提取码 `bwvb` |
| webtext2019zh 社区问答json版 | [Kaggle](https://www.kaggle.com/datasets/terrychanorg/webtext2019zhjsonwebtext2019zh?resource=download&select=web_text_zh_train.json) | `kagglehub` |

> 详细下载流程见下方 **V3 多源数据下载** 章节。

### 完整数据流水线（5 步）

```
4 源原始数据 ──► 提取纯文本 ──► 清洗过滤 ──► 合并构建 ──► 训练时自动预分词
(17 GB)         (~3 GB 压缩)   (9.8 GB)    13,831,090 行   train_ids.npy
下载/解压         第 1-2 步       第 3 步       第 4 步         CPU 单线程 ~90min
```

| 阶段 | 输入 | 输出 | 工具 | 耗时 |
|------|------|------|------|------|
| 下载 | 百度网盘/Kaggle/ModelScope | 原始压缩包 | 手动 + `download_v3_data.py` | 视网速 |
| 提取 | 原始 JSON 压缩包 | `*_raw.txt` | `download_v3_data.py` | ~5min |
| 清洗 | 4 个 raw.txt (~3GB) | `*_clean.txt` | `tools/clean_text.py` | ~5min |
| 构建 | 4 个 clean.txt | `train/valid/test.txt` | `tools/build_dataset.py` | ~2min |
| 预分词 | `.txt` 首次加载 | `.npy` 缓存 | `xfind_dataset.py` (自动) | **~90min** |

#### 第 1 步：准备原始数据

将下载好的数据文件放入 `data/raw/` 目录：

| 数据源 | 文件名 | 放置路径 |
|--------|--------|----------|
| 百度百科 | `563w_baidubaike.json.7z` | `data/raw/` |
| 中文新闻 | `new2016zh.zip` | `data/raw/` |
| 社区问答 | `webtext2019zh.zip` | `data/` |

```bash
pip install py7zr kagglehub
```

#### 第 2 步：提取纯文本

> 自动化脚本支持 JSONL / JSON 数组 / 7z / zip / gzip 多种格式，一次运行全部提取。

```bash
# 一键提取所有源
python tools/download_v3_data.py

# 或逐个提取
python tools/download_v3_data.py --source news     # 新闻 → news_raw.txt
python tools/download_v3_data.py --source baike    # 百科 → baike_raw.txt
python tools/download_v3_data.py --source qa       # 问答 → qa_raw.txt
```

**提取逻辑**：

| 数据源 | 提取字段 | 格式 |
|--------|----------|------|
| news2016zh | title + desc + content | `标题：xxx 内容：xxx` |
| 百度百科 | title + summary + text | `词条：xxx 摘要：xxx 正文：xxx` |
| 社区问答 | title + content（过滤 star=0） | `问题：xxx 回答：xxx` |

**提取结果**：

| 产物 | 行数 |
|------|------|
| `data/raw/news_raw.txt` | 2,402,818 篇 |
| `data/raw/baike_raw.txt` | 2,137,581 条 |
| `data/raw/qa_raw.txt` | 4,028,889 组 |

#### 第 3 步：文本清洗

对每个源单独清洗（去噪 + 繁体转简体）：

```bash
python tools/clean_text.py --input data/raw/news_raw.txt --output data/raw/news_clean.txt --convert_zh
python tools/clean_text.py --input data/raw/baike_raw.txt --output data/raw/baike_clean.txt --convert_zh
python tools/clean_text.py --input data/raw/qa_raw.txt --output data/raw/qa_clean.txt --convert_zh
python tools/clean_text.py --input data/raw/wiki_clean.txt --output data/raw/wiki_clean_v3.txt --convert_zh
```

| 过滤规则 | 作用 |
|----------|------|
| `min_len=10` | 过滤过短条目 |
| `max_len=2000` | 过滤异常长条目 |
| 中英文占比 > 30% | 过滤纯数字/符号/空格 |
| `--convert_zh` | 繁体统一转简体（需 `pip install zhconv`） |

**清洗结果汇总**：

| 数据源 | 清洗前 | 清洗后 | 留存率 |
|--------|--------|--------|--------|
| 中文维基 | 5,646,694 | 5,646,653 | 100.0% |
| 社区问答 | 4,028,889 | 4,027,723 | 99.97% |
| 百度百科 | 2,137,581 | 2,135,342 | 99.9% |
| 中文新闻 | 2,402,818 | 2,021,331 | 84.1% |

> 新闻清洗率偏低（84%）属正常——新闻数据含更多噪声、短讯、横幅广告等低质量内容。

#### 第 4 步：合并构建训练/验证/测试集

```bash
python tools/build_dataset.py \
    --input data/raw/news_clean.txt \
            data/raw/baike_clean.txt \
            data/raw/qa_clean.txt \
            data/raw/wiki_clean_v3.txt \
    --output_dir data/v3_splits \
    --ratios 0.38 0.29 0.21 0.12 \
    --total_tokens 1.2
```

自动打乱 + 配比裁剪 + 90/5/5 切分，行间插入 `<|endoftext|>` 分隔符。

**训练数据配比**（`--ratios 0.38 0.29 0.21 0.12`，字符占比 ≈ 分词语料占比）：

| 数据源 | 行数 | 字符 | 占比 | tokens |
|--------|------|------|------|--------|
| 中文维基 | 5,171,660 | 638M | 38.9% | ~0.46B |
| 中文新闻 | 595,998 | 487M | 29.7% | ~0.35B |
| 百度百科 | 2,135,342 | 316M | 19.2% | ~0.23B |
| 社区问答 | 1,086,103 | 202M | 12.3% | ~0.14B |
| **合计** | **8,989,103** | **1,643M** | **100%** | **~1.17B** |

**切分结果**：

| 划分 | 文件 | 大小 | 行数 | 占比 |
|------|------|------|------|------|
| train | `data/v3_splits/train.txt` | 4.26 GB | 8,090,192 | 90% |
| valid | `data/v3_splits/valid.txt` | 236 MB | 449,455 | 5% |
| test | `data/v3_splits/test.txt` | 236 MB | 449,455 | 5% |
| **合计** | | **~4.7 GB** | **8,989,102** | — |

#### 自动预分词（训练时触发）

首次训练时，数据集模块会自动对文本分词并保存为 numpy memmap 格式：

```
data/v3_splits/
├── train.txt           # 原始文本
├── train_ids.npy       # 预分词 token ID（~3.2 GB）
├── valid.txt
├── valid_ids.npy       # 预分词 token ID（~180 MB）
├── test.txt
└── test_ids.npy        # 预分词 token ID（~180 MB）
```

后续加载直接读取 `.npy` 文件（memmap 磁盘映射），秒级完成，内存仅 ~1MB。

> **关于首次预分词速度**：BPE 分词本质是字符串匹配（贪心最长匹配查字典），不是矩阵运算，
> 因此 **无法用 GPU 加速**。首次处理 809 万行文本约需 45 分钟（CPU 单线程），
> 这是正常现象，耐心等待即可。`.npy` 文件生成后，后续训练秒级加载。

| 数据集 | tokens | 样本数（seq=1024, stride=768） |
|--------|--------|-------------------------------|
| train | ~11.7 亿 | ~1,520,000 |
| valid | ~0.33 亿 | ~43,000 |
| test | ~0.35 亿 | ~45,000 |

#### 内部预处理详解

`xfind_dataset.py` 在首次运行时自动完成以下流程：

**1. 分块读取 + BPE 编码**

```python
# 每次读取 1MB 文本 → 调用 SentencePiece encode() → 累积 token ID 列表
chunk_size = 1024 * 1024  # 1MB
with open(text_file, 'r', encoding='utf-8') as f:
    while True:
        chunk = f.read(chunk_size)
        if not chunk: break
        ids = tokenizer.encode(chunk)  # CPU BPE 编码
        all_ids.extend(ids)
```

**2. 持久化缓存（numpy memmap）**

```python
ids_array = np.array(all_ids, dtype=np.uint32)
np.save(ids_file, ids_array)           # 写入磁盘，一次性开销
self.all_ids = np.load(ids_file, mmap_mode='r')  # 后续秒级加载
```

**3. 滑动窗口采样（`__getitem__`）**

```python
# stride=768, max_seq_len=1024
# 样本数 = (总 tokens - 1024) // 768
start = idx * self.stride          # 第 idx 个窗口起点
end = start + max_seq_len + 1      # 多取 1 个 token 作为 label
return torch.from_numpy(ids[start:end].astype(np.int64))
```

**4. collate_fn 批处理**

```python
# input:  [0, 1, 2, ..., 1023]  → 模型输入
# target: [1, 2, 3, ..., 1024]  → 预测目标（左移一位）
```

| 参数 | v1 | v2 | 说明 |
|------|-----|-----|------|
| stride | 512 (50%重叠) | 768 (25%重叠) | 减少模板重复强化 |
| label_smoothing | 0.0 | 0.1 | 防止 softmax 过尖导致生成死循环 |
| EOS 分隔符 | 无 | `<\|endoftext\|>` | 文档间插入，帮助模型学习边界 |

---

## V3 多源数据下载

V3 版本使用 4 源混合训练（中文维基 + 新闻 + 百科 + 问答），合计约 1.2B tokens。以下为新增三个数据源的下载链接：

| 数据源 | 平台 | 链接 | 提取码/说明 |
|--------|------|------|-------------|
| 中文新闻 (news2016zh) | 百度网盘 | [pan.baidu.com](https://pan.baidu.com/s/1LJeq1dkA0wmYd9ZGZw72Xg) | 提取码: `film` |
| 百度百科 (563w_baidubaike) | 百度网盘 | [pan.baidu.com](https://pan.baidu.com/s/1jIpCHnWLTNYabftavo3DVw?pwd=bwvb) | 提取码: `bwvb` |
| webtext2019zh 社区问答json版 | Kaggle | [kaggle.com](https://www.kaggle.com/datasets/terrychanorg/webtext2019zhjsonwebtext2019zh?resource=download&select=web_text_zh_train.json) | 使用 `kagglehub` 直接下载 |

```bash
# 下载三个数据源并提取纯文本
pip install py7zr kagglehub
python tools/download_v3_data.py

# 或逐个下载/提取
python tools/download_v3_data.py --source news     # 新闻
python tools/download_v3_data.py --source baike    # 百科
python tools/download_v3_data.py --source qa       # 问答
```

> 下载后自动提取为 `data/raw/news_raw.txt`、`baike_raw.txt`、`qa_raw.txt`，再通过 `clean_text.py` 和 `build_dataset.py` 与已有维基数据合并即可完成 V3 数据集构建。

---

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.5+ with CUDA 12.4
- RTX 4070 Ti 12GB（或同等显存）

### 安装

```bash
pip install -r requirements.txt
```

### 1. 数据准备

V3 使用 4 源混合数据（维基 + 新闻 + 百科 + 问答），详细流水线见上方 [完整数据流水线](#完整数据流水线5-步)。快速复现只需两步：

```bash
# Step 1: 下载并提取原始数据（详见 V3 多源数据下载章节）
pip install py7zr kagglehub
python tools/download_v3_data.py

# Step 2: 清洗 + 合并构建
python tools/clean_text.py --input data/raw/news_raw.txt --output data/raw/news_clean.txt --convert_zh
python tools/clean_text.py --input data/raw/baike_raw.txt --output data/raw/baike_clean.txt --convert_zh
python tools/clean_text.py --input data/raw/qa_raw.txt --output data/raw/qa_clean.txt --convert_zh
python tools/build_dataset.py \
    --input data/raw/news_clean.txt data/raw/baike_clean.txt \
            data/raw/qa_clean.txt data/raw/wiki_clean.txt \
    --output_dir data/v3_splits
```

产物：`data/v3_splits/train.txt`（4.26 GB，809 万行）、`valid.txt`、`test.txt`

### 2. 训练

```bash
# V3 多源训练（4 源混合 1.2B+ tokens）
conda activate dl2llm
python xfind_train.py --data_dir ./data/v3_splits --epochs 8
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--epochs` | 8 | 训练轮数 |
| `--batch_size` | 8 | Micro-batch 大小（显存安全值） |
| `--accumulate_grad` | 8 | 梯度累积步数（有效 batch = 64） |
| `--label_smoothing` | 0.1 | 标签平滑 |
| `--lr` | 3e-4 | 峰值学习率 |

> 所有参数均有合理默认值，`python xfind_train.py --data_dir ./data/v3_splits` 即可直接启动训练。
>
> **训练耗时**：首次运行会自动分词（约 45 分钟），随后每 epoch 约 8 小时（809 万行，~1.5M 样本），8 epoch 总计约 3.5 天。
>
> **断点续训**：训练过程中每个 epoch 自动保存 checkpoint。若中途中断，指定 checkpoint 路径续训：
> ```bash
> python xfind_train.py --data_dir ./data/v3_splits --load_checkpoint checkpoints/checkpoint_epoch_2.pt
> ```

训练监控：

```bash
tensorboard --logdir ./checkpoints/runs
```

### 3. 推理

```bash
# 单次生成
python xfind_infer.py --model checkpoints/best_model.pt --prompt "人工智能是"

# 交互模式
python xfind_infer.py --model checkpoints/best_model.pt

# 不同采样策略
python xfind_infer.py --model checkpoints/best_model.pt --temperature 0.8 --top_k 50 --top_p 0.9
```

| 采样参数 | 默认值 | 说明 |
|----------|--------|------|
| `--temperature` | 1.0 | 越高越随机（0.8-1.3 推荐） |
| `--top_k` | 50 | Top-K 采样 (0=禁用) |
| `--top_p` | 0.9 | Nucleus 采样 (0=禁用) |
| `--max_new_tokens` | 256 | 最大生成长度 |

### 4. 量化导出

```bash
python xfind_quantize.py --input checkpoints/best_model.pt --output checkpoints/model_fp16.pt
```

---

## 模型详细介绍

### BPE Tokenizer

基于 SentencePiece 的 32K 词表 BPE 分词器，在中文维基百科上自训练。将原始文本切分为子词单元，输出 token ID 序列作为模型输入。支持中英混合分词，无需语言标记。

### Pre-Norm + RMSNorm

采用 Pre-Norm 架构：归一化放在每个子层（注意力 / 前馈网络）之前，而非之后。相比传统 Post-Norm，梯度流动更顺畅，训练更稳定，是现代 LLM 的统一选择。

归一化函数使用 RMSNorm（均方根归一化），替代 LayerNorm。去掉均值中心化，仅做缩放归一化，计算更快，效果相当：

```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
```

### RoPE（旋转位置编码）

通过旋转矩阵对 Q 和 K 的每对维度施加位置相关的旋转，天然支持相对位置编码和序列长度外推（1024 → 2048/4096）。

### GQA（分组查询注意力）

8 个查询头共享 4 个 KV 头，每个 KV 头服务 2 个查询头。相比全 MHA（8Q/8KV），KV Cache 减半；相比 MQA（8Q/1KV），保留更好的表达能力。

### SwiGLU

门控前馈网络，使用 SiLU 激活的门控机制替代传统 ReLU：

```
FFN(x) = SiLU(x @ W_gate) * (x @ W_up) @ W_down
```

三层投影（gate/up/down），d_model(512) → d_ff(1365) → d_model(512)。

### KV Cache

自回归推理时，缓存已生成的 Key/Value 投影矩阵，每步仅计算新增 token 的注意力，避免历史序列的重复计算。将单步推理复杂度从 O(n²) 降至 O(n)，是长文本生成的必备优化。

### Decoder-only Stack

整体数据流：

```
Token Embedding
  → N × DecoderLayer(
      RMSNorm → GQA(含 RoPE) → +残差
    → RMSNorm → SwiGLU → +残差
    )
  → RMSNorm → LM Head → Softmax
```

8 层堆叠，每层结构相同，通过残差连接保持梯度流动。LM Head 与 Token Embedding 权重绑定，共享词表空间。

---

## 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW（betas=0.9,0.95, eps=1e-8, wd=0.01） |
| 学习率 | 3e-4 峰值，1% Warmup + 余弦退火 |
| 精度 | BF16 AMP |
| 有效 batch | 64（batch_size=8 × accumulate_grad=8） |
| 训练轮数 | 8 epochs |
| 梯度裁剪 | 1.0 |
| 标签平滑 | 0.1（Label Smoothing） |
| 损失函数 | CrossEntropy（ignore pad, label_smoothing=0.1） |

### 显存实测（4070 Ti 12GB）

| batch_size | max_seq_len | accumulate | 有效batch | 显存 | 速度 |
|------------|-------------|------------|-----------|------|------|
| 8 ✅（默认） | 1024 | 8 | 64 | ~5 GB | ~4.3 it/s |
| 16 ❌ | 1024 | 4 | 64 | >12 GB | 显存溢出（注意力矩阵 2.1GB） |

### V1 训练结果

V1 使用 label_smoothing=0.0, stride=512, epochs=5，在中文维基百科上训练：

| Epoch | Val Loss | PPL | 单轮降幅 |
|-------|----------|-----|---------|
| 1 | ~3.87 | 47.94 | — |
| 2 | ~3.82 | 45.60 | -2.34 |
| 3 | 3.809 | 45.11 | -0.49 |
| 4 | ~3.76 | 42.95 | -2.16 |
| 5 | 3.643 | **38.19** | **-4.76（最大降幅）** |

> PPL 每 epoch 持续刷新，第 5 轮降幅最大，模型远未饱和。生成中文可辨识但 token 重复循环严重——这是 39M 参数容量不足的典型表现。

### V3 训练结果（4 源混合 1.2B tokens，label_smoothing=0.1）

| Epoch | Train Loss | Val Loss | Val PPL | PPL 降幅 |
|-------|-----------|----------|---------|---------|
| 0 | 5.2603 | 3.8242 | 45.80 | — |
| 1 | 4.7488 | 3.7038 | 40.60 | -5.20 |
| 2 | 4.6784 | 3.6490 | 38.44 | -2.16 |
| 3 | 4.6405 | 3.6143 | 37.12 | -1.32 |
| 4 | 4.6147 | 3.5887 | 36.19 | -0.93 |
| 5 | 4.5957 | 3.5702 | 35.55 | -0.64 |
| 6 | 4.5824 | 3.5585 | 35.13 | -0.42 |
| 7 | 4.5746 | 3.5532 | **34.93** | -0.20 |

> **最终 PPL = 34.93**，较 V1 的 38.19 下降了 3.26 个 point（8.5%）。8 个 epoch 全程无过拟合，val_loss 每个 epoch 持续下降。
>
> V3 相比 V1 的三项关键改进全部见效：（1）4 源混合数据替代单源 wiki，数据多样性大幅提升；（2）label_smoothing=0.1 替代 0.0，softmax 不再过度尖锐，生成循环显著减轻；（3）stride=768 替代 512，样本间重叠降至 25% 减少过拟合风险。
>
> 生成效果：epoch 0 输出模板化循环（"今日天气: 23 点:..."），epoch 1 后能生成完整通顺的中文句子（"一看就是一场狂风暴雨的,你应该看看最近天气情况"），epoch 7 在长 prompt 下可产出主题一致、语法正确的短文。PPL ~35 时采样仍不稳定，短 prompt 容易塌缩——这是 39M 模型的容量天花板。后续通过 SFT（指令微调）可大幅改善生成质量和稳定性。

---
## 开发计划

> 详细的阶段任务、配置方案和数据配比见 **[开发计划文档](docs/XFIND-Lite-语言模型开发计划.md)**。
>
> V2 已暂停（epoch 4/8，PPL=45.01），**V3 训练完成**（4 源混合数据，8 epoch 最优 PPL=34.93），下一步：SFT 指令微调。
> 39M 架构已验证完毕，数据策略（4 源配比、label_smoothing、stride）在 V3 得到验证，后续逐级复用到 80M 和 0.6B。
>
> 四个阶段（架构/训练/数据/推理）全部完成，V3 epoch 7 最优 PPL=34.93。

---

## 长期迭代路线

| 版本 | 参数量 | 定位 | 硬件需求 | 操作系统 | 状态 |
|------|--------|------|----------|----------|------|
| XFIND-Nano | ~39M | 教学入门 / 快速消融 | 单卡 12GB | Windows + Linux | 已完成 |
| XFIND-Lite | ~80M | 科研进阶 / 可复现消融基线 | 单卡 12GB 可训 / **4×48GB 加速** | **Linux** | 自给自足 |
| XFIND-Pro | ~0.6B | 前沿探索 / 架构创新验证 | 多卡集群 | **Linux** | **寻求合作** |

> 中间挡位（120M/300M）已省略——既牺牲了 80M 的迭代速度，又够不到 0.6B 的生成质量。
>
> **验证路线**：39M 验证架构组件在小模型上的相对收益 → 80M 在更大规模验证数据策略 → 0.6B 完成端到端验证。每个版本的结论逐级复用，降低大模型试错成本。
>
> **各版本数据需求**：39M → 0.78B（Chinchilla），80M → 1.5B，0.6B → **12B**（Chinchilla 最优，6-12B 实用区间）。数据配比方案先在XFIND-Nano v3 验证，再逐级复用。

---

## 架构参考

XFIND-Nano 采用与 **LLaMA 3** 和 **Qwen3** 等行业主流模型相同的架构设计，基于以下研究的公认最佳实践组合：

| 组件 | 论文 | 出处 |
|------|------|------|
| Pre-Norm | On Layer Normalization in the Transformer Architecture | Xiong et al., ICML 2020 |
| RoPE | RoFormer: Enhanced Transformer with Rotary Position Embedding | Su et al., 2021 ([arXiv:2104.09864](https://arxiv.org/abs/2104.09864)) |
| RMSNorm | Root Mean Square Layer Normalization | Zhang & Sennrich, NeurIPS 2019 ([arXiv:1910.07467](https://arxiv.org/abs/1910.07467)) |
| GQA | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | Ainslie et al., EMNLP 2023 ([arXiv:2305.13245](https://arxiv.org/abs/2305.13245)) |
| SwiGLU | GLU Variants Improve Transformer | Shazeer, 2020 ([arXiv:2002.05202](https://arxiv.org/abs/2002.05202)) |

```bibtex
@inproceedings{prenorm,
  title     = {On Layer Normalization in the Transformer Architecture},
  author    = {Ruibin Xiong and Yunchang Yang and Di He and Kai Zheng and Shuxin Zheng and Chen Xing and Huishuai Zhang and Yanyan Lan and Liwei Wang and Tie-Yan Liu},
  booktitle = {ICML},
  year      = {2020}
}

@article{rope,
  title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author  = {Jianlin Su and Yu Lu and Shengfeng Pan and Ahmed Murtadha and Bo Wen and Yunfeng Liu},
  journal = {arXiv preprint arXiv:2104.09864},
  year    = {2021}
}

@article{rmsnorm,
  title   = {Root Mean Square Layer Normalization},
  author  = {Biao Zhang and Rico Sennrich},
  journal = {NeurIPS},
  year    = {2019}
}

@inproceedings{gqa,
  title     = {GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
  author    = {Joshua Ainslie and James Lee-Thorp and Michiel de Jong and Yury Zemlyanskiy and Federico Lebr{\'o}n and Sumit Sanghai},
  booktitle = {EMNLP},
  year      = {2023}
}

@article{swiglu,
  title   = {GLU Variants Improve Transformer},
  author  = {Noam Shazeer},
  journal = {arXiv preprint arXiv:2002.05202},
  year    = {2020}
}
```

---

## 许可证

Apache License 2.0
