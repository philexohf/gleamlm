# XFIND-LLM —— 面向教育和科研的自研语言模型

<img src="./assets/Xfind-logo.png" style="zoom: 25%;" />

> **从零自研的现代架构 LLM 系列**——39M 快速迭代，80M 策略验证，0.6B 生产级。降低 LLM 架构研究的算力门槛。

**定位**：教育科研平台，用纯 PyTorch 从零实现 SwiGLU / GQA / RoPE / RMSNorm 等工业级组件，完整覆盖分词器训练、数据预处理、混合精度训练、KV Cache 推理全流程。最低仅需单张 12GB 消费级显卡，架构与数据和具体使用的语言完全无关。

**意义**：代码全栈开源、每行手写，训练流程一键复现。让算力匮乏地区的研究者也能参与 LLM 架构研究，每一条训练曲线、每一步配置均可检验。

---

## 与工业级大模型的距离

XFIND-Mini 在**架构设计**和**工程实现**上与 Qwen3 / LLaMA 3 等工业级模型站在同一条地基上：

### 架构一致性

| 组件 | XFIND-Mini | LLaMA 3 | Qwen3 |
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

| 维度 | XFIND-Mini | Qwen3-0.6B | 差距 |
|------|------------|------------|------|
| 参数量 | 39M | 0.6B | 15x |
| 训练数据 | 0.46B tokens | 18T tokens | 39,000x |
| GPU | 1 × 4070 Ti | 数千 GPU·月 | 1,000x+ |
| 人类对齐 | 未应用 | SFT + RLHF | 在 XFIND-0.6B 应用 |

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
> MoE 是面向千亿参数模型的扩展性技术（用稀疏激活换取更大参数规模），而 XFIND-Mini 只有 39M 参数，全部激活也没有任何瓶颈。在小模型上强行拆分 expert 会导致：（1）每个 expert 参数量过小（~10M），表达能力严重不足；（2）gate 路由网络引入额外训练开销和负载均衡损失；（3）expert collapse 等训练不稳定问题在小模型上更容易发生。对于 39M 规模，**密集模型（Dense Model）就是最优解**。

---

## 参数规格（XFIND-Mini ~39M）

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

### 数据来源：中文维基百科

从 [modelscope.cn/datasets/caoaolong/zhwiki](https://www.modelscope.cn/datasets/caoaolong/zhwiki) 下载，包含 1388 万条中文维基百科条目（JSONL 格式）。

### 完整数据流水线（4 步）

```
1388 万条 JSONL ──► 提取纯文本 ──► 清洗过滤 ──► 构建数据集 ──► 训练时自动预分词
   (2.35 GB)          (~2 GB)       (1.83 GB)    train/valid/test    (train_ids.npy)
   原始维基            第 1-2 步       第 3 步        第 4 步            CPU 单线程 30-60min
```

| 阶段 | 输入 | 输出 | 工具 | 耗时 |
|------|------|------|------|------|
| 下载 | ModelScope | `zhwiki_dataset.jsonl` | `git clone` | ~10min |
| 提取 | JSONL 1388万行 | `wiki_raw.txt` | 内联 Python | ~2min |
| 清洗 | 2GB 原始文本 | `wiki_clean.txt` | `tools/clean_text.py` | ~3min |
| 构建 | 清洗后文本 | `train/valid/test.txt` | `tools/build_dataset.py` | ~1min |
| 预分词 | `.txt` 首次加载 | `.npy` 缓存 | `xfind_dataset.py` (自动) | **30-60min** |

#### 第 1 步：下载原始数据

```bash
git clone https://www.modelscope.cn/datasets/caoaolong/zhwiki.git data/zhwiki_raw
```

原始文件：`data/zhwiki_raw/zhwiki_dataset.jsonl`（2.35 GB，1388 万行）

每行为一个 JSON 对象：

```json
{"id": "文章ID", "title": "标题", "content": "正文", "source": "来源URL"}
```

#### 第 2 步：提取纯文本

从 JSONL 中提取 `content` 字段，写入纯文本文件：

```bash
python -c "
import json
with open('data/zhwiki_raw/zhwiki_dataset.jsonl', 'r', encoding='utf-8') as fin, \
     open('data/raw/wiki_raw.txt', 'w', encoding='utf-8') as fout:
    for line in fin:
        text = json.loads(line)['content'].strip()
        if text:
            fout.write(text + '\n'
"
```

产物：`data/raw/wiki_raw.txt`（~2 GB）

#### 第 3 步：文本清洗

```bash
python tools/clean_text.py --input data/raw/wiki_raw.txt \
                           --output data/raw/wiki_clean.txt \
                           --min_len 30 --max_len 3000
```

| 过滤规则 | 作用 |
|----------|------|
| `min_len=30` | 过滤过短的占位条目/一句话条目 |
| `max_len=3000` | 过滤表格展开、模板残留等异常长条目 |
| 中英文占比 > 30% | 过滤纯数字、纯符号、纯空格条目 |

**数据量变化**：

| 阶段 | 大小 | 行数 | 过滤率 |
|------|------|------|--------|
| 原始 JSONL | 2.35 GB | 1388 万 | — |
| 提取纯文本 | ~2 GB | 1388 万 | — |
| 清洗后 | 1.83 GB | 565 万 | 59.3% |

> **关于文档级去重**：本项目使用中文维基百科作为数据源，维基本身已自然去重（每个条目唯一），
> 因此未引入 MD5 / MinHash 等文档级去重步骤。若将来改用 Web 爬取语料（CommonCrawl 等），
> 需要加入去重以清理转载和重复页面。

#### 第 4 步：构建训练/验证/测试集

```bash
python tools/build_dataset.py --input data/raw/wiki_clean.txt \
                               --output_dir data/splits \
                               --train_ratio 0.95 --valid_ratio 0.025
```

自动按比例随机打乱并切分：

| 划分 | 文件 | 大小 | 行数 | 占比 |
|------|------|------|------|------|
| train | `data/splits/train.txt` | 743 MB | 1073 万 | 95% |
| valid | `data/splits/valid.txt` | 19.5 MB | 28 万 | 2.5% |
| test | `data/splits/test.txt` | 19.5 MB | 28 万 | 2.5% |

#### 自动预分词（训练时触发）

首次训练时，数据集模块会自动对文本分词并保存为 numpy memmap 格式：

```
data/splits/
├── train.txt           # 原始文本
├── train_ids.npy       # 预分词 token ID（~1.7 GB）
├── valid.txt
├── valid_ids.npy       # 预分词 token ID（~48 MB）
├── test.txt
└── test_ids.npy        # 预分词 token ID（~48 MB）
```

后续加载直接读取 `.npy` 文件（memmap 磁盘映射），秒级完成，内存仅 ~1MB。

> **关于首次预分词速度**：BPE 分词本质是字符串匹配（贪心最长匹配查字典），不是矩阵运算，
> 因此 **无法用 GPU 加速**。首次处理 565 万行文本约需 30-60 分钟（CPU 单线程），
> 这是正常现象，耐心等待即可。`.npy` 文件生成后，后续训练秒级加载。

| 数据集 | tokens | 样本数（v2: seq=1024, stride=768） |
|--------|--------|-------------------------------|
| train | 4.56 亿 | ~593,749 |
| valid | 1200 万 | ~15,624 |
| test | 1200 万 | ~15,624 |

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

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.5+ with CUDA 12.4
- RTX 4070 Ti 12GB（或同等显存）

### 安装

```bash
pip install -r requirements.txt
```

### 1. 数据准备（约 10 分钟）

```bash
# Step 1: 下载中文维基百科（约 2.35 GB）
git clone https://www.modelscope.cn/datasets/caoaolong/zhwiki.git data/zhwiki_raw

# Step 2: 提取纯文本
python -c "
import json
with open('data/zhwiki_raw/zhwiki_dataset.jsonl', 'r', encoding='utf-8') as fin, \
     open('data/raw/wiki_raw.txt', 'w', encoding='utf-8') as fout:
    for line in fin:
        text = json.loads(line)['content'].strip()
        if text:
            fout.write(text + '\n'
"

# Step 3: 文本清洗
python tools/clean_text.py --input data/raw/wiki_raw.txt \
                           --output data/raw/wiki_clean.txt \
                           --min_len 30 --max_len 3000

# Step 4: 切分训练/验证/测试集
python tools/build_dataset.py --input data/raw/wiki_clean.txt \
                               --output_dir data/splits \
                               --train_ratio 0.95 --valid_ratio 0.025
```

产物：`data/splits/train.txt`（743 MB，1073 万行）、`valid.txt`、`test.txt`

### 2. 训练

```bash
# 首次运行会自动训练 32K BPE 分词器（约 10 分钟），然后启动模型训练
conda activate dl2llm
python xfind_train.py
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--epochs` | 8 | 训练轮数 |
| `--batch_size` | 8 | Micro-batch 大小（显存安全值） |
| `--accumulate_grad` | 8 | 梯度累积步数（有效 batch = 64） |
| `--label_smoothing` | 0.1 | 标签平滑（防止 softmax 过尖） |
| `--lr` | 3e-4 | 峰值学习率 |

> 所有参数均有合理默认值，`python xfind_train.py` 即可直接启动训练。
>
> **训练耗时**：每 epoch 约 4 小时（batch_size=8, 592K 样本, ~4.3 it/s），8 epoch 总计约 32 小时。
>
> **断点续训**：训练过程中每个 epoch 自动保存 checkpoint。若中途中断，指定 checkpoint 路径续训：
> ```bash
> python xfind_train.py --load_checkpoint checkpoints/checkpoint_epoch_2.pt
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

---
## 开发计划

> 详细的阶段任务、配置方案和数据配比见 **[开发计划文档](Xfind-Mini-大模型开发计划-v1.1.md)**。
>
> 当前 **V2 训练中**（epoch 4/8，PPL=45.01），V3 数据扩展和 XFIND-80M 紧随其后。
> 39M 架构已验证完毕，瓶颈在数据规模与配比。V3 将用数据改进推动同一架构的生成质量，验证后的策略逐级复用到 80M 和 0.6B。
>
> 四个阶段（架构/训练/数据/推理）全部完成，V1 epoch 5 最优 PPL=38.19。

---

## 长期迭代路线

| 版本 | 参数量 | 定位 | 硬件需求 | 操作系统 | 状态 |
|------|--------|------|----------|----------|------|
| XFIND-Mini | ~39M | 教育科研 / 架构消融验证 | 单卡 12GB | Windows/Linux | 已完成 |
| XFIND-80M | ~80M | 实验基线 / 数据策略验证 | 4×48GB | Linux | 自给自足 |
| XFIND-0.6B | ~0.6B | 生产级多语言 / 顶会论文发表 | 多卡集群 | Linux | **寻求合作** |

> 中间挡位（120M/300M）已省略——既牺牲了 80M 的迭代速度，又够不到 0.6B 的生成质量。
>
> **验证路线**：39M 验证架构组件在小模型上的相对收益 → 80M 在更大规模验证数据策略 → 0.6B 完成端到端验证。每个版本的结论逐级复用，降低大模型试错成本。
>
> **各版本数据需求**：39M → 0.78B（Chinchilla），80M → 1.5B，0.6B → **12B**（Chinchilla 最优，6-12B 实用区间）。数据配比方案先在XFIND-Mini v3 验证，再逐级复用。

---

## 架构参考

XFIND-Mini 采用与 **LLaMA 3** 和 **Qwen3** 等行业主流模型相同的架构设计，基于以下研究的公认最佳实践组合：

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
