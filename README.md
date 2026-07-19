# GleamLM —— 面向教育和研究的小型语言模型

<img src="./assets/GleamLM.png" width="300" alt="GleamLM Logo" />



 **项目持续开发中， 点个 Star ⭐ 收藏，更新不错过。**

## 项目简介

纯 PyTorch 从零实现，零 HuggingFace 依赖，覆盖 **多源中文数据管线**（下载→清洗→去重→字符加权配比）→ **BBPE 分词器训练**（自研，零外部依赖）→ **Decoder-only 模型**（SwiGLU / GQA / RoPE / QK-Norm）→ **AMP + DDP 训练**（断点续训保存 optimizer/scheduler/scaler 全量状态）→ **SFT / DPO 对齐**（ChatML + loss mask）→FP16量化 → **KV Cache 流式推理**全链路。

| 版本 | 参数量 | 定位 | 状态 |
|------|--------|------|------|
| **GleamLM-Nano** | ~40M | 教学入门，单卡 12GB 即可完整训练 | ✅ 已完成 |
| **GleamLM-Lite** | ~87M | 消融实验平台，FFN 3.4× 扩容，Windows/Linux 双平台 | ✅ 已完成 |

## 技术架构

| 组件 | 方案 | 对标 |
|------|------|------|
| 范式 | Decoder-only | LLaMA 3 / Qwen3 |
| 归一化 | Pre-Norm + RMSNorm | LLaMA 3 / Qwen3 |
| 位置编码 | RoPE（支持长度外推） | LLaMA 3 / Qwen3 |
| 注意力 | GQA（8 Q-heads / 4 KV-heads）+ QK-Norm | LLaMA 3 |
| 激活函数 | SwiGLU | LLaMA 3 / Qwen3 |
| Tokenizer | BBPE 12K（自研，纯 Python） | — |
| 训练精度 | BF16/FP16 AMP | — |
| 分布式 | DDP（`torchrun` 一行启动） | — |
| 推理加速 | KV Cache + 流式生成 + 多采样策略 | — |

### 模型规格

| 参数 | Nano ~40M | Lite ~87M |
|------|:---:|:---:|
| 上下文窗口 | 1024 | **2048** |
| 词表大小 | 12,002（自研 BBPE） | 12,002（复用） |
| 网络层数 | 12 | 12 |
| 模型维度 | 512 | **768** |
| QK-Norm | ✅ | ✅ |
| 查询头 / KV 头 | 8 / 4 | **12 / 6** |
| SwiGLU 中间维度 | 1365 | **2048**（3.4× FFN 容量） |
| Dropout | 0.1 | 0.0 |
| Flash Attention | — | ✅ |
| Z-Loss | — | 1e-4 |
| 参数量 | **~40M** | **~87M** |
| Embed 占比 | 15% | 11% |
| FFN 参数 | 16.8M (41%) | **56.6M (65%)** |

> Lite 设计原则：测试证实 12 层是中文生成的硬阈值，且事实知识 100% 存于 FFN。因此保持 12 层不动，d_model 扩至 768，d_ff 按 SwiGLU 标准公式扩至 2048（3.4× FFN 容量），词表复用 Nano 的 12K。

---

## 项目结构

```
GleamLM/
├── gleamlm/                     # 共享核心库
│   ├── models/model.py          # GleamLMModel（RMSNorm/RoPE/GQA/SwiGLU/QK-Norm）
│   ├── tokenizer/tokenizer.py   # BBPE 12K 分词器（纯 Python 零依赖）
│   ├── dataset/dataset.py       # LMDataset（memmap 滑动窗口 + 预分词缓存）
│   ├── training/                # 共享训练模块（提取自 nano/lite）
│   │   ├── base_trainer.py      # set_seed / evaluate / checkpoint save/load / optimizer/scheduler/dataloader
│   │   ├── sft_trainer.py       # SFTDataset / train_one_epoch_sft / evaluate_sft
│   │   └── dpo_trainer.py       # DPODataset / dpad_collate / dpo_loss / train_one_epoch_dpo
│   ├── inference/               # KV Cache 流式生成 + 多采样策略
│   │   ├── generate.py          # generate_response (ChatML + KV Cache)
│   │   ├── sampler.py           # temperature / top-k / top-p / repetition_penalty
│   │   ├── streamer.py          # TextStreamer（流式输出 + stop_on_endoftext）
│   │   └── cli.py               # 统一推理 CLI（支持 --variant nano|lite|pro）
│   ├── preprocessing/           # 数据预处理库
│   │   ├── clean_text.py        # 文本清洗（HTML剥离/长度过滤/繁简转换）
│   │   ├── dedup_text.py        # 去重（MD5 exact / prefix）
│   │   ├── filter_qa.py         # QA 质量过滤
│   │   └── build_dataset.py     # 多源字符加权混合 + train/valid/test 切分
│   ├── deploy/                 # 模型部署工具
│   │   └── quantize.py          # FP32 → FP16（变体无关，自动识别架构）
│   └── utils/                   # 工具模块
│       ├── config.py            # YAML 配置加载 / deep merge / CLI override
│       ├── torch_utils.py       # Cosine LR / WSD LR / safe_autocast / Z-Loss
│       ├── paths.py             # 统一路径常量 (get_root_dir / get_default_checkpoint_dir)
│       └── checkpoint.py        # assert_same_architecture（checkpoint 加载前架构校验）
│
├── scripts/                      # 统一训练入口（--variant nano|lite|pro）
│   ├── train.py                  # 预训练（AMP + DDP + Cosine/WSD + 断点续训）
│   ├── sft.py                    # SFT 指令微调（ChatML + loss mask）
│   ├── dpo.py                    # DPO 偏好对齐（policy + 冻结 ref）
│   └── infer.py                  # 推理（KV Cache + 交互式 + SFT 模式）
│
├── configs/                     # YAML 配置继承
│   ├── base.yaml                # 全局默认值（含 sft/dpo 块）
│   ├── nano.yaml                # 40M 配置
│   ├── lite.yaml                # 87M 配置
│   └── pro.yaml                 # 126M 配置
│
├── experimental/                 # 实验性工具
│   └── gleamlm_hf_fixed.py      # HuggingFace 迁移尝试验证
│
├── tools/                       # 评估 + 验证工具
│   ├── eval_ppl.py              # PPL 评估
│   ├── eval_knowledge.py        # 知识评估
│   ├── eval_layer_dropout.py    # 层 dropout 测试
│   ├── generate_samples.py      # 文本样例生成
│   ├── quantize.py              # 模型量化导出
│   ├── check_ckpt.py            # Checkpoint 信息查看
│   ├── quick_run.py             # 快速训练+验证一体化
│   └── verify_paths.py          # 路径验证
│
├── data_tools/                  # 数据获取 & SFT/DPO 数据生成
│   ├── pretrain/                # 预训练数据管线
│   │   ├── download.py          # 多源数据下载
│   │   ├── extract_parquet.py   # Parquet → txt
│   │   ├── pipeline.py          # 一键管道（去重→清洗→SimHash→混合切分）
│   │   └── score.py             # 质量评分
│   ├── sft/                     # SFT 数据生成
│   │   ├── generate.py          # 统一入口（硬编码/API）
│   │   ├── generate_longform.py # 长文 SFT
│   │   ├── generate_multiturn.py# 多轮 SFT
│   │   └── clean_format.py      # 格式清洗
│   ├── dpo/                     # DPO 数据生成
│   │   └── generate_rejected.py # rejected 数据生成
│   └── shared/                  # 共享模块
│       └── api_client.py        # DeepSeek API 客户端
│
├── tests/                       # 核心库测试
│   ├── test_model.py            # 模型前向/反向/KV Cache 测试
│   ├── test_tokenizer.py        # Tokenizer 冒烟测试
│   ├── test_dataset.py          # 数据集和 collate_fn 测试
│   ├── test_sampler.py          # 采样策略 / repetition_penalty 回归测试
│   ├── test_utils_config.py     # 配置加载 / deep merge / CLI override 测试
│   └── test_evaluation.py       # 评估模块测试
│
├── data/
│   ├── nano_data/               # Nano 训练/验证/测试 + .npy 缓存
│   ├── lite_data/               # Lite 训练/验证/测试 + .npy 缓存
│   ├── pro_data/                # Pro 训练/验证/测试 + .npy 缓存
│   ├── nano/                    # Nano SFT/DPO 数据
│   ├── lite/                    # Lite SFT/DPO 数据
│   └── pro/                     # Pro SFT/DPO 数据
│
├── checkpoints/                 # 模型输出（按变体分目录）
│   ├── nano/
│   │   ├── best_model.pt
│   │   ├── sft/
│   │   └── dpo/
│   ├── lite/
│   └── pro/
│
├── docs/                        # 开发文档
├── requirements.txt             # Python 依赖
└── README.md
```

---

## 快速开始

### 环境

- Python 3.10+
- PyTorch 2.5+ with CUDA 12.4
- RTX 4070 Ti 12GB（或同等显存）

```bash
pip install -r requirements.txt
```

### 1. 数据准备（一键管线）

```bash
# 下载原始数据（仅首次）
pip install py7zr kagglehub
python data_tools/download_data.py

# 一键：清洗 → 去重 → QA过滤 → 字符加权配比 → 混合切分
python data_tools/prepare_data.py --input data/raw --output data/nano_data

# 自定义配比（字符占比）
python data_tools/prepare_data.py --ratios 0.30 0.12 0.43 0.15
```

### 2. 预训练

```bash
# 统一入口：--variant 选择变体
python scripts/train.py --variant nano
python scripts/train.py --variant lite
python scripts/train.py --variant pro

# 断点续训
python scripts/train.py --variant nano --load_checkpoint checkpoints/nano/checkpoint_epoch_3.pt

# 监控
tensorboard --logdir ./checkpoints/nano/runs
```

优化器：AdamW（β=0.9,0.95，wd=0.01），BF16 AMP，Cosine Warmup + Decay，Flash Attention（`F.scaled_dot_product_attention`）。首次运行自动 BBPE 分词，后续 mmap 加载 ~1MB。

### 3. 推理

```bash
# 统一 CLI
python scripts/infer.py --model checkpoints/nano/best_model.pt
python scripts/infer.py --model checkpoints/lite/sft/sft_best.pt --sft  # SFT 对话
python scripts/infer.py --model checkpoints/lite/best_model.pt  # 交互模式
```

### 4. SFT 指令微调

```bash
python scripts/sft.py --variant nano
python scripts/sft.py --variant lite --epochs 2 --lr 2e-5
python scripts/sft.py --variant pro
```

### 5. DPO 偏好对齐

```bash
python scripts/dpo.py --variant nano
python scripts/dpo.py --variant lite
python scripts/dpo.py --variant pro
```

### 6. 量化导出

```bash
python tools/quantize.py --input checkpoints/nano/best_model.pt --output checkpoints/nano/model_fp16.pt
python tools/quantize.py --input checkpoints/nano/dpo/dpo_best.pt --output checkpoints/nano/dpo/dpo_fp16.pt
```

### 7. 运行测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 数据集

### 数据来源与清洗

| 数据源 | 原始 | 清洗后 | 保留率 |
|--------|:---:|:---:|:---:|
| 中文维基 | 565万 | 545万 | 96.4% |
| 百度百科 | 214万 | 213万 | 99.8% |
| 新闻 2016 | 202万 | 171万 | 84.5% |
| 社区问答 | 403万 | 92万 | 22.8% |
| **合计** | **1,384万** | **1,021万** | **73.8%** |

### GleamLM-Nano 字符加权配比

各源行均字符差异巨大（新闻 ~752 字/行 vs 维基 ~123 字/行），`prepare_data.py` 自动按字符占比换算行数配比：

| 源 | 目标字符比 | 行均字符 | → 行数配比 |
|---|---|---|---|
| wiki | 30% | 123 | 52.8% |
| baike | 12% | 145 | 17.9% |
| news | 43% | 752 | 12.4% |
| qa | 15% | 192 | 16.9% |

> Nano 最终数据：train 6.48 GB / valid 0.36 GB / test 0.36 GB，~1.2B 训练字符。

### GleamLM-Lite 五源配比

Lite 在四源基础上引入 [Chinese FineWeb Edu](https://huggingface.co/datasets/opencsg/chinese-fineweb-edu)（教育级质量过滤网页文本），数据量从 ~1.2B 提升至 ~4.3B tokens：

| 数据源 | token 估算 | 字符配比 | 文件大小 |
|--------|-----------|:---:|------|
| Chinese FineWeb Edu | ~1.5B | 35% | 5.8 GB |
| 中文新闻 | ~870M | 20% | — |
| 中文维基 | ~870M | 20% | — |
| 百度百科 | ~650M | 15% | — |
| 社区问答 | ~435M | 10% | — |
| **总计** | **~4.3B** | **100%** | **13.85 GB** |

> Chinchilla 最优 ~1.74B tokens（87M × 20），当前 2.5× 超出，保留多 epoch 训练余地。

---

## GleamLM-Lite 训练结果

GleamLM-Lite（87M）预训练完成，并完成了 SFT → DPO 对齐全链路。87M 在中文对话可用性上显著优于 40M 的 Nano 版，但仍受限于参数量，存在事实性幻觉。

| 阶段 | 关键参数 | 结果 |
|------|----------|------|
| 预训练 | lr=4e-4, epochs=2, Cosine, Z-Loss=1e-4, FlashAttn | ~4.3B tokens, 五源混合（含 Chinese FineWeb Edu） |
| SFT 指令微调 | 2300 条 API 蒸馏数据, ChatML+loss mask, lr=2e-5, epochs=2 | 首版含 markdown 标记污染（56% 数据），清洗后重训，最终 loss=2.40 |
| DPO 偏好对齐 | 500 对 chosen/rejected, β=0.1, lr=1e-7, epochs=1 | SFT clean → DPO，loss=0.64，消除自我介绍崩溃 |

> **关键经验**：DPO 的 rejected 必须用 SFT 模型自己生成，不能用预训练基座（方向是反的）。API 数据返回的 markdown 格式化是训练数据里的隐形杀手——56% 的输出含 `**bold**` 标记，需正则清洗后再入训。87M 的可靠输出窗口在 128-150 token，超 164 幻觉骤增。

### 预训练对比

| 参数 | Nano 40M | Lite 87M |
|------|------|------|
| 优化器 | AdamW | AdamW |
| 学习率 | 3e-4 | 4e-4 |
| LR 调度 | Cosine | Cosine（WSD 为后续消融选项） |
| Attention | 手写 | `F.scaled_dot_product_attention` |
| Z-Loss | 无 | 1e-4 |
| Dropout | 0.1 | 0.0 |
| 数据量 | ~1.2B chars | ~4.3B tokens |

### GleamLM-Lite SFT + DPO（87M 对齐）

#### SFT 指令微调

SFT 数据采用 DeepSeek API 蒸馏的 2300 条高质量中文问答（`data/sft_api_new.jsonl`），ChatML 格式。

首版 SFT 训练后，模型输出频繁出现 `**加粗标题**`、`1. 编号列表` 等 markdown 标记——经统计，**56% 的训练数据从 API 获取时就自带 markdown 格式化**。编写正则清洗脚本（去 bold/列表/标题标记）后，输出格式变为干净的纯文本段落。

| 参数 | 值 | 说明 |
|------|-----|------|
| 训练数据 | 2308 条（清洗后） | JSONL，instruction+output |
| 训练轮数 | 2 epochs | lr=2e-5, dropout=0.05 |
| Batch size | 8 | accumulate=4，有效 batch=32 |
| 格式 | ChatML + loss mask | 仅 assistant 部分计算损失 |
| SFT(clean) loss | 2.40 | vs 首版(raw) 2.36，略高但输出质量更好 |

#### DPO 偏好对齐

基于 SFT(clean) 模型，用其自己生成"差回答"作为 rejected，API 的好回答作为 chosen，共 500 对。DPO lr=1e-7，β=0.1，1 epoch。

> **重要教训**：最初用预训练基座生成 rejected——方向完全反了。DPO 的目标是让模型知道"别像你自己那样差，要像 API 那样好"，而不是"别像预训练基座那样胡扯"。**rejected 必须用 SFT 模型自己生成**。

#### 四阶段效果对比

同一 prompt，各阶段生成效果（temperature=0.7, max_new_tokens=128, repetition_penalty=1.15）：

| Prompt | 预训练 | SFT(raw) | SFT(clean) | SFT+DPO |
|--------|--------|----------|------------|---------|
| 介绍一下你自己 | 空白 | 无限重复崩溃 | 有内容但偏题 | 有内容，结构改善 |
| 失眠怎么办 | 空白 | `**markdown**` 标记 | 纯文本，连贯 | 有具体建议 |
| 什么是机器学习 | 空白 | 定义还行 | 定义简洁 | 多领域交叉定义 |
| 北京好玩地方 | 空白 | 城市幻觉 | 重复崩溃 | 有内容提及 |
| 番茄炒蛋 | 空白 | 步骤正确 | 方法略偏 | 分步骤+调料 |

> 87M 在事实知识广度和对话流畅性上明显优于 40M，但幻觉仍然较严重（如"天空为什么是蓝色"会回答"气温相互作用"而非瑞利散射）。建议推理时 max_new_tokens 控制在 128-150，超过 164 幻觉和重复骤增。

---

## GleamLM-Nano 训练结果

### （BBPE 12K + 字符加权四源混合，~40M）

![](./assets/train_loss.jpg)



![](./assets/val.jpg)

训练配置：`batch_size=4, accumulate_grad=16`（等效 64），`label_smoothing=0.1`，`stride=768`，Cosine Warmup + Decay，12GB 显存持续 ~92% 满载。

| Epoch | Train Loss | Val Loss | PPL | PPL↓ | 备注 |
|-------|-----------|----------|-----|------|------|
| 0 | 3.2960 | 2.8064 | 16.55 | — | 语法收敛，生成通顺但内容空洞 |
| 1 | 2.8764 | 2.7045 | 14.95 | -1.60 | 首句沾边，后续漂移 |
| 2 | 2.8053 | 2.6568 | 14.25 | -0.70 | 高频事实固化中 |
| 3 | 2.7655 | 2.6255 | 13.81 | -0.44 | 边际收益递减，改善持续 |
| 4 | 2.7440 | 2.6136 | **13.65** | -0.16 | 训练完成，全程无过拟合 |



**最佳结果**：`val_loss=2.6136`，`val_ppl=13.65`，模型保存至 `./checkpoints`。

> 输出通顺、格式清晰、首句基本沾边。5 个 epoch 全程无过拟合，val_loss 和 ppl 持续下降，边际收益递减但仍未完全收敛。长尾事实知识受限于 40M 参数容量，后续将通过 SFT + DPO 对齐改善。

**Epoch 4 最佳模型生成样例**（temperature=0.5, repetition_penalty=1.1, max_new_tokens=35）：

| 输入 | 输出（节选） |
|------|------|
| `中国有五千年的` | 历史，是中华人民共和国的一部分。（首词正确预测"历史"）... |
| `机器学习是人工智能的` | 一个重要方面。（精准命中常见搭配）... |
| `读书的好处是` | 每个人都会有自己的兴趣爱好和想法，不管你是否喜欢阅读，都可以通过阅读... |
| `世界上最高的山峰是` | 位于中国西藏自治区拉萨市南部的一座山峰，海拔高度1,463米。（地理关联正确）... |

> 模型对高频搭配和常见知识有一定记忆（如"五千年→历史"、"AI→一个方面"），能保持续写方向大致相关。但在长尾知识上仍会发散到无关话题。这是 40M 小模型在纯预训练阶段的物理上限，后续通过 SFT + DPO 对齐可显著改善。

### GleamLM-Nano SFT + DPO（40M 对齐验证）

#### SFT 数据生成

采用 DeepSeek-V4-Pro API 蒸馏生成 10000 条高质量中文指令数据（`data/sft_data.jsonl`），三类配比：

> **API 配置**：如需重新生成数据，需设置环境变量 `DEEPSEEK_API_KEY`（DeepSeek 控制台创建 API Key）。当前仓库已包含生成好的 `data/sft_data.jsonl`，无需额外配置即可直接训练。

| 类别 | 占比 | 条数 | 内容范围 |
|------|:---:|------|----------|
| **A 类 · 通用问答** | 40% | 4000 | 烹饪技巧、家务整理、健康习惯、学习方法、安全科技、旅行出行、生活妙招 |
| **B 类 · 知识回答** | 30% | 3000 | 历史（25 条基础）、地理（19 条）、科学（25 条）、文化（18 条），通过模板扩展至 3000 条 |
| **C 类 · 创作与闲聊** | 30% | 3000 | 描写创作（夕阳、大海、星空等）、情感感悟（孤独、成长、友情等）、日常聊天、观点讨论 |

数据格式为**标准 ChatML**（V4 BBPE 12K 词表原生支持 `<|im_start|>` / `<|im_end|>` 特殊 token）：

```
<|im_start|>system
你是一个乐于助人的AI助手。<|im_end|>
<|im_start|>user
如何煮出一碗好吃的面条？<|im_end|>
<|im_start|>assistant
煮好面条的诀窍：水要多，水开后下面，用筷子拨散防止粘连...<|im_end|>
```

训练时仅对 `assistant` 部分计算 loss（loss mask），确保模型学会回答而非重复问题。

#### SFT 指令微调

```bash
# 从头训练
python scripts/sft.py --variant nano

# 断点续训
python scripts/sft.py --variant nano --resume checkpoints/nano/sft/sft_epoch_1.pt
```

| 参数 | 值 | 说明 |
|------|-----|------|
| 训练数据 | 10000 条 | JSONL 格式，ChatML 包装 |
| 训练轮数 | 3 epochs | 避免过拟合 |
| 学习率 | 5e-6 | 预训练的 1/60，保护通用能力 |
| Batch size | 8 | accumulate=4，有效 batch=32 |
| 格式 | ChatML + loss mask | 仅 assistant 部分计算损失 |
| 预计耗时 | ~55 分钟 | 单卡 12GB |
| 续训 | `--resume PATH` | 从 checkpoint 恢复 optimizer/scheduler/scaler 状态续训 |

- **ChatML + loss mask**：BBPE 原生支持 `<|im_start|>`（ID=1）、`<|im_end|>`（ID=2），无需格式绕过
- **评估方式**：对比微调前后对同一 prompt 的生成质量，检验是否从"续写"转为"直接回答"

**SFT 训练结果**（lr=5e-6, epochs=3）：

| Epoch | train_loss | 说明 |
|-------|-----------|------|
| 0 | 3.3279 | 初始状态，loss 与预训练末期接近 |
| 1 | ~2.8 | 开始适应 ChatML 格式 |
| 2 | ~2.2 | 对话格式基本学会 |

> 提升 10 倍学习率后，模型仅需 3 个 epoch 即可掌握对话格式。loss 从 3.3 降至 2.2，说明模型有效学习了指令跟随能力。

**SFT 后生成样例**（`--sft --temperature 0.7 --repetition_penalty 1.15 --max_new_tokens 128`）：

| Prompt | 模型输出 | 评价 |
|--------|----------|------|
| 你好，请介绍一下你自己 | 如果你是个人，建议你先学会分析别人的优劣... | 格式正确（直接回答），但内容偏移到人生建议 |
| 什么是机器学习 | 机器学习是指将信息传递给机器人，从而实现机器学习的一种方法... | 方向沾边，夹杂大量无关细节 |
| 请用一句话描述北京的秋天 | 北京是世界上最大的热带气旋生物多样性保护区... | 完全幻觉，缺乏事实锚点 |
| 写一首关于春天的五言诗 | 春天是温暖的季节，是安静的季节... | 没写成诗，只是在描述春天 |
| 请解释一下什么是光合作用 | 光合作用是一种天然的氧化物，分子量约2000万个太阳质量... | 方向沾边，但事实严重错误 |

> **结论**：SFT 成功让模型从"续写"转为"直接回答"，格式层面完全达标。但 40M 参数容量不足以支撑事实性知识的精准记忆——这是小模型的物理上限，而非训练问题。后续通过 DPO 对齐可进一步提升安全性和回答质量。

#### DPO 偏好对齐

```bash
python scripts/dpo.py --variant nano --model_path checkpoints/nano/sft/sft_best.pt
```

| 参数 | 值 | 说明 |
|------|-----|------|
| 训练数据 | 150 对 chosen/rejected | SFT 模型生成 rejected（回答同一问题但答错），DeepSeek 输出作为 chosen |
| 训练轮数 | 1 epoch | β=0.1，学习率 1e-7 |
| DPO loss | 0.89 → 0.79 | 偏好信号有效学习，loss 下降 11% |
| 预计耗时 | ~2 分钟 | 150 对数据，batch=2×2 |

**DPO 后生成样例**（`--sft --temperature 0.7 --repetition_penalty 1.15 --max_new_tokens 128`）：

| Prompt | SFT 后 | DPO 后 | 改善 |
|--------|--------|--------|:---:|
| 北京秋天 | 北京是世界上最大的热带气旋生物多样性保护区 | 落叶遍野、金黄如雪、红得让人心旷神怡 | 🟢 |
| 光合作用 | 天然的氧化物，分子量约2000万个太阳质量 | 生物体生长发育和光照时间变化 | 🟢 |
| 自我介绍 | 如果你是个人，建议先学会分析别人的优劣 | 练字孩子的成长故事 | ⬜ 叙事更连贯但仍跑题 |
| 机器学习 | 将信息传递给机器人 | 操作系统/计算机模块分离 | ⬜ 方向修正，细节仍幻觉 |
| 五言诗 | 春天是温暖的季节，是安静的季节 | 描写+引经据典（三国/水浒） | ⬜ 更有文采，但未成诗 |

> **DPO 结论**：最显著的效果是纠正方向性错误（不再说北京是保护区、光合作用有太阳质量）。但 40M 参数注定无法记住精准事实。GleamLM-Nano 全链路（预训练→SFT→DPO）至此收尾，下一阶段转向 GleamLM-Lite（80M）预训练。

---

## 版本路线

| 版本 | 参数量 | 定位 | 状态 |
|------|--------|------|------|
| GleamLM-Nano | ~40M | 教学入门 / 单卡 12GB | ✅ 已完成 |
| GleamLM-Lite | ~87M | 消融实验平台 / FFN 3.4× | ✅ 已完成 |
| GleamLM-Pro | ~126M | 科研进阶 / 18L×768d / BBPE 12K | 🔨 开发中 |
| GleamLM-0.6B | ~0.6B | 工业级验证 / 37L×1024d / BBPE 24K | 📋 寻求合作 |

---

## 安全提示

所有 checkpoint 加载使用 `torch.load(weights_only=False)`，这是加载优化器状态、Python 对象（如 argparse Namespace）等非张量数据的必要条件。**请勿加载来源不明的 checkpoint 文件**，否则存在 pickle 反序列化攻击风险。仅加载自己训练或可信来源的 checkpoint。

---

## 许可证

Apache License 2.0
