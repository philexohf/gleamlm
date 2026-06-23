# 烁珑 GleamLM —— 面向教育和研究的小型语言模型

<img src="./assets/GleamLM.png" style="zoom: 25%;" />

## 项目简介

纯 PyTorch 从零实现，零 HuggingFace 依赖，覆盖 **四源中文数据管线**（下载→清洗→去重→字符加权配比）→ **BBPE 分词器训练**（自研BBPE，零外部依赖）→ **Decoder-only 模型**（SwiGLU / GQA / RoPE / QK-Norm）→ **AMP + DDP 训练**（断点续训保存 optimizer/scheduler/scaler 全量状态）→ **SFT / DPO 对齐**（ChatML + loss mask）→FP16量化 → **KV Cache 流式推理**全链路。GleamLM-Nano模型单卡 12GB 显存即可训练，Windows/Linux 双平台兼容。

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

### 模型规格（GleamLM-Nano ~40M）

| 参数 | 值 |
|------|-----|
| 上下文窗口 | 1024（RoPE 支持外推至 2048/4096） |
| 词表大小 | 12,003（BBPE 自研） |
| 网络层数 | 12 |
| 模型维度 | 512 |
| QK-Norm | ✅ |
| 查询头 / KV 头 | 8 / 4（GQA） |
| SwiGLU 中间维度 | 1365 |
| Dropout | 0.1 |
| 参数量 | **~40M**（Embed 6.1M + Transformer 34.6M） |

---

## 项目结构

```
GleamLM/
├── gleamlm_train.py           # 预训练脚本（AMP + DDP + Cosine + 断点续训）
├── gleamlm_infer.py           # 推理脚本（KV Cache + 交互式对话）
├── gleamlm_dataset.py         # 数据集（滑动窗口 + memmap 预分词）
├── gleamlm_sft.py             # SFT 指令微调（ChatML + loss mask）
├── gleamlm_dpo.py             # DPO 偏好对齐（policy + frozen reference）
├── gleamlm_quantize.py        # FP16 量化导出
├── quick_test_sft_dpo.py      # SFT+DPO 全链路快速验证
│
├── models/
│   ├── gleamlm_model.py       # 模型定义（RMSNorm / RoPE / GQA / SwiGLU）
│   └── gleamlm_config.py      # 全局配置 + 路径常量
│
├── tokenizer/
│   └── bbpe_tokenizer.py      # V4 BBPE 分词器（607行，纯 Python 零依赖）
│
├── inference/
│   ├── sampler.py             # Temperature / TopK / TopP 采样
│   └── streamer.py            # KV Cache 流式生成
│
├── utils/
│   └── logging.py             # 统一日志模块
│
├── tools/
│   ├── prepare_data.py        # 一键数据管线（清洗→去重→混合→切分）
│   ├── build_dataset.py       # 流式多源混合 + train/valid/test 切分
│   ├── clean_text.py          # 文本清洗（长度/语言/广告过滤）
│   ├── dedup_text.py          # 去重（MD5 exact / prefix）
│   ├── filter_qa.py           # QA 专项过滤
│   ├── download_data.py       # 多源数据下载
│   ├── eval_ppl.py            # PPL 评估工具
│   └── check_ckpt.py          # Checkpoint 检查
│
├── scripts/
│   ├── generate_sft_data.py   # DeepSeek API 蒸馏 SFT 数据
│   ├── generate_rejected.py   # 基模型生成 DPO rejected 样本
│   └── clean_sft_data.py      # SFT 数据格式清洗
│
├── tests/
│   ├── test_tokenizer.py      # Tokenizer 冒烟测试
│   ├── test_model.py          # 模型前向/反向/KV Cache 测试
│   └── test_dataset.py        # 数据集和 collate_fn 测试
│
├── data/
│   ├── raw/                   # 原始语料 + 清洗后文本
│   └── splits/                # train/valid/test + .npy 预分词缓存
│
├── checkpoints/               # 模型检查点 + TensorBoard 日志
├── requirements.txt           # Python 依赖（5 个包）
├── requirements-dev.txt       # 开发依赖
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
python tools/download_data.py

# 一键：清洗 → 去重 → QA过滤 → 字符加权配比 → 混合切分
python tools/prepare_data.py --input data/raw --output data/splits

# 自定义配比（字符占比）
python tools/prepare_data.py --ratios 0.30 0.12 0.43 0.15
```

### 2. 预训练

```bash
python gleamlm_train.py --data_dir ./data/splits --epochs 5

# 断点续训
python gleamlm_train.py --data_dir ./data/splits --load_checkpoint checkpoints/checkpoint_epoch_3.pt

# 监控
tensorboard --logdir ./checkpoints/runs
```

| 关键参数 | 默认值 | 说明 |
|----------|--------|------|
| `--epochs` | 5 | 训练轮数 |
| `--batch_size` | 8 | Micro-batch（显存安全） |
| `--accumulate_grad` | 8 | 梯度累积（有效 batch=64） |
| `--lr` | 3e-4 | 峰值学习率 |
| `--label_smoothing` | 0.1 | 标签平滑 |

优化器：AdamW（β=0.9,0.95，wd=0.01），BF16 AMP，Cosine Warmup + Decay。首次运行自动 BBPE 分词（~35 分钟），后续 mmap 加载 ~1MB。

### 3. 推理

```bash
# 单次生成
python gleamlm_infer.py --model checkpoints/best_model.pt --prompt "人工智能是"

# 交互模式
python gleamlm_infer.py --model checkpoints/best_model.pt

# 调整采样
python gleamlm_infer.py --model checkpoints/best_model.pt --temperature 0.8 --top_k 50 --top_p 0.9
```

### 4. SFT 指令微调

```bash
python gleamlm_sft.py --data_path ./data/sft_data.jsonl --model_path ./checkpoints/best_model.pt
```

### 5. DPO 偏好对齐

```bash
python gleamlm_dpo.py --data_path ./data/dpo_data.jsonl --model_path ./checkpoints/sft/sft_best.pt
```

### 6. 量化导出

```bash
python gleamlm_quantize.py --input checkpoints/best_model.pt --output checkpoints/model_fp16.pt
```

### 7. 运行测试

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## 数据集

### 数据来源与清洗

| 数据源 | 原始 | 清洗后 | 保留率 | 来源 |
|--------|:---:|:---:|:---:|------|
| 中文维基 | 565万 | 545万 | 96.4% | [modelscope](https://www.modelscope.cn/datasets/caoaolong/zhwiki) |
| 百度百科 | 214万 | 213万 | 99.8% | 百度网盘（提取码 `bwvb`） |
| 新闻 2016 | 202万 | 171万 | 84.5% | [百度网盘](https://pan.baidu.com/s/1LJeq1dkA0wmYd9ZGZw72Xg)（提取码 `film`） |
| 社区问答 | 403万 | 92万 | 22.8% | [Kaggle](https://www.kaggle.com/datasets/terrychanorg/webtext2019zhjsonwebtext2019zh) |
| **合计** | **1,384万** | **1,021万** | **73.8%** | — |

最终切分为 `train.txt`（6.48 GB，90%）/ `valid.txt`（0.36 GB，5%）/ `test.txt`（0.36 GB，5%），合计 7.20 GB。

### GleamLM-Nano V4 字符加权配比

各源行均字符差异巨大（新闻 ~752 字/行 vs 维基 ~123 字/行），`prepare_data.py` 自动按字符占比换算行数配比：

| 源 | 目标字符比 | 行均字符 | → 行数配比 |
|---|---|---|---|
| wiki | 30% | 123 | 48.7% |
| baike | 12% | 145 | 16.5% |
| news | 33% | 752 | 8.8% |
| qa | 25% | 192 | 26.0% |

---

## 训练结果

### GleamLM-Nano V3（4 源混合 1.2B tokens，label_smoothing=0.1）

| Epoch | Val Loss | PPL | PPL↓ |
|-------|----------|-----|------|
| 0 | 3.8242 | 45.80 | — |
| 1 | 3.7038 | 40.60 | -5.20 |
| 2 | 3.6490 | 38.44 | -2.16 |
| 3 | 3.6143 | 37.12 | -1.32 |
| 4 | 3.5887 | 36.19 | -0.93 |
| 5 | 3.5702 | 35.55 | -0.64 |
| 6 | 3.5585 | 35.13 | -0.42 |
| 7 | 3.5532 | **34.93** | -0.20 |

> V3 三项关键改进：（1）4 源混合替代单源 Wiki；（2）`label_smoothing=0.1`；（3）`stride=768` 降低过拟合。8 epoch 全程无过拟合，PPL 最终 34.93。

### GleamLM-Nano V3 SFT + DPO（39M 对齐验证）

- **SFT**：995 条 DeepSeek 蒸馏数据，1 epoch，ChatML 格式 + loss mask，模型学会直接回应问题
- **DPO**：150 对 chosen/rejected，DPO loss 0.95 → 0.60，流程验证通过

---

## 版本路线

| 版本 | 参数量 | 定位 | 状态 |
|------|--------|------|------|
| GleamLM-Nano | ~40M | 教学级 / 单卡资源 | 已完成 |
| GleamLM-Lite | ~80M | 教学级 / 服务器资源 | 规划中 |
| GleamLM-Pro | ~126M | 科研进阶 / 服务器资源 | 规划中 |
| GleamLM-0.6B | ~0.6B | 工业级验证 / 算力集群 | 寻求合作 |

---

## 许可证

Apache License 2.0
