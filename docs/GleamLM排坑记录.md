# GleamLM 排坑记录

记录项目开发中遇到的问题及其修复方案。

---

## 1. TextStreamer 字节缓冲区解码吞文本

**现象**：流式生成只输出第一个 chunk（约 4 个 token），后续全部消失。

**根因**：[`streamer.py`](file:///h:/MyGitHub/GleamLM/gleamlm/inference/streamer.py#L78-L81) 正常解码路径中：

```python
text = byte_buffer.decode('utf-8')
new_part = text[len(total_decoded):]
total_decoded = text          # ← 设为当前（只有新 4 token）的文本
byte_buffer = bytearray()
```

第 2 轮循环：`total_decoded` 是上一轮 4 token 的短文本，但 `byte_buffer` 已被清空后新值也是短文本。`text[len(total_decoded):]` 切出空字符串，后续所有文本被吞。

**修复**：清空 `byte_buffer` 时同时重置 `total_decoded = ""`，因为 streamer 每次 yield 的是独立增量而非累积文本。

---

## 2. `<|endoftext|>` 在 base 模式提前截断生成

**现象**：base 模型推理输出很短（几个字就停），但实际生成了完整 100 个 token。

**根因**：模型生成 `<|endoftext|>`（训练数据中的文档分隔符），`generate_text()` 把它当作硬停止信号，立即 `return`。但 base 模型中 `<|endoftext|>` 只是文本分隔符，不是 EOS token。

**修复**：
- `generate_text()` 新增 `stop_on_endoftext` 参数，默认 `False`
- `infer.py` 仅在 SFT 模式（`--sft`）传 `stop_on_endoftext=True`
- Base 模式正常生成不受影响

---

## 3. infer.py 切片逻辑与 streamer 增量输出不匹配

**现象**：`<|endoftext|>` 显示为 `doftext|>` 等碎片。

**根因**：streamer 修复后每次 yield 独立增量，但 `infer.py` 仍用 `chunk[prev_len:]` 假设累积文本，导致从错误位置切片，吞掉了 chunk 开头字符。

**修复**：`infer.py` 直接使用 `chunk`，移除 `prev_len` 和切片逻辑，用 `generated_text += chunk` 累积完整文本。

---

## 4. Flash Attention 内存连续性隐患

**现象**：潜在性能损失。`expand()` + `reshape()` 后 K/V 张量可能内存不连续，`F.scaled_dot_product_attention` 检测到非连续输入后静默降级到普通 attention。

**修复**：[`model.py` GQA Flash 路径](file:///h:/MyGitHub/GleamLM/gleamlm/models/model.py) K_fa/V_fa 后追加 `.contiguous()`，确保 Flash Attention kernel 正常启用。

---

## 5. 训练进度条 step 重复显示

**现象**：梯度累积时 step 每 N 个 batch 才变一次，进度条看起来 step 卡住不动。

**根因**：`set_postfix` 在每个 batch 都执行，但 `global_step` 仅在梯度更新时递增。

**修复**：将 `set_postfix` 移到梯度累积块内，只在 `optimizer.step()` 后更新。

---

## 6. gleamlm 包安装方式导致源码修改不生效

**现象**：修改 `streamer.py` 后运行 `infer.py` 仍报旧代码错误。

**根因**：gleamlm 通过 `pip install` 安装到 site-packages，导入的是安装副本而非源码。

**解决方案**：改为 editable 开发模式：
```powershell
pip install -e .
```
此后源码修改即时生效，无需手动同步。

---

## 7. 训练前首次分词内存爆满（~60-70 GB）

**现象**：首次训练时，`LMDataset` 将文本全量读入内存做分词，Python list 每个 int ~28 字节，远超设备内存。

**处理**：
1. 首次运行 `LMDataset.__init__` 分词 → 保存 `train_ids.npy` → `del` 释放
2. 后续训练直接 `np.load(..., mmap_mode='r')`，内存几乎不占

**建议**：若首次分词内存峰值仍无法接受，可改为流式分词。

---

## 8. DDP 多进程竞争分词缓存

**现象**：多卡训练首次运行时，所有 rank 同时进入分词逻辑，多个进程竞争写入同一 `.npy` 文件。

**修复**：[`dataset.py`](file:///h:/MyGitHub/GleamLM/gleamlm/dataset/dataset.py) 改为仅 rank 0 执行分词和 `np.save`，其他 rank 通过 `dist.barrier()` 等待。

---

## 9. CLI 参数覆盖绕过配置校验

**现象**：`--d_model 1` 等非法 CLI 值不会被 `_validate_config` 捕获。

**根因**：[`config.py`](file:///h:/MyGitHub/GleamLM/gleamlm/utils/config.py) `load_config()` 先校验再覆盖。

**修复**：在 `_apply_overrides` 之后新增一次 `_validate_config(cfg_dict)` 调用。

---

## 10. DPO 训练参数计数重复 tied weights

**现象**：日志显示参数量比实际大（Nano 多约 6M）。

**根因**：`sum(p.numel() for p in model.parameters())` 会重复计算 `lm_head` 和 `token_embed` 共享的权重。

**修复**：改用 `model.get_num_params()[0]`，该方法通过 `id(p)` 去重。

---

## 11. DPO PAD_ID 硬编码与 tokenizer 不一致

**现象**：`PAD_ID=0` 硬编码，但 tokenizer 的 `pad_id` 在 256+。

**修复**：`DPODataset.__getitem__` 返回 `_pad_id`，`dpad_collate` 从 batch 读取。

---

## 12. 多选题评估重复计算 prompt

**现象**：`_mc_generate` 对 A/B/C/D 四个选项各自做完整前向传播，prompt 部分重复计算 4 次。

**修复**：[`benchmark.py`](file:///h:/MyGitHub/GleamLM/gleamlm/evaluation/benchmark.py) 改为先预填充一次 prompt 得到 KV Cache，再对每个选项仅做增量推理。

---

## 13. Epoch 0 推理 token 循环退化及采样参数优化

**现象**：Epoch 0 最佳模型（val_ppl=11.77）在推理时出现严重 token 循环重复：4 个测试 prompt 中 3 个出现退化。

**根因**：Epoch 0 模型概率分布尚未平滑，某些 token 的概率过于尖锐。低温度和默认 repetition_penalty=1.0 放大了这一问题。

**修复**：repetition_penalty 从 1.0 → **1.1**。temperature 保持 Lite 0.8 / Nano 1.0。

**验证**：同组 prompt 用优化后参数重新推理，token 循环完全消失。

---

## 14. 数据集内存爆炸 → numpy memmap

**现象**：训练启动后进程卡死，数据集将所有样本存储为 `torch.Tensor` 列表，8万+ 样本 × 1025 tokens × 8 bytes = 7.3 GB，远超内存。

**解决**：改用 numpy memmap 磁盘映射：
- 首次加载时分词并保存为 `.npy`（`np.save`）
- 后续加载使用 `np.load(ids_file, mmap_mode='r')`，内存占用降至 ~1MB
- `__getitem__` 直接切片 memmap 转为 tensor

```python
ids_array = np.array(all_ids, dtype=np.uint32)
np.save(ids_file, ids_array)
self.all_ids = np.load(ids_file, mmap_mode='r')  # 约 1MB

def __getitem__(self, idx):
    ids = self.all_ids[start:end].astype(np.int64)
    return torch.from_numpy(ids)
```

**教训**：大规模数据集用 memmap，不要将全部样本存为 tensor。这是 GleamLM 数据管线的核心设计，至今生效。

---

## 15. DataLoader 多进程在 Windows 上卡死

**现象**：`num_workers=4` 时训练数据加载后卡住不动。

**原因**：Windows 使用 `spawn` 多进程模式，每个 worker 需要重新 import 所有模块并打开 memmap 文件，启动极慢。

**解决**：单卡训练时 `num_workers=0`（主进程加载），多卡 DDP 时才用多进程。

**教训**：Windows 上的 DataLoader 多进程尽量避免，memmap + 单进程效率足够。

---

## 16. batch/seq 显存悬崖

**现象**：batch=16, seq=1024 时，单步前向 62s + 反向 180s = 242s。batch=8 时仅 0.23s。

**根因**：batch=16, seq=1024 时中间激活（注意力权重 8层×8头×[16,1024,1024]）占据 ~3.6 GB 显存，加上模型权重和优化器状态超过 12GB 可用空间，CUDA 回退到系统内存交换。

**解决**：`batch_size=4, accumulate_grad=16` — 有效 batch=64 但避开显存悬崖。

**教训**：从小 batch 起步，逐级放大，不要直接上大 batch。Flash Attention 可避免存储完整注意力矩阵。

---

## 17. RoPE 实现优化：复数 → 实数

**原因**：原始 RoPE 使用 `torch.view_as_complex` 复数运算，每次需要 `.float()` 类型转换。

**解决**：改用实数运算，通过 `rotate_half` 函数完成旋转：

```python
def _rotate_half(x):
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_emb(xq, xk, cos, sin):
    return xq * cos + _rotate_half(xq) * sin, xk * cos + _rotate_half(xk) * sin
```

实数版本比复数版本快 2-3 倍，且避免了 FP32/FP16 类型转换。当前 `model.py` 仍使用此实现。

---

## 18. torch.load 的 weights_only 兼容性

**现象**：PyTorch 2.5.1 在部分 Windows 环境下，`torch.load(path, weights_only=False)` 报错 `weights_only is an invalid keyword argument for Unpickler()`。

**原因**：PyTorch 2.5.1 的 `weights_only` 参数被错误透传给底层的 `pickle.Unpickler`。

**修复**：

```python
# 兼容写法
try:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
except TypeError:
    checkpoint = torch.load(model_path, map_location=device)
```

**注意**：PyTorch 2.6+ 已修复，但向后兼容仍需保留 `try/except`。

---

## 19. Windows 终端 GBK 编码导致打印失败

**现象**：推理输出包含某些 Unicode 字符时 `print()` 报 `UnicodeEncodeError: 'gbk' codec can't encode`。

**解决**：
```python
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
```
所有推理脚本（`infer.py`、`generate_samples.py`）均已加入此修复。

---

## 20. LambdaLR 学习率乘数语义 — 实际 LR 仅预期的 1/3333

**现象**：训练多 epoch 后模型接近随机（PPL ~53,000），LR 监控显示 scheduler 返回 8e-8（预期 3e-4）。

**原因**：`get_lr_cosine()` 返回**绝对值**（如 3e-4），但 `torch.optim.lr_scheduler.LambdaLR` 把它当作**乘数**——实际 LR = base_lr × 返回值 = 3e-4 × 3e-4 = 9e-8，差了 3,333 倍。

**修复**：
```python
# 错误（返回绝对值 → LambdaLR 二次相乘）
def get_lr_cosine(step, total_steps, peak_lr, ...):
    return peak_lr * step / max(1, warmup_steps)   # 返回 3e-4

# 正确（返回乘数 0~1）
def get_lr_cosine(step, total_steps, warmup_ratio=0.01, min_lr_ratio=0.1):
    return step / max(1, warmup_steps)              # 返回 0→1
```

**教训**：**LambdaLR 的 lambda 永远返回 0~1 范围的值**，不在函数内乘以峰值学习率。这是项目早期最严重的 bug——此前所有训练都在错误 LR 下进行，权重基本无效。当前 `torch_utils.py` 的 `get_lr_cosine` 和 `get_lr_wsd` 均严格遵守此约束。

---

## 21. 单源数据天花板 — PPL 不降 + 生成持续 token 循环

**现象**：仅用 0.46B tokens 纯中文维基百科训练，epoch 4→5 PPL 完全不降（45 左右停留），生成始终严重 token 循环。

**根因**：不是训练超参问题——是数据量不足 + 信息多样性太低。Chinchilla 最优需要 0.78B tokens，纯维基百科的句式高度模板化，模型学到的语言模式本身就是重复的。

**解决**：V3 多源混合数据（维基 0.46B + 新闻 0.35B + 百科 0.25B + 问答 0.14B，共 1.2B tokens）。多源加入后 PPL 立刻从 45 降到 ~35。Lite 87M 进一步引入 Chinese FineWeb Edu 扩展到 4.3B tokens。

**教训**：数据多样性的重要性远超训练超参调优。模型质量的上限由数据决定，而非架构。

---

## 22. RoPE _rotate_half 维度配对错误

**时间**：2026-06-16 代码审查发现。已影响 V1/V2/V3 全部训练。

**根因**：`precompute_freqs_cis` 生成的 cos/sin 是前后半分重复排列（dim 0 与 dim 32 共享同一频率 θ₀），但旧 `_rotate_half` 用**奇偶交替拆分**——dim 0 与 dim 1 配对，用了不同频率，破坏了 2D 旋转几何。

```python
# 错误（奇偶交替）→ dim k 配 dim k+1，频率不同
x1, x2 = x[..., ::2], x[..., 1::2]

# 正确（前后半分）→ dim k 配 dim k+d/2，频率相同
d2 = x.shape[-1] // 2
x1, x2 = x[..., :d2], x[..., d2:]
```

**教训**：RoPE 正确性取决于 `_rotate_half` 配对逻辑与 `precompute_freqs_cis` 频率排列一一对应。这类 bug 不会报错——模型仍能训练、仍能收敛——但位置编码在数学上已经偏离了 RoPE 论文设计。

当前 `model.py` 的 `_rotate_half` 使用正确的"前后半分"拆分。

---

## 23. KV Cache 生成时 RoPE 位置偏移 — 新 token 始终用位置 0

**时间**：2026-06-16 代码审查发现。已影响 V3 全部生成推理。

**根因**：逐 token 生成阶段，`apply_rotary_emb` 总是取 `cos[:1]`（`seq_len=1`），对应于位置 0。但新 token 的实际位置应是 KV Cache 长度——第 11 个 token 应取位置 10，而非 0。

```python
# 修改前 — 生成阶段永远用位置 0
cos = cos[:seq_len]   # cos[:1] → 位置 0

# 修改后 — 根据 KV Cache 偏移正确切取
cos = cos[offset:offset + seq_len]
```

**教训**：KV Cache 推理的正确性有两个维度——张量拼接和位置编码偏移量。短 prompt 测不出位置偏移问题，必须用长 prompt + 长生成验证。

当前 `model.py` 的 `apply_rotary_emb` 已增加 `offset` 参数，`GroupedQueryAttention.forward` 从 `past_kv[0].size(2)` 获取实际偏移。

---

## 24. PPL 评估 ignore_index 遗漏 — 评估结果系统性偏高

**现象**：训练脚本与独立评估脚本对同一模型给出的 PPL 不一致。

**根因**：`CrossEntropyLoss` 缺少 `ignore_index`。padding token 的 loss 被计入分子（`total_loss`），但分母 `total_tokens` 排除了 padding → avg_loss 被高估 → PPL 虚高。

```python
# 错误
criterion = nn.CrossEntropyLoss(reduction='sum')
total_tokens += (target_ids != 0).sum()  # 分子含 padding，分母不含

# 正确
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='sum')
total_tokens += (target_ids != pad_token_id).sum()
```

**教训**：`CrossEntropyLoss` 的 `ignore_index` 不是可选项——只要数据有 padding 就必须设置。一个遗漏参数让评估体系全盘失效。

---

## 25. training 后 evaluate() CUDA 僵死 — 进程静默挂起 36 小时

**现象**：Lite 87M epoch 1 训练完毕后，`evaluate()` 不再输出任何日志或 TensorBoard 事件。进程 PID 27832 存活但 CPU 零增长，持续 36 小时没有结束，GPU 被占用但未报错。

**定位过程**：
1. TB 日志显示训练已完成（step 96,150，LR 衰减至 0）
2. `checkpoint_epoch_1.pt` 未落盘 — 卡在 save 前的 `evaluate()` 调用
3. epoch 0 的 `evaluate()` 成功（当时 CUDA 内存干净）
4. 进程 CPU 持续 7.3h（89% 来自训练，此后零增长），RAM 18.5 GB

**根因**：经过 96,150 步梯度累积 + AMP + optimizer 状态后，CUDA 内存高度碎片化。第 1 个 eval batch 触发隐式 CUDA 错误（非显式 OOM，而是碎片导致分配延迟），PyTorch Windows 版未正确抛出异常，进程在后续 `cuda.synchronize()` 处永久阻塞。

**修复**（`train.py` evaluate 函数）：
```python
# 修改后
@torch.no_grad()
def evaluate(model, val_loader, eval_criterion, device, pad_token_id=0, world_size=1):
    torch.cuda.empty_cache()    # 清除碎片，防 CUDA 死锁
    model.eval()
    ...
```

同时增加 tqdm 进度条使 eval 进度可见，避免误判为僵死。

**教训**：
1. 长时间训练后 eval 前必须 `torch.cuda.empty_cache()` — 碎片导致的 CUDA 错误不会抛异常，只会死锁。
2. 任何长循环都必须有进度输出，否则无法区分"慢"和"死"。
3. `pin_memory=True` + `num_workers=0` 组合在碎片化严重时加剧风险（已在 [#1](?)#1) 修复）。
 4. epoch 1 的 GPU 权重全部丢失（未落盘），但由于训练已收敛（loss 3.38 → 3.36），实际无损失。

---

## 26. to_namespace NO_PREFIX 键冲突 — sft/dpo 覆盖 training 参数

**现象**：CLI `--variant nano` 训练时，实际 LR=1e-7（预期 4e-4），batch=2（预期 8），seq_len=512（预期 1024）。训练日志显示各项参数均为错误值。

**根因**：[`config.py`](file:///h:/MyGitHub/GleamLM/gleamlm/utils/config.py) `to_namespace` 函数将 `_NO_PREFIX_SECTIONS`（training / lr / model / data / advanced / sft / dpo）的键展平到同一层级。`training.lr`、`sft.lr`、`dpo.lr` 都映射到同一个 `lr` 键，后遍历到的 section（dpo 的 `lr: 1e-7`）直接覆盖前面的 section（training 的 `lr: 4e-4`）。

其他冲突键包括：`epochs`（training=1 vs dpo=1 vs sft=3）、`batch_size`（training=8 vs sft/dpo=2/4）、`accumulate_grad`、`warmup_ratio`、`max_seq_len` 等。

```
加载顺序: training.lr=4e-4 → model.lr 不存在 → ... → sft.lr=1e-7 → dpo.lr=1e-7
最终 result["lr"] = 1e-7    ← 错误！应该是 training 的 4e-4
```

**修复**：
```python
# 错误（后到覆盖先到）
result[k] = v

# 正确（先到先得，training 值不受后续 section 覆盖）
result.setdefault(k, v)
```

**教训**：YAML config 展平到 flat namespace 时，同名键的优先级由展平顺序决定，不是由语义决定。`setdefault` 确保最早出现的 section 获胜。

---

## 27. collate_fn 非连续张量 + view(-1) 崩溃 — 错误包装为 NaN

**现象**：训练脚本 `scripts/train.py` 运行时报 `[FATAL] NaN/Inf loss at step 0`，但完全相同的 standalone 测试脚本正常。实际异常被 NaN 报告掩盖。

**定位过程**：
1. 模型 standalone forward pass — 正常，无 NaN
2. 模型 + backward + optimizer step — 正常
3. collate_fn + DataLoader + 完整训练流程 — 崩溃
4. 追踪到 `target_ids.view(-1)` 抛出 `RuntimeError: view size is not compatible with input tensor's size and stride`

**根因**：`collate_fn`（[`dataset.py`](file:///h:/MyGitHub/GleamLM/gleamlm/dataset/dataset.py)）中：

```python
batch_tensor = torch.stack(padded)     # (B, max_len)
input_ids = batch_tensor[:, :-1]       # (B, max_len-1), stride=(max_len, 1) 非连续！
target_ids = batch_tensor[:, 1:]       # (B, max_len-1), stride=(max_len, 1) 非连续！
```

切片操作 `[:, :-1]` 返回的是 view，stride 不是标准的行优先排列。传入训练循环后 `target_ids.view(-1)` 因 stride 不兼容而抛出 RuntimeError，但异常发生在 `CrossEntropyLoss` 内部被包装。

**修复**：
1. `collate_fn` 返回值加 `.contiguous()` 确保内存连续
2. `base_trainer.py` 中所有 `.view(-1, N)` → `.reshape(-1, N)` 增加防御层

**教训**：PyTorch 切片操作产生非连续张量，必须加 `.contiguous()` 或使用 `.reshape()`（自动处理非连续情况）再传给需要 contiguous input 的操作（如 `view()`、`CrossEntropyLoss` 内部展开）。

---

## 28. nano bf16 配置缺失 — 修复 #26/#27 后仍 NaN

**现象**：修复 #26、#27 后训练仍然 `NaN at step 0`。

**根因**：新版 `safe_autocast` 增加了 `dtype` 参数（相比 v0.2.1 写死了 `dtype=torch.bfloat16`），用于兼容不支持 bf16 的老 GPU（GTX 1080/2080 等）。[`base_trainer.py`](file:///h:/MyGitHub/GleamLM/gleamlm/training/base_trainer.py) 据此决定 AMP 精度：

```python
amp_dtype = torch.bfloat16 if getattr(args, "bf16", False) else torch.float16
```

nano 配置 `bf16: false` → `amp_dtype = torch.float16` → GPU 上启用 FP16 混合精度。FP16 最大值 ~65504，40M 模型（12 层 × 512d）深层激活值容易超限 → logits = NaN → loss = NaN。BF16 和 FP32 具有相同的指数位宽（8 bits），值域完全一致，不会溢出。

**注意**：本地测试环境为 Windows CPU，`safe_autocast(dtype=torch.float16)` 在 CPU 上不匹配 bf16 检查分支，落到 FP32 直通（`yield`），因此此 bug 在本地测试环境无法重现。对比 v0.2.1 lite 时该版本 `bf16: true`，从未触发此问题。`dtype` 参数本身是正当的 GPU 兼容性设计，不是 bug——问题在于 nano 配置错用了 `bf16: false`。

**修复**：`nano.yaml` 中 `bf16: false` → `bf16: true`。

**教训**：`bf16` 配置应该按 GPU 代数默认：现代 GPU（A100/H100/RTX30+）默认 `true`，老 GPU 才需要 FP16。CPU 训练永远走 FP32，不受此配置影响。

---

## 29. Nano 缺少 Z-Loss 防御 — 最终仍 NaN，standalone 正常

**现象**：修复 #26-#28 后，`scripts/train.py` 仍 `NaN at step 0`。但用完全相同配置的 standalone 脚本（`debug_train2.py`）却正常运行，loss 从 9.90 正常下降。

**根因**：项目文档明确指出 Z-Loss（`1e-4 × mean(logsumexp(logits)²)`）防止 logit 值域爆炸，**小模型尤其敏感**。`nano.yaml` 中 `z_loss_weight: 0.0`，而 `lite.yaml` 设为 `0.0001`。

Z-Loss 原理：交叉熵 `-log(softmax(logits)[correct])` 只关心正确 token 概率，不约束 softmax 分母的 `sum(exp(logits))` 大小。如果 logits 全体上浮 1000 倍，CE 值不变，但 `exp(logits)` → Inf → backward NaN。Z-Loss 通过惩罚 `logsumexp(logits)²` 直接约束 softmax 分母，防止数值上溢。

**为什么 standalone 正常**：standalone 的 `random.seed(42)` 在模型创建前执行，而 train.py 的 DataLoader shuffle 在模型创建前消耗了随机数，导致模型初始化权重不同。特定初始权重组合触发了 logit 爆炸。

**修复**：`nano.yaml` 中 `z_loss_weight: 0.0` → `0.0001`。

**验证**：`debug_train2.py`（完整复制 `scripts/train.py` 流程 + 全部 4 个修复）跑通：8 micro-batch 无 NaN，首个 optimizer step grad_norm=1.88。

**教训**：
1. Z-Loss 对小型模型不是可选的 — 参数越少，logit 值域越容易失控。
2. 训练脚本的任何前置操作（DataLoader shuffle、DDP init）都会改变模型随机初始化，导致"完全相同的配置"在不同脚本中行为不同。

---

## 30. Flash Attention `enable_gqa=True` 在 Windows RTX 4070 Ti 上性能退化 488x

**现象**：lite 87M（12L×768d，seq_len=2048，batch=4）GPU 训练时 forward 耗时 4-8s/batch，总 17s/batch。但 v0.2.1/v0.3.1 同样配置 forward 仅 ~0.4s。

**定位**：逐层 profile 发现每层注意力耗时 ~500ms。测试三种 SDPA backend：

| backend | 延迟 |
|---|---|
| flash | **488ms** |
| mem_efficient | 1ms |
| math | 107ms |

PyTorch 2.6 默认选择了 flash backend，但此 backend 在 Windows + RTX 4070 Ti 组合下触发极端性能退化。

**根因**：v0.2.1/v0.3.1 的 flash attention 路径**手动 expand K/V**（从 `(B, kv_heads, S, d)` → `(B, all_heads, S, d)` via `unsqueeze.expand.reshape.contiguous()`），然后调 `F.scaled_dot_product_attention(Q, K_fa, V_fa, is_causal=True)`——不传 `enable_gqa`。新版改成直接用 `enable_gqa=True` 让 PyTorch 内部处理 GQA 头扩展，在特定 GPU/OS 组合上触发了 buggy flash backend。

**修复**：回退 `GroupedQueryAttention.forward()` flash 路径到 v0.2.1 实现：

```python
# 修复前（enable_gqa=True → 488ms 严重性能退化）
if self.use_flash_attn:
    output = F.scaled_dot_product_attention(Q, K, V, enable_gqa=True, ...)

# 修复后（手动 expand K/V → 1ms 如 v0.2.1/v0.3.1）
if self.use_flash_attn and past_kv is None:
    K_fa = K.unsqueeze(2).expand(...).reshape(...).contiguous()
    V_fa = V.unsqueeze(2).expand(...).reshape(...).contiguous()
    output = F.scaled_dot_product_attention(Q, K_fa, V_fa, is_causal=True)
```

**教训**：
1. `enable_gqa` 看似方便，但 PyTorch 的 SDPA backend 选择在不同平台上有不可预期的表现。
2. 已在一个平台验证过的实现，不应为"更简洁的 API"而改变——能跑赢 benchmark 的代码才是好代码。
3. SDPA 的 backend 选择需要用 `torch.backends.cuda.sdp_kernel` 或 `torch.nn.attention.sdpa_kernel` 显式锁定，尤其在 Windows + 特定 GPU 组合上。
