# GleamLM — Domain Glossary

## Core Architecture

- **Decoder-only Transformer**: Autoregressive architecture, no encoder, no cross-attention.
- **Pre-Norm**: RMSNorm applied _before_ each sublayer, not after.
- **RMSNorm**: Root Mean Square Layer Normalization; replaces LayerNorm.
- **RoPE**: Rotary Position Embedding, implemented via real-number ops (`x*cos + rotate_half(x)*sin`), not complex numbers.
- **GQA (Grouped Query Attention)**: Q heads > KV heads (e.g. 8Q / 4KV). KV heads repeat across groups.
- **QK-Norm**: Applying RMSNorm to Q and K vectors before RoPE (LLaMA 3 / Qwen3 standard).
- **SwiGLU**: Gated activation: `silu(W_gate * x) * (W_up * x)`, output via `W_down`.
- **FFN capacity**: The intermediate dimension `d_ff` of SwiGLU, typically `8/3 * d_model` by the standard formula.
- **Weight tying**: Embedding table reused as output projection (`lm_head`), reducing parameters.
- **KV Cache**: Store past key/value tensors for incremental generation, avoiding recomputation.

## Tokenizer

- **BBPE (Byte-Level BPE)**: Works on UTF-8 byte sequences; 256 base tokens + BPE merges. Self-developed, zero dependencies.
- **ChatML tokens**: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>` — special tokens for chat format. Registered as single IDs (12000, 12001).
- **CJK pre-tokenization**: Each Chinese character is a separate token unit; non-CJK kept as contiguous segments.

## Architecture Philosophy

- **Deep-Narrow**: More layers (≥12) with narrow hidden dimension (512/768). Proven superior for small Chinese models.
- **"Embedding is the gatekeeper, FFN is the brain"**: Minimize embedding parameters to maximize Transformer capacity. All factual knowledge resides in FFN weights.
- **12 layers is the minimum viable threshold for Chinese generation**: Dropping from 12→11 layers causes a 60% output diversity cliff.

## Training

- **AMP (Automatic Mixed Precision)**: BF16/FP16 training with `GradScaler`.
- **DDP (Distributed Data Parallel)**: Single-command multi-GPU via `torchrun`.
- **memmap dataset**: Tokenized data stored as `.npy` files, loaded via `np.load(..., mmap_mode='r')`. ~1 MB RAM for any dataset size.
- **Gradient accumulation**: `effective_batch = micro_batch × accumulate_grad`.
- **Z-Loss**: Regularizer `1e-4 * mean(logsumexp(logits)^2)`, prevents logit explosion.
- **WSD scheduler**: Warmup → Stable → Decay (3-phase learning rate).
- **Cosine scheduler**: Cosine Annealing + Warmup (2-phase learning rate).
- **Chinchilla optimal**: `tokens ≈ 20 × params` for compute-optimal training.

## Inference

- **Streaming generation**: Yield tokens incrementally (every 4 tokens by default).
- **Repetition penalty**: Divide logits of already-generated tokens to reduce loops.
- **Sampling strategies**: temperature, top-k, top-p, greedy (temperature=0).

## Data

- **Character-weighted mixing**: Convert target character% ratios to line-count ratios based on average characters-per-line.
- **Sliding window**: Overlapping windows with `stride = 3/4 * max_seq_len`.

## Model Variants

- **GleamLM-Nano (~40M)**: 12L × 512d, BBPE 12K. Baseline, complete (v0.1.0).
- **GleamLM-Lite (~87M)**: 12L × 768d, d_ff=2048 (3.4× FFN). Training in progress.
- **GleamLM-Pro (~126M)**: 18L × 768d. Planned.
- **GleamLM-0.6B (~597M)**: 37L × 1024d, 64K vocab, YaRN, Linux-only. Planned.

## _Avoid_

- Do not import from HuggingFace (`transformers`, `datasets`, `tokenizers`). GleamLM is self-contained.
- Do not use `torch.view_as_complex` for RoPE. Use real-number operations.
- Do not assume GPU is available; code paths must handle CPU fallback.
- Do not add comments explaining _what_ the code does; comments are for _why_.
