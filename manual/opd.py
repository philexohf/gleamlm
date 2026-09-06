"""
OPD (On-Policy Distillation) — 教师打分 + 序列级 Reverse KL。

核心思想:
  学生模型 (Student) 自己 rollout 轨迹 (on-policy)，教师模型 (Teacher) 对轨迹
  给出逐 token 监督 (蒸馏)。相比 RL 的稀疏奖励，OPD 每个 token 都有监督信号;
  相比 off-policy 蒸馏 (manual/distill.py)，学生自采样消除了 exposure bias。

教师模式:
  - local (唯一): 本地 HF 因果模型目录 (transformers 直接加载)，如
    checkpoints/Qwen3-0.6B。对 prompt+completion 前向求 log π_T(completion|prompt)
    (两次前向相减，跨 tokenizer 精确)。
  历史: ollama/DeepSeek API 教师已移除——DeepSeek 不返回完整预填 logprob (打分链
  断裂)，ollama 8B 打分慢 (~10s/次) 且偶发 timeout；本地 HF 0.6B 打分 ~0.03s 最优。

教师 tokenizer ≠ 学生 BBPE，逐 token 的 KL 对不上；但序列级可跨 tokenizer:

    log π_T(y) = Σ_t log p_T(t_i)     ← 教师返回的逐 token logprob 求和

  这是文本 y 的模型密度 (密度与 tokenizer 切分无关!)，因此我们优化序列级
  reverse KL:

    min KL(π_θ || π_T) = E_{y~π_θ}[ log π_θ(y) − log π_T(y) ]

  梯度 (score function / REINFORCE，E[∇log π_θ] = 0 消去一项):

    ∇ ≈ E_{y~π_θ}[ (log π_θ(y) − log π_T(y)) · ∇ log π_θ(y) ]

  实现上 (THUNLP 的 token_reward_direct 精神):
    - 学生采样时记录 log π_S(y)          (detach，作为 reward 一部分)
    - 教师返回 log π_T(y)                (教师 token 空间，求和)
    - 优势 A = log π_T(y) − log π_S(y)    (per-token 长度归一化，固定开启)
    - 损失 L = −(A − baseline) · log π_θ(y)   (policy 项重新前向，带梯度)

  符号核对 (对齐 THUNLP verl 参考实现): 其 rm_scores = −kl_val = −(logπ_S − logπ_T)
  = logπ_T − logπ_S，loss = −A·logπ 最小化 → 梯度提升教师更信的 token、压低学生
  过度自信的 token = 最小化 reverse KL。若误用 A = logπ_S − logπ_T 会反转为最大化
  reverse KL (数值单步验证: KL 1.32→1.66 而非下降)。

设计取舍:
  - OPD vs RL:  RL 一条轨迹一个 reward，OPD 每个 token 都有监督 → 样本效率高
  - OPD vs SFT 蒸馏: SFT 用静态数据 (off-policy)，训练分布 ≠ 生成分布 →
    exposure bias; OPD 学生自采样，分布天然对齐
  - 教师只能做序列级: tokenizer 不同，逐 token 无法对齐;
    但 log π_T(y) = Σ log p_T(t_i) 是文本密度，与切分无关 → 序列级可行
  - 用 REINFORCE 而非直接最小化: KL 的采样梯度含 score function 项;
    直接反传 log π_θ(y) 会变成熵最大化 (方向反了)
  - baseline 降低方差; 组内 leave-one-out 是 GRPO 的贡献
  - 长度归一化: 学生倾向拉长输出刷 logprob (length inflation)，除以长度缓解
 用法:
  python manual/opd.py \
    --model checkpoints/nano/dpo/dpo_best.pt \
    --data data/opd_prompts.jsonl \
    --output_dir checkpoints/nano/opd \
    --teacher_model_path checkpoints/Qwen3-0.6B

工程可靠性 (v3):
  - ChatML 帧对齐 (THUNLP OPD 论文 §5.2): 数据文件存裸 user 输入，训练时套成
    <|im_start|>user…<|im_end|>\n<|im_start|>assistant\n 再给双方。学生 SFT 与
    教师 Qwen 均在 ChatML 下训练，裸文本 rollout 会让师生双双 OOD (教师对每条
    续写恒定"惊讶"，log_pi_T 系统性偏低 ~1.2-1.5 nats，优势失真)；同帧后两边
    各回各自训练分布。BBPE 与 Qwen 的 im_start/im_end 标签文本一致，同一字符串
    在两边编码为各自的 special id，跨 tokenizer 精确性不受影响。

工程可靠性 (v2):
  - 采样用 model.eval() (关 dropout)，更新前 model.train()：
    避免采样 logits 与更新前向的 dropout mask 不一致，破坏 on-policy 性
  - 教师打分失败整组作废，保证 LOO baseline 组结构 (同一 prompt 恰 n_samples 条)
  - 温度 T<1e-6 走真正贪心分支 (1e-6 缩放会让 softmax 溢出成 nan)
  - 过短样本过滤 + 长度归一化下限 + loss 非有限值跳过
  - 打分缓存 (同轨迹不重复打分) + 周期 checkpoint (opd_checkpoint.pt) 断点续训
"""

import argparse
import itertools
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.chatml import format_chatml
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, extract_checkpoint_config
from gleamlm.utils.torch_utils import clean_state_dict, safe_autocast


class OPDDataset(Dataset):
    """JSONL prompt 数据集 — 每行 {"prompt": "..."}（兼容纯文本行）。

    存裸 user 输入；训练时在 train() 内统一套 ChatML 帧（见 format_chatml 调用点）。
    """

    def __init__(self, data_path: str):
        self.data = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    prompt = item.get("prompt", item.get("instruction", ""))
                except json.JSONDecodeError:
                    prompt = line
                if prompt:
                    self.data.append(prompt)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def tokenize_prompts(prompts: list[str], tokenizer, max_seq_len: int) -> torch.Tensor:
    ids = [tokenizer.encode(p, add_bos=False) for p in prompts]
    ids = [t[: max_seq_len - 8] for t in ids]
    max_len = max(len(t) for t in ids)
    padded = [t + [tokenizer.pad_id] * (max_len - len(t)) for t in ids]
    return torch.tensor(padded)


def _display_prompt(prompt: str, limit: int = 20) -> str:
    """日志展示: 从 ChatML 帧中还原裸 user 文本，避免打印 <|im_start|> 标签。"""
    for role in ("user", "system"):
        marker = f"<|im_start|>{role}\n"
        if prompt.startswith(marker):
            rest = prompt[len(marker) :]
            return rest.split("<|im_end|>")[0].strip()[:limit]
    return prompt[:limit]


def sample_with_logprobs(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    eos_id: int,
    max_len: int,
) -> tuple[torch.Tensor, list[float]]:
    """自回归采样并记录每个生成 token 的 logprob (no_grad)。

    采样分布 = softmax(logits / T)。T=0 退化为贪心 (argmax)，
    T 越大分布越平 → 探索更强。采样时的 logprob 是 reward 的一部分，
    必须 detach (它描述"旧策略"对这条轨迹的置信度)。
    """
    ids = prompt_ids.unsqueeze(0).clone()
    logprobs: list[float] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _, _, _ = model(ids)
            last = logits[:, -1, :]
            if temperature < 1e-6:
                # 贪心分支: T→0 的极限。不能用 1e-6 缩放 (logits 放大 1e6 倍
                # → exp 溢出 inf → softmax 出 nan)，也不能用 0 记 logprob
                # (会虚高 log π_S，污染优势估计)。用原始 logits 的 log_softmax
                nxt = last.argmax(dim=-1, keepdim=True)
                logp = F.log_softmax(last, dim=-1).gather(-1, nxt)
            else:
                # Categorical(logits=...) 内部做 log_softmax，数值稳定；
                # 直接传 softmax 后的 probs 在低温时易 underflow 失真
                dist = torch.distributions.Categorical(logits=last / temperature)
                nxt = dist.sample().unsqueeze(-1)
                logp = dist.log_prob(nxt.squeeze(-1))
            if nxt.item() == eos_id:
                # eos 终止 token 不计入 logprobs/ids，保证与教师打分(去 eos 文本)
                # 逐 token 对齐，logprobs 数与 Stage 4 重算的生成 token 数一致
                break
            logprobs.append(logp.item())
            ids = torch.cat([ids, nxt], dim=-1)
            if ids.size(1) >= max_len:
                break
    return ids, logprobs


def group_loo_baseline(advantages: torch.Tensor, group_size: int) -> torch.Tensor:
    """组内 leave-one-out baseline (GRPO 风格)。

    baseline_i = (Σ_j A_j − A_i) / (group_size − 1)：同一 prompt 的多个采样
    互为 baseline，减去"组内其他样本的平均质量"，保留本样本相对组内的优势。
    比 batch mean 方差更小，且不引入额外模型 (对比 PPO 的 value network)。
    """
    B = advantages.size(0)
    n_groups = B // group_size
    reshaped = advantages.view(n_groups, group_size)
    sums = reshaped.sum(dim=1, keepdim=True)
    baselines = (sums - reshaped) / (group_size - 1)
    return baselines.view(B)


class LocalTeacher:
    """本地教师 — 用 HF 因果模型对任意文本求 log π_T(completion | prompt)。

    Qwen3-0.6B 等本地 HF 模型作为教师（OPD 唯一教师方式）。核心:
      对 prompt + completion 拼接文本用教师 tokenizer 编码、前向拿 logits，
      两次前向相减: log p(prompt+completion) − log p(prompt) = log p(completion|prompt)，
      数学精确、无 BPE 边界对齐问题。
    教师 tokenizer ≠ 学生 BBPE，但序列级求和跨 tokenizer 可比。
    注意: 教师模型目录需匹配（Qwen3-0.6B 用 checkpoints/Qwen3-0.6B）。
    """

    def __init__(self, model_path: str, device: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            )
            .to(self.device)
            .eval()
        )

    def _seq_logprob(self, ids: torch.Tensor) -> float:
        """一次前向求整条 id 序列的 logprob（shift 后 gather）。"""
        with torch.no_grad():
            logits = self.model(ids).logits  # [1, S, V]
        shift_logits = logits[:, :-1, :].float()
        shift_ids = ids[:, 1:]
        logp = torch.log_softmax(shift_logits, dim=-1)
        log_probs = logp.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
        return log_probs.sum().item()

    def score(self, prompt: str, completion: str) -> float:
        """返回 log π_T(completion | prompt)。

        用两次前向相减: log p(prompt+completion) - log p(prompt)。
        数学精确、无 BPE 边界对齐问题；与 API 模式(只对 assistant 消息打分)
        和学生的 log_pi_S(只计生成部分) 语义一致。
        """
        full_ids = self.tok(prompt + completion, return_tensors="pt").input_ids.to(self.device)
        prompt_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        return self._seq_logprob(full_ids) - self._seq_logprob(prompt_ids)

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BBPETokenizer.load(args.tokenizer_path or DEFAULT_TOKENIZER_PATH)

    dataset = OPDDataset(args.data)
    # loader 在 resume 恢复随机状态之后创建（shuffle 依赖 rng）

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = extract_checkpoint_config(ckpt)
    model = GleamLMModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        num_kv_heads=cfg["num_kv_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg.get("dropout", 0.0),
        max_seq_len=args.seq_len,
        pad_token_id=tokenizer.pad_id,
        tie_weights=cfg.get("tie_weights", True),
        use_flash_attn=cfg.get("use_flash_attn", False),
    ).to(device)
    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    # strict=True：结构与 checkpoint 不一致会直接报错，避免静默跑在错误初始权重上
    model.load_state_dict(clean_state_dict(sd), strict=True)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 教师初始化: 本地 HF 模型 (OPD 唯一教师方式)
    if not args.teacher_model_path:
        raise ValueError("--teacher_model_path 必填 (如 checkpoints/Qwen3-0.6B)")
    local_teacher = LocalTeacher(args.teacher_model_path, device=str(device))
    print(f"本地教师已加载: {args.teacher_model_path}")
    score_cache: dict[str, float] = {}  # key = prompt+completion → log π_T (同轨迹不重复打分)

    total = sum(p.numel() for p in model.parameters())
    print(
        f"OPD — student: {total / 1e6:.2f}M, teacher: local ({args.teacher_model_path}), "
        f"T={args.temperature}, n_samples={args.n_samples}"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # LOO 组结构由 n_samples 决定（每个 prompt 采 n_samples 条，rollouts 中连续排列），
    # 与 batch_size（prompt 数）无整除关系约束。打分失败导致组不完整时，
    # Stage 3 会回退到 batch mean baseline，无需在此限制合法配置。
    global_step = 0
    start_epoch = 0
    last_batch_idx = -1  # 当前 epoch 内最后已遍历的 batch index（含打分失败），resume 从 +1 继续
    resume_path = os.path.join(args.output_dir, "opd_checkpoint.pt")
    if args.resume and os.path.exists(resume_path):
        r_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        rsd = r_ckpt.get("model_state_dict", r_ckpt.get("model", r_ckpt))
        model.load_state_dict(clean_state_dict(rsd), strict=True)
        if "optimizer_state_dict" in r_ckpt:
            optimizer.load_state_dict(r_ckpt["optimizer_state_dict"])
        global_step = r_ckpt.get("step", 0)
        start_epoch = r_ckpt.get("epoch", 0)
        last_batch_idx = r_ckpt.get("last_batch_idx", -1)
        # 恢复随机状态：必须在 DataLoader 创建之前，否则 shuffle 顺序错乱、
        # 采样轨迹与中断前不一致（保证 resume 严格复现原轨迹）
        if "python_rng" in r_ckpt:
            random.setstate(r_ckpt["python_rng"])
        if "torch_rng" in r_ckpt:
            torch.random.set_rng_state(r_ckpt["torch_rng"])
        print(
            f"[resume] 已从 {resume_path} 续训 (step {global_step}, epoch {start_epoch}, "
            f"last_batch_idx {last_batch_idx})，随机状态已恢复"
        )
    elif args.resume:
        print(f"[warn] --resume 但未找到 {resume_path}，从头训练")

    # loader 在 resume 恢复 rng 之后创建，保证 shuffle 顺序与中断前一致
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: b)
    for epoch in range(start_epoch, args.epochs):
        batch_iter = enumerate(loader)
        if epoch == start_epoch and last_batch_idx >= 0:
            # 跳过已遍历的 batch（含打分失败的），从下一个继续
            batch_iter = itertools.islice(batch_iter, last_batch_idx + 1, None)
        for batch_idx, batch_prompts in batch_iter:
            # ── Stage 1: 学生 on-policy 采样，按 prompt 显式分组 ──
            # 每组恰 n_samples 条（组结构对 LOO baseline 是硬约束，不能依赖
            # "过短被跳过后的连续排列"——那会跨 prompt 错位成假组）。
            # 采样用 eval()（关闭 dropout），否则采样 logits 受随机失活影响，
            # 与 Stage 4 更新时重新前向的 dropout mask 不一致，破坏 on-policy 性。
            model.eval()
            group_samples: dict[
                str, list
            ] = {}  # prompt -> [(gen_text, log_pi_S, gen_len, gen_ids, prompt_len)]
            skipped = 0
            for raw_prompt in batch_prompts:
                # THUNLP §5.2: 模板/内容需贴近教师后训练分布，否则教师的逐 token
                # 监督在 OOD 状态上不可靠。学生 SFT 与教师 Qwen 均为 ChatML；
                # 把裸 user 输入套成 <|im_start|>user…<|im_end|>\n<|im_start|>assistant\n
                # 生成提示，学生在此分布内 rollout、教师也在此分布内打分（历史裸文本
                # rollout 使师生双双 OOD：log_pi_S≈-3.2 vs log_pi_T≈-4.4，优势失真）。
                prompt = format_chatml(
                    [{"role": "user", "content": raw_prompt}], add_generation_prompt=True
                )
                prompt_ids = tokenize_prompts([prompt], tokenizer, args.seq_len).to(device)
                prompt_len = prompt_ids.size(1)
                group: list = []
                for _ in range(args.n_samples):
                    gen_ids, logprobs = sample_with_logprobs(
                        model,
                        prompt_ids[0],
                        args.max_new_tokens,
                        args.temperature,
                        tokenizer.eos_id,
                        args.seq_len,
                    )
                    if not logprobs or len(logprobs) < 4:
                        # 过短样本: 长度 1-3 的 logprob 噪声大 (优势被单 token 主导)
                        skipped += 1
                        continue
                    # 注: 长度不可能 > max_new_tokens*1.1 (采样循环上限即 max_new_tokens)，
                    # 被 max 截断的样本保留 (工业常规: 截断样本同样携带 on-policy 信号)
                    gen_ids = gen_ids.squeeze(0)
                    gen_text = tokenizer.decode(gen_ids[prompt_len:].tolist(), skip_special=True)
                    log_pi_S = sum(logprobs)  # 学生序列级 logprob (detach, 采样时已 no_grad)
                    group.append((gen_text, log_pi_S, len(logprobs), gen_ids, prompt_len))
                if len(group) == args.n_samples:
                    group_samples[prompt] = group
            if skipped:
                print(f"[warn] 过滤过短样本 {skipped} 条")

            if not group_samples:
                print("[warn] 本 batch 无完整采样组，跳过")
                continue

            # ── Stage 2: 教师打分。以完整组为单位打分，只保留整组打标成功的组 ──
            # (保证 LOO baseline 的组结构：组内恰 n_samples 条、同一 prompt)
            scored_groups: dict[
                str, list
            ] = {}  # prompt -> [(gen_text, log_pi_S, gen_len, gen_ids, prompt_len, lp_t)]
            for prompt, group in group_samples.items():
                scored: list = []
                for gen_text, log_pi_S, gen_len, gen_ids, prompt_len in group:
                    cache_key = prompt + "\x00" + gen_text
                    if args.api_cache and cache_key in score_cache:
                        lp_t = score_cache[cache_key]  # 同轨迹重新打分结果一致 (教师 temperature=0)
                    else:
                        # 本地教师: log π_T(completion | prompt)
                        lp_t = local_teacher.score(prompt, gen_text)
                        if lp_t is not None and args.api_cache:
                            score_cache[cache_key] = lp_t
                    if lp_t is None:
                        print(f"[warn] prompt={_display_prompt(prompt)}... 打分失败，整组作废")
                        break
                    scored.append((gen_text, log_pi_S, gen_len, gen_ids, prompt_len, lp_t))
                if len(scored) == args.n_samples:
                    scored_groups[prompt] = scored

            if not scored_groups:
                print("[warn] 本 batch 无整组打标成功的样本，跳过")
                continue

            # ── Stage 3: 展平为样本级张量 + per-token 归一化 ──
            valid: list = []  # (prompt, gen_text, log_pi_S, gen_len, gen_ids, prompt_len)
            teacher_lp: list[float] = []
            for prompt, scored in scored_groups.items():
                for gen_text, log_pi_S, gen_len, gen_ids, prompt_len, lp_t in scored:
                    valid.append((prompt, gen_text, log_pi_S, gen_len, gen_ids, prompt_len))
                    teacher_lp.append(lp_t)
            log_pi_S = torch.tensor([v[2] for v in valid], device=device, dtype=torch.float)
            log_pi_T = torch.tensor(teacher_lp, device=device, dtype=torch.float)
            # A 是序列级总量。policy loss 用 per-token 平均
            # logprob（.mean()），为使梯度尺度一致（不被生成长度隐式缩放），
            # 优势一律归一化为 per-token 平均（除以生成长度）。
            lengths = torch.tensor([v[3] for v in valid], device=device, dtype=torch.float)
            lengths = lengths.clamp(min=4.0)  # 长度下限: 极短序列的 logprob 噪声被放大
            log_pi_S = log_pi_S / lengths
            log_pi_T = log_pi_T / lengths
            # 优势符号 (对齐 THUNLP verl 实现): 参考代码 rm_scores = -kl_val = -(S-T) = T-S，
            # 再以 loss = -advantage·logπ 最小化 → 梯度提升"教师比学生更信"的 token 概率、
            # 压低"学生过度自信"的 token。若用 A=S-T 配合 -A·logπ 会让方向反转 → 最大化
            # reverse KL (数值验证: 单步后 KL 不降反升)。此处 A 用 logπ_T − logπ_S:
            #   A > 0: 教师比学生更信这条轨迹 → 提高学生概率 (向教师靠拢)
            #   A < 0: 学生比教师更信 → 降低学生概率 (避免过度自信)
            advantages = log_pi_T - log_pi_S
            # LOO baseline：valid 按 prompt 分组连续排列（组内 n_samples 条），
            # 可安全按 n_samples reshape。防御: 若 valid 长度不是 n_samples
            # 整数倍（理论上不会，但保护 reshape 崩溃），回退 batch mean。
            if args.n_samples > 1 and advantages.numel() % args.n_samples == 0:
                advantages = advantages - group_loo_baseline(advantages, args.n_samples)
            else:
                advantages = advantages - advantages.mean()

            # ── Stage 4: policy 项重新前向 (带梯度)，直接复用采样时的原始 gen_ids ──
            # 不用 decode→encode 还原: 特殊 token 无法无损还原，会引入与采样轨迹的偏差
            # 关键: 采样在 softmax(last/T) 分布下进行，重算 log π_θ 也必须除以 T，
            # 否则 tok_lp 与采样记录的 log_pi_S 不在同一温度尺度，优势 A 数值错乱。
            model.train()  # 更新需开 dropout（若配置了）；采样阶段已切 eval 关闭随机失活
            with safe_autocast():
                losses = []
                aux_loss = None  # MoE 负载均衡损失，循环内捕获，循环外统一加一次
                for i, (_, _, _, _, gen_ids, prompt_len) in enumerate(valid):
                    gen_ids = gen_ids[: args.seq_len].unsqueeze(0)
                    if gen_ids.size(1) <= prompt_len:
                        continue  # 空 response guard: 截断后无生成 token，跳过
                    logits, _, aux_loss, _ = model(gen_ids)
                    # shift: logits[t] 预测 token[t+1]，生成部分取 [prompt_len-1:-1]
                    resp_logits = logits[:, prompt_len - 1 : -1, :]
                    resp_tokens = gen_ids[:, prompt_len:].unsqueeze(-1)
                    # 温度一致性: 采样在 softmax(logits/T) 分布下，重算 log π_θ 须除 T。
                    # 但贪心分支 (T<1e-6) 采样用原始 logits 的 log_softmax (等价 T=1)，
                    # 因此重算也不能除 T，否则数值爆炸 (logits/T 巨大 → softmax 溢出)。
                    scale_T = args.temperature if args.temperature >= 1e-6 else 1.0
                    scaled_logits = resp_logits / scale_T
                    resp_log_probs = F.log_softmax(scaled_logits, dim=-1)
                    tok_lp = resp_log_probs.gather(-1, resp_tokens).squeeze(-1)  # [1, T]
                    policy_loss = -(advantages[i].detach() * tok_lp).mean()
                    if args.entropy_coeff > 0:
                        probs = F.softmax(scaled_logits, dim=-1)
                        entropy = -(probs * resp_log_probs).sum(-1).mean()
                        policy_loss = policy_loss - args.entropy_coeff * entropy
                    losses.append(policy_loss)

                if not losses:
                    print(f"[warn] step {global_step} 全部样本被空 response guard 跳过")
                    optimizer.zero_grad()
                    continue
                loss = torch.stack(losses).mean()
                # aux_loss (MoE 负载均衡) 是 batch 级全局损失，只加一次，
                # 不能在每个样本循环内累加（会乘 len(valid) 次）
                if aux_loss is not None:
                    loss = loss + args.aux_coeff * aux_loss

            if not torch.isfinite(loss):
                print(f"[warn] step {global_step} loss 非有限值 ({loss.item()}), 跳过本 batch")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()

            if global_step % args.log_interval == 0:
                print(
                    f"step {global_step}  loss={loss.item():.4f}  "
                    f"mean_A={advantages.mean().item():+.3f}  "
                    f"log_pi_S={log_pi_S.mean().item():+.2f}  "
                    f"log_pi_T={log_pi_T.mean().item():+.2f}"
                )
            global_step += 1

            # 周期保存: 中途崩溃可 resume (含完整训练状态 + 数据位置 + 随机状态)
            if global_step % args.save_interval == 0:
                ckpt_path = os.path.join(args.output_dir, "opd_checkpoint.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "_config": extract_checkpoint_config(ckpt),
                        "step": global_step,
                        "epoch": epoch,
                        "last_batch_idx": batch_idx,
                        "python_rng": random.getstate(),
                        "torch_rng": torch.random.get_rng_state(),
                        "method": "opd_local",
                    },
                    ckpt_path,
                )
                print(f"  [ckpt] saved step {global_step} -> {ckpt_path}")

    # 最终保存 (也写入周期 checkpoint 路径, 保证 resume 始终可用)
    final_path = os.path.join(args.output_dir, "opd_final.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "_config": extract_checkpoint_config(ckpt),
            "step": global_step,
            "epoch": args.epochs,
            "last_batch_idx": -1,  # 训练完成，resume 从头（若有下一轮）
            "python_rng": random.getstate(),
            "torch_rng": torch.random.get_rng_state(),
            "method": "opd_local",
        },
        final_path,
    )
    print(f"OPD model saved: {final_path}")
    local_teacher.close()


def parse_args():
    p = argparse.ArgumentParser(description="GleamLM OPD — 本地 HF 教师蒸馏")
    p.add_argument("--model", type=str, required=True, help="Student checkpoint")
    p.add_argument(
        "--data", type=str, required=True, help="JSONL prompts (每行 prompt/instruction 或纯文本)"
    )
    p.add_argument("--output_dir", type=str, default="./checkpoints/opd")
    p.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="训练轮数 (默认 4, nano 演示标准: 40 prompt × 4 = 80 步)",
    )
    p.add_argument("--batch_size", type=int, default=2, help="每 step 的 prompt 数")
    p.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="每个 prompt 采样条数 (THUNLP 用 4; >1 时启用组内 LOO baseline)",
    )
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument(
        "--temperature", type=float, default=1.0, help="采样温度 (0 = 贪心, 退化为 on-policy SFT)"
    )
    p.add_argument(
        "--lr", type=float, default=5e-6, help="OPD 用小 lr: 更新方向来自采样轨迹, 大 lr 易崩"
    )
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--entropy_coeff", type=float, default=0.01, help="熵正则: 防止策略过早坍缩")
    p.add_argument(
        "--length_norm",
        action="store_true",
        help="[已废弃] 优势长度归一化现在始终开启（policy loss 用 per-token 平均，"
        "数学必需，否则梯度被隐式缩放；此参数仅为向后兼容保留，不再影响行为）",
    )
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer_path", type=str, default="")
    # ── 教师 (本地 HF 模型，OPD 唯一方式) ──
    p.add_argument(
        "--teacher_model_path",
        type=str,
        default=None,
        help="本地 HF 教师模型目录，如 checkpoints/Qwen3-0.6B",
    )
    # ── 工程可靠性 (v2) ──
    p.add_argument("--aux_coeff", type=float, default=0.01, help="MoE aux loss 系数")
    p.add_argument(
        "--resume",
        action="store_true",
        help="从 output_dir/opd_checkpoint.pt 断点续训 (周期保存，含 step/epoch/batch/rng)",
    )
    p.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="每 N 步保存一次周期 checkpoint (opd_checkpoint.pt)",
    )
    p.add_argument(
        "--no_score_cache",
        action="store_true",
        help="关闭教师打分结果缓存 (默认开启；同 prompt+completion 重复打分结果一致)",
    )
    args = p.parse_args()
    args.api_cache = not args.no_score_cache  # 兼容保留: 打分缓存开关
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
