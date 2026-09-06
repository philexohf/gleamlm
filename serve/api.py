"""
GleamLM API Server — FastAPI + OpenAI 兼容接口。

生产级推理需要 batching、continuous batching、KV cache 管理；
/v1/completions 和 /v1/chat/completions 是行业标准接口。
此处为基础实现演示架构，生产推荐 vLLM (见 deploy/)。

用法:
  python serve/api.py \
    --model checkpoints/pro/final.pt \
    --tokenizer_path checkpoints/bbpe_12k \
    --port 8000

  curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "max_tokens": 50}'
"""

import argparse
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from hf.hf_config import gleamlm_config_from_core
from hf.hf_model import GleamLMForCausalLM, load_from_checkpoint
from gleamlm.utils.chatml import format_chatml
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, extract_checkpoint_config


class CompletionRequest(BaseModel):
    model: str = "gleamlm"
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.15  # 对齐训练评估默认值
    stop: Optional[list[str]] = None
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "gleamlm"
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.15  # 对齐训练评估默认值
    stop: Optional[list[str]] = None
    stream: bool = False


class ModelServer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self, model_path: str, tokenizer_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BBPETokenizer.load(tokenizer_path)

        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        cfg = extract_checkpoint_config(ckpt)
        hf_config = gleamlm_config_from_core(cfg)
        self.model = GleamLMForCausalLM(hf_config).to(self.device)
        missing, unexpected = load_from_checkpoint(self.model, ckpt, strict=True)
        if missing or unexpected:
            print(f"[warn] server load — missing={missing} unexpected={unexpected}")
        self.model.eval()

        total = sum(p.numel() for p in self.model.parameters())
        print(f"Server loaded: {total/1e6:.2f}M on {self.device}")


server = ModelServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="GleamLM API", version="0.2.0", lifespan=lifespan)


def _check_request(req, prompt_len: int) -> None:
    """参数校验: 负 temperature / 非法 max_tokens / 超上下文长度。"""
    if req.temperature < 0:
        raise HTTPException(status_code=400, detail="temperature 不能为负")
    if req.max_tokens <= 0 or req.max_tokens > 2048:
        raise HTTPException(status_code=400, detail="max_tokens 需在 (0, 2048] 范围")
    max_pos = server.model.config.max_position_embeddings
    if prompt_len + req.max_tokens > max_pos:
        raise HTTPException(
            status_code=400,
            detail=f"prompt({prompt_len}) + max_tokens({req.max_tokens}) 超过上下文长度 {max_pos}",
        )


def _sample_token(logits: torch.Tensor, params, generated: list[int] | None = None) -> torch.Tensor:
    """单步采样: repetition_penalty → top_k → (T>0: 缩放+multinomial | T≤0: argmax)。"""
    # 与 gleamlm.inference.generator.sample_token 同款重复惩罚（HF 算法）
    penalty = getattr(params, "repetition_penalty", 1.0)
    if penalty != 1.0 and generated:
        for gid in set(generated):
            scores = logits[..., gid]
            logits[..., gid] = torch.where(
                scores < 0, scores * penalty, scores / penalty
            )
    if params.top_k > 0:
        vals, _ = logits.topk(params.top_k, dim=-1)
        logits = logits.masked_fill(logits < vals[:, -1:], float("-inf"))
    if params.temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits / params.temperature, dim=-1)
    return torch.multinomial(probs, 1)


def _apply_stop(text: str, stops: Optional[list[str]]) -> str:
    """按 stop 字符串截断文本（支持跨 token 边界拼接后命中）。"""
    if not stops:
        return text
    for s in stops:
        if s and s in text:
            return text.split(s)[0]
    return text


def _stop_ids() -> set[int]:
    """ChatML 训练下模型以 im_end/eos 收尾；serve 需同样识别才不超生成。"""
    tk = server.tokenizer
    ids = {tk.eos_id, tk.im_end_id, tk.pad_id}
    ids.discard(None)
    return ids


def _generate(input_ids: torch.Tensor, params, prompt_len: int) -> list[int]:
    """自回归生成，只返回新增 token（不含 prompt，避免把用户输入当 completion 返回）。"""
    tokens: list[int] = []
    generated: list[int] = input_ids[0].tolist()  # 重复惩罚需看已生成序列
    stop_ids = _stop_ids()
    with torch.no_grad():
        for _ in range(params.max_tokens):
            logits, _, _, _ = server.model.model(input_ids)
            nxt = _sample_token(logits[:, -1, :], params, generated)
            token_id = int(nxt.item())
            if token_id in stop_ids:
                break
            input_ids = torch.cat([input_ids, nxt], dim=-1)
            tokens.append(token_id)
            generated.append(token_id)
            if params.stop:
                text = server.tokenizer.decode(tokens, skip_special=True)
                if _apply_stop(text, params.stop) != text:
                    break
    return tokens


def _step(input_ids: torch.Tensor, params, generated: list[int]) -> tuple[torch.Tensor, int, bool]:
    """单步前向+采样（供流式生成在线程池中执行，避免阻塞事件循环）。
    返回 (下一 token, 其 id, 是否命中终止符 eos/im_end/pad)。"""
    with torch.no_grad():
        logits, _, _, _ = server.model.model(input_ids)
        nxt = _sample_token(logits[:, -1, :], params, generated)
    token_id = int(nxt.item())
    return nxt, token_id, token_id in _stop_ids()


async def _stream(input_ids: torch.Tensor, params, prompt_len: int, chat: bool):
    """SSE 流式响应: 每 token 一个 data 帧，完成后发 [DONE]。

    stop 命中时只发出截断前的增量文本（与非流式 _apply_stop 行为一致）。
    """
    total_decoded = ""
    emitted_len = 0
    generated: list[int] = input_ids[0].tolist()
    for _ in range(params.max_tokens):
        nxt, nxt_id, stop_hit = await asyncio.to_thread(_step, input_ids, params, generated)
        input_ids = torch.cat([input_ids, nxt], dim=-1)
        generated.append(nxt_id)
        chunk = server.tokenizer.decode([nxt_id], skip_special=True)
        total_decoded += chunk
        if params.stop:
            trimmed = _apply_stop(total_decoded, params.stop)
            if trimmed != total_decoded:
                stop_hit = True
                total_decoded = trimmed
        delta = total_decoded[emitted_len:]
        emitted_len = len(total_decoded)
        if delta:
            if chat:
                payload = {"choices": [{"delta": {"content": delta}}]}
            else:
                payload = {"choices": [{"text": delta}]}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        if stop_hit:
            break
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    input_ids = torch.tensor([[server.tokenizer.bos_id] + server.tokenizer.encode(req.prompt)], device=server.device)
    prompt_len = input_ids.size(1)
    _check_request(req, prompt_len)
    if req.stream:
        return StreamingResponse(_stream(input_ids, req, prompt_len, chat=False),
                                 media_type="text/event-stream")
    tokens = _generate(input_ids, req, prompt_len)
    text = server.tokenizer.decode(tokens, skip_special=True)
    text = _apply_stop(text, req.stop)
    return {"id": "cmpl-1", "object": "text_completion", "choices": [{"text": text}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    # 不自动注入 system：SFT 训练仅 20% 样本带 system（80% 无），
    # 纯 user/assistant 帧更贴合训练主流分布；调用方自带 system 时原样保留
    prompt = format_chatml(messages, add_generation_prompt=True)
    input_ids = torch.tensor([server.tokenizer.encode(prompt, add_bos=False)], device=server.device)
    prompt_len = input_ids.size(1)
    _check_request(req, prompt_len)
    if req.stream:
        return StreamingResponse(_stream(input_ids, req, prompt_len, chat=True),
                                 media_type="text/event-stream")
    tokens = _generate(input_ids, req, prompt_len)
    text = server.tokenizer.decode(tokens, skip_special=True)
    text = _apply_stop(text, req.stop)
    return {"id": "chat-1", "object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": text}}]}


@app.get("/health")
async def health():
    return {"status": "ok", "model": "gleamlm"}


@app.get("/")
async def root():
    """聊天界面（浏览器直接访问根路径即对话页，前端直连 /v1/chat/completions）。"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "chat.html"))


def parse_args():
    p = argparse.ArgumentParser(description="GleamLM API Server")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer_path", default="")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    server.load(args.model, args.tokenizer_path or DEFAULT_TOKENIZER_PATH)
    uvicorn.run(app, host=args.host, port=args.port)
