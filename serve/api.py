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
from fastapi.responses import StreamingResponse
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


def _sample_token(logits: torch.Tensor, params) -> torch.Tensor:
    """单步采样: top_k → (T>0: 缩放+multinomial | T≤0: argmax)。"""
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


def _generate(input_ids: torch.Tensor, params, prompt_len: int) -> list[int]:
    """自回归生成，只返回新增 token（不含 prompt，避免把用户输入当 completion 返回）。"""
    tokens: list[int] = []
    with torch.no_grad():
        for _ in range(params.max_tokens):
            logits, _, _, _ = server.model.model(input_ids)
            nxt = _sample_token(logits[:, -1, :], params)
            input_ids = torch.cat([input_ids, nxt], dim=-1)
            if nxt.item() == server.tokenizer.eos_id:
                break
            tokens.append(int(nxt.item()))
            if params.stop:
                text = server.tokenizer.decode(tokens, skip_special=True)
                if _apply_stop(text, params.stop) != text:
                    break
    return tokens


def _step(input_ids: torch.Tensor, params) -> tuple[torch.Tensor, int]:
    """单步前向+采样（供流式生成在线程池中执行，避免阻塞事件循环）。"""
    with torch.no_grad():
        logits, _, _, _ = server.model.model(input_ids)
        nxt = _sample_token(logits[:, -1, :], params)
    return nxt, int(nxt.item())


async def _stream(input_ids: torch.Tensor, params, prompt_len: int, chat: bool):
    """SSE 流式响应: 每 token 一个 data 帧，完成后发 [DONE]。

    stop 命中时只发出截断前的增量文本（与非流式 _apply_stop 行为一致）。
    """
    total_decoded = ""
    emitted_len = 0
    for _ in range(params.max_tokens):
        nxt, nxt_id = await asyncio.to_thread(_step, input_ids, params)
        input_ids = torch.cat([input_ids, nxt], dim=-1)
        chunk = server.tokenizer.decode([nxt_id], skip_special=True)
        total_decoded += chunk
        stop_hit = nxt_id == server.tokenizer.eos_id
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
    # 默认注入 system 提示，与训练时的 ChatML 协议保持一致
    if not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": "You are GleamLM, a helpful assistant."})
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
