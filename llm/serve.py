"""Chapter 6b — Use your model like a real LLM: an OpenAI-compatible HTTP API.

Run:   python -m llm.serve --ckpt checkpoints/best.pt --tokenizer checkpoints/tokenizer.json
Then any OpenAI client works:

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
    client.chat.completions.create(model="tiny-llm", messages=[{"role": "user", "content": "Hi"}])
"""
# NOTE: no `from __future__ import annotations` here — FastAPI must resolve the
# locally defined pydantic models at runtime.
import argparse
import json
import time
import uuid

import torch

from .finetune import format_chat
from .generate import stream
from .tokenizer import BPETokenizer
from .trainer import load_checkpoint, pick_device


def build_app(model, tokenizer, model_name: str = "tiny-llm"):
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = model_name
        messages: list[Message]
        max_tokens: int = 200
        temperature: float = 0.8
        top_p: float = 0.95
        stream: bool = False

    app = FastAPI(title="llm-from-scratch")
    device = next(model.parameters()).device
    eos = tokenizer.token_id("<|eos|>")

    def token_iter(req: ChatRequest):
        prompt = format_chat([m.model_dump() for m in req.messages], add_generation_prompt=True)
        x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        yield from stream(model, x, req.max_tokens, req.temperature, 50, req.top_p, stop_ids={eos})

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "you"}]}

    @app.get("/health")
    def health():
        return {"status": "ok", "params": model.num_params()}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        if req.stream:
            def sse():
                for tok in token_iter(req):
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                             "choices": [{"index": 0, "delta": {"content": tokenizer.decode([tok])}, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")
        ids = list(token_iter(req))
        return {"id": cid, "object": "chat.completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": tokenizer.decode(ids)},
                             "finish_reason": "stop"}],
                "usage": {"completion_tokens": len(ids)}}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    import uvicorn
    model = load_checkpoint(a.ckpt, pick_device())
    uvicorn.run(build_app(model, BPETokenizer.load(a.tokenizer)), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
