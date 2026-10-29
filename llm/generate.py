"""Chapter 4 — Inference: turning a trained network into a text generator.

Sampling strategies implemented here:
  * greedy            (temperature = 0)
  * temperature       (flatten / sharpen the distribution)
  * top-k             (keep the k most likely tokens)
  * top-p / nucleus   (keep the smallest set whose probability mass >= p)
  * repetition penalty
All of them run on top of the KV cache, so each new token costs one forward pass of length 1.
"""
from __future__ import annotations

from typing import Iterator

import torch

from .model import GPT, KVCache


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                top_p: float | None = None, prev: torch.Tensor | None = None,
                repetition_penalty: float = 1.0) -> torch.Tensor:
    logits = logits.clone()
    if prev is not None and repetition_penalty != 1.0:
        scores = logits.gather(-1, prev)
        scores = torch.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty)
        logits.scatter_(-1, prev, scores)
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1:]
        logits[logits < kth] = float("-inf")
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        remove = cum - sorted_logits.softmax(-1) > top_p   # keep at least one token
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(logits.softmax(-1), 1)


@torch.no_grad()
def stream(model: GPT, idx: torch.Tensor, max_new_tokens: int = 100, temperature: float = 0.8,
           top_k: int | None = 50, top_p: float | None = 0.95, repetition_penalty: float = 1.1,
           stop_ids: set[int] | None = None) -> Iterator[int]:
    """Yield token ids one at a time (perfect for streaming APIs)."""
    model.eval()
    ctx = model.cfg.context_length
    idx = idx[:, -ctx:]
    cache = KVCache(model.cfg.n_layers)
    logits, _ = model(idx, cache=cache)                       # prefill
    seq = idx
    for _ in range(max_new_tokens):
        nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p, seq, repetition_penalty)
        tok = int(nxt.item())
        if stop_ids and tok in stop_ids:
            return
        yield tok
        seq = torch.cat([seq, nxt], dim=1)
        if cache.length >= ctx:                                # sliding window: rebuild cache
            cache = KVCache(model.cfg.n_layers)
            logits, _ = model(seq[:, -ctx:], cache=cache)
        else:
            logits, _ = model(nxt, cache=cache)


def generate(model: GPT, tokenizer, prompt: str, **kw) -> str:
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt) or [tokenizer.token_id("<|bos|>")]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    stop = {tokenizer.token_id("<|eos|>")}
    return tokenizer.decode(list(stream(model, x, stop_ids=stop, **kw)))
