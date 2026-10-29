"""Chapter 5b — From base model to assistant: SFT and DPO.

SFT (supervised fine-tuning): train on chat-formatted examples, computing the loss
only on the assistant's tokens (the prompt is masked with -100).

DPO (Direct Preference Optimization, Rafailov et al. 2023): given a prompt with a
`chosen` and a `rejected` answer, push the policy to prefer `chosen` relative to a
frozen reference model — RLHF-style alignment without a reward model or PPO.
"""
from __future__ import annotations

import copy
import random

import torch
import torch.nn.functional as F

from .model import GPT

# ----------------------------------------------------------------- chat template
def format_chat(messages: list[dict], add_generation_prompt: bool = False) -> str:
    out = "<|bos|>"
    for m in messages:
        tag = "<|user|>" if m["role"] == "user" else "<|assistant|>"
        out += f"{tag}{m['content']}"
        if m["role"] == "assistant":
            out += "<|eos|>"
    if add_generation_prompt:
        out += "<|assistant|>"
    return out


def encode_sft_example(tokenizer, prompt: str, response: str, max_len: int):
    p = tokenizer.encode(format_chat([{"role": "user", "content": prompt}], add_generation_prompt=True))
    r = tokenizer.encode(response + "<|eos|>")
    ids = (p + r)[: max_len + 1]
    labels = ([-100] * len(p) + r)[: max_len + 1]
    return ids, labels


def _pad_batch(rows, pad_id):
    T = max(len(ids) for ids, _ in rows) - 1
    x = torch.full((len(rows), T), pad_id, dtype=torch.long)
    y = torch.full((len(rows), T), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(rows):
        n = len(ids) - 1
        x[i, :n] = torch.tensor(ids[:-1])
        y[i, :n] = torch.tensor(labels[1:])
    return x, y


def sft(model: GPT, tokenizer, examples: list[dict], steps: int = 200, batch_size: int = 8,
        lr: float = 1e-3, device: str = "cpu", log=print) -> GPT:
    """examples: [{"prompt": ..., "response": ...}, ...]"""
    model.to(device).train()
    rows = [encode_sft_example(tokenizer, e["prompt"], e["response"], model.cfg.context_length) for e in examples]
    pad = tokenizer.token_id("<|pad|>")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    for step in range(steps):
        x, y = _pad_batch(random.sample(rows, min(batch_size, len(rows))), pad)
        _, loss = model(x.to(device), y.to(device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            log(f"sft step {step:4d} | loss {loss.item():.4f}")
    return model


# ----------------------------------------------------------------- DPO
def _sequence_logprob(model, x, y):
    logits, _ = model(x, all_logits=True)
    logp = F.log_softmax(logits.float(), -1)
    mask = y != -100
    tok_logp = logp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (tok_logp * mask).sum(-1)


def dpo(model: GPT, tokenizer, pairs: list[dict], steps: int = 30, beta: float = 0.1,
        lr: float = 5e-5, batch_size: int = 4, device: str = "cpu", log=print) -> GPT:
    """pairs: [{"prompt": ..., "chosen": ..., "rejected": ...}, ...]"""
    model.to(device).train()
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad = False
    T, pad = model.cfg.context_length, tokenizer.token_id("<|pad|>")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    for step in range(steps):
        batch = random.sample(pairs, min(batch_size, len(pairs)))
        xc, yc = _pad_batch([encode_sft_example(tokenizer, b["prompt"], b["chosen"], T) for b in batch], pad)
        xr, yr = _pad_batch([encode_sft_example(tokenizer, b["prompt"], b["rejected"], T) for b in batch], pad)
        xc, yc, xr, yr = (t.to(device) for t in (xc, yc, xr, yr))
        pi_c, pi_r = _sequence_logprob(model, xc, yc), _sequence_logprob(model, xr, yr)
        with torch.no_grad():
            ref_c, ref_r = _sequence_logprob(ref, xc, yc), _sequence_logprob(ref, xr, yr)
        margin = beta * ((pi_c - ref_c) - (pi_r - ref_r))
        loss = -F.logsigmoid(margin).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == steps - 1:
            log(f"dpo step {step:4d} | loss {loss.item():.4f} | reward-acc {(margin > 0).float().mean().item():.2f}")
    return model
