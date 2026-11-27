"""Advanced — batched generation: many prompts at once with left padding + attention masks.

`stream()` (Chapter 4) generates for ONE prompt. Serving several users one after another wastes
the hardware: a decoding step with batch 1 is a skinny (1, d) × (d, d) matmul that is limited by
reading the weights, not by arithmetic. Stack B prompts into one (B, d) matrix and the same weights
are read once for B tokens, so throughput (tokens / second) grows almost linearly with B until
compute becomes the bottleneck. Every inference server does this (vLLM, TGI, llama.cpp `-np`).

The catch: prompts have different lengths, and a tensor must be rectangular.

Left padding. Put the pad tokens on the LEFT so every prompt *ends* at the same column:

        col →   0     1     2     3     4
    row 0     <pad> <pad>  To    be     ,        ← short prompt
    row 1      O    Romeo   ,   Romeo   !        ← long prompt

Then the next token of every row is predicted from the last column, and each decoding step
appends exactly one column to all rows — the KV cache stays one rectangular (B, H, S, D) tensor.
(Right padding would put the "last real token" at a different column per row and make the new
tokens land after the pads.)

Key-padding mask. Pad tokens must be invisible: a real query may only attend to real keys.

    allowed[b, i, j] = causal(i, j)  and  key_valid[b, j]          (and the sliding window, if any)

`cache.key_valid` is a (B, S_total) boolean tensor carried next to the KV cache; every attention
layer combines it with its causal / windowed mask (`padded_attention_mask`). One detail: a pad
*query* in the first columns would see no key at all — softmax over all -inf is NaN, and a NaN in
the residual stream poisons later layers through 0 · NaN. So every position may always attend to
itself; real queries never sit on a pad key, so their result is unaffected.

Why RoPE needs no fix. Row 0 above sees its first real token at absolute position 2, not 0. RoPE
makes q·k depend only on the *distance* between positions (rotation by mθ, then by nθ, gives
a dot product that depends on m - n), so shifting a whole prompt right by p pads leaves every
attention score unchanged. Padded batched output equals one-prompt-at-a-time output up to
floating-point rounding (tested). Models with learned absolute positions (GPT-2) must pass
explicit position ids instead.

Finished rows. When a row samples a stop token it is marked done; it keeps riding along in the
batch (feeding a pad, masked out) until every row is done or the token budget is spent. That is
"static batching". Continuous batching (vLLM) instead evicts finished rows and admits new requests
mid-flight — a good follow-up exercise.

Opt-in: `stream()` / `generate()` and the model's default path are unchanged; the mask code only
runs when `cache.key_valid` is set, which only `batch_stream()` does.
"""
from __future__ import annotations

from typing import Iterator

import torch

from .sliding_window import attention_mask


def left_pad(prompts: list[list[int]], pad_id: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack token lists into (B, L) ids + (B, L) validity mask, padded on the LEFT."""
    if not prompts or any(len(p) == 0 for p in prompts):
        raise ValueError("every prompt needs at least one token")
    L = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), L), pad_id, dtype=torch.long, device=device)
    valid = torch.zeros((len(prompts), L), dtype=torch.bool, device=device)
    for b, p in enumerate(prompts):
        ids[b, L - len(p):] = torch.tensor(p, dtype=torch.long, device=device)
        valid[b, L - len(p):] = True
    return ids, valid


def padded_attention_mask(T: int, S: int, window: int, cache, device=None) -> torch.Tensor | None:
    """(B, 1, T, S) boolean mask = causal/window mask AND key is not padding (or is the query itself).

    Returns None when the cache carries no padding information — the caller then uses its usual
    unbatched mask, so the default code path is untouched.
    """
    if cache is None or getattr(cache, "key_valid", None) is None:
        return None
    valid = cache.key_valid[:, -S:]                          # a rolling cache keeps only the last S keys
    base = attention_mask(T, S, window, device=device)       # (T, S): query i sits at key S - T + i
    q_pos = torch.arange(T, device=device)[:, None] + (S - T)
    self_key = q_pos == torch.arange(S, device=device)[None, :]   # always see yourself: no all -inf rows
    return ((base[None] & valid[:, None, :]) | self_key[None])[:, None]   # (B, 1, T, S)


def _prev_tokens(seq: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Tokens for the repetition penalty, with pads replaced by the row's first real token.

    Penalising a token twice changes nothing (gather/scatter write the same value), so duplicates
    are harmless — but the pad id itself must not be penalised.
    """
    first = seq.gather(1, valid.float().argmax(1, keepdim=True))   # first real token of each row
    return torch.where(valid, seq, first)


@torch.no_grad()
def batch_stream(model, prompts: list[list[int]], max_new_tokens: int = 100, temperature: float = 0.8,
                 top_k: int | None = 50, top_p: float | None = 0.95, repetition_penalty: float = 1.1,
                 stop_ids: set[int] | None = None, pad_id: int = 0) -> Iterator[list[int | None]]:
    """Generate for B prompts at once. Yields one list per step: the new token of each row, or None
    for rows that already stopped. Stops when every row is done, after max_new_tokens, or when the
    padded sequence reaches the model's context (no cache rebuild, unlike `stream()`)."""
    from .generate import sample_next   # generate.py imports model.py; keep this module light

    model.eval()
    device = next(model.parameters()).device
    ctx = model.cfg.max_seq_len
    prompts = [list(p)[-ctx:] for p in prompts]
    ids, valid = left_pad(prompts, pad_id, device)
    B = ids.shape[0]
    cache = model.new_cache()
    cache.key_valid = valid
    logits, _ = model(ids, cache=cache)                      # prefill all rows together
    seq, seq_valid = ids, valid
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        prev = _prev_tokens(seq, seq_valid)
        nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p, prev, repetition_penalty)  # (B, 1)
        if stop_ids:
            done = done | torch.isin(nxt[:, 0], torch.tensor(sorted(stop_ids), device=device))
        alive = ~done
        if not alive.any():
            return
        yield [int(t) if a else None for t, a in zip(nxt[:, 0].tolist(), alive.tolist())]
        nxt = torch.where(alive[:, None], nxt, torch.full_like(nxt, pad_id))
        seq = torch.cat([seq, nxt], dim=1)
        seq_valid = torch.cat([seq_valid, alive[:, None]], dim=1)
        if cache.length >= ctx:
            return
        cache.key_valid = seq_valid                          # the new column, masked for finished rows
        logits, _ = model(nxt, cache=cache)


def batch_generate_ids(model, prompts: list[list[int]], **kw) -> list[list[int]]:
    """Run `batch_stream` to the end and return the generated ids of each row."""
    out: list[list[int]] = [[] for _ in prompts]
    for step in batch_stream(model, prompts, **kw):
        for b, t in enumerate(step):
            if t is not None:
                out[b].append(t)
    return out


def generate_batch(model, tokenizer, prompts: list[str], **kw) -> list[str]:
    """Text in, text out — the batched counterpart of `generate.generate()`."""
    bos = tokenizer.token_id("<|bos|>")
    ids = [tokenizer.encode(p) or [bos] for p in prompts]
    kw.setdefault("stop_ids", {tokenizer.token_id("<|eos|>")})
    kw.setdefault("pad_id", tokenizer.token_id("<|pad|>"))
    return [tokenizer.decode(o) for o in batch_generate_ids(model, ids, **kw)]
