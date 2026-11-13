"""Advanced — training-stability tricks: QK-norm and logit soft-capping.

Large or long training runs sometimes blow up: a few attention logits (q·k) grow without bound,
softmax turns into a one-hot, gradients spike and the loss diverges. Two cheap fixes are used by
recent open models:

  * QK-norm (OLMo 2, Gemma 3, Qwen3): apply an RMSNorm to every query and key head *before*
    RoPE and the dot product. q·k is then bounded by the norm weights instead of by however big
    the projection weights have grown, so attention logits cannot explode.

  * Logit soft-capping (Gemma 2): squash logits smoothly into (-cap, cap) with

        softcap(x) = cap * tanh(x / cap)

    It is ~identity for |x| << cap and saturates for large |x|. Gemma 2 caps attention scores at
    50 and the final vocabulary logits at 30. Unlike hard clipping, gradients never become
    exactly zero.

Both are opt-in through ModelConfig(qk_norm=..., attn_softcap=..., final_softcap=...); the
defaults reproduce the plain Chapter 2 model exactly.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """cap * tanh(x / cap): smooth, differentiable squashing into (-cap, cap). cap <= 0 = off."""
    if cap <= 0:
        return x
    return cap * torch.tanh(x / cap)


def softcapped_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cap: float,
                         dropout_p: float = 0.0) -> torch.Tensor:
    """Causal attention with soft-capped scores (what Gemma 2 does).

    F.scaled_dot_product_attention can't apply a function between q·k and softmax, so this is the
    attention formula written out by hand:

        scores = softcap(q·kᵀ / √d, cap) ; masked ; softmax ; · v

    q: (B, H, T, D), k/v: (B, H, S, D) with S >= T (S > T when a KV cache holds earlier tokens).
    Query i sits at absolute position S - T + i and may attend to keys 0 .. S - T + i.
    """
    T, S = q.shape[2], k.shape[2]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])     # (B, H, T, S)
    scores = softcap(scores, cap)
    allowed = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(S - T)
    scores = scores.masked_fill(~allowed, float("-inf"))
    att = F.softmax(scores.float(), dim=-1).type_as(q)
    if dropout_p > 0:
        att = F.dropout(att, p=dropout_p)
    return att @ v
