"""Advanced — Multi-head Latent Attention (MLA), the attention of DeepSeek-V2 / V3 / R1.

During generation the KV cache, not the weights, is what eats memory: standard attention stores
2 · n_kv_heads · head_dim numbers per token per layer. GQA shrinks that by sharing K/V heads.
MLA shrinks it further by caching a small *latent* vector instead of the keys and values:

    c_kv   = RMSNorm(W_dkv · x)              (d_c numbers,  d_c << 2 · n_heads · head_dim)
    k_h    = W_uk[h] · c_kv                  per-head keys,   reconstructed ("up-projected")
    v_h    = W_uv[h] · c_kv                  per-head values, reconstructed

Only c_kv goes into the cache. Every head still gets its *own* keys and values (unlike GQA/MQA,
where heads literally share them) — they are just constrained to a d_c-dimensional subspace.

Decoupled RoPE. RoPE rotates keys by position, and a rotation can't be pulled through W_uk (see
"absorption" below), so MLA splits each query/key into two parts:

    q_h = [ q_nope_h  ;  RoPE(q_rope_h) ]        k_h = [ W_uk[h] · c_kv  ;  RoPE(k_rope) ]
          head_dim        rope_dim                       head_dim             rope_dim

k_rope = W_kr · x is a single small vector *shared by all heads* (like MQA) and it is cached too.
Cache per token per layer: d_c + rope_dim numbers. DeepSeek-V3: 512 + 64 = 576, versus
2 · 128 heads · 128 = 32768 for plain multi-head attention (57x smaller).

Weight absorption (inference trick). For the content part of the score,

    q_nope_h · k_h = q_nope_h · (W_uk[h] c_kv) = (W_uk[h]ᵀ q_nope_h) · c_kv

so the query can be moved into latent space once and dotted with the cached latents directly —
the per-head keys are never materialised. Likewise Σ_j a_j v_j = W_uv[h] · (Σ_j a_j c_kv_j), so
values are up-projected *after* the attention-weighted sum. This is what `absorb=True` does; it
gives exactly the same output as the naive path (tested), it just never builds full K/V.

Opt-in: ModelConfig(mla_kv_rank=d_c, mla_rope_dim=d_r). mla_kv_rank = 0 (default) keeps the
GQA attention of Chapter 2. Works with the KV cache, sliding windows and attention soft-capping.
Not done here (DeepSeek also does it): a low-rank *query* compression to save activation memory.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .model import KVCache, RMSNorm, apply_rope
from .sliding_window import attention_mask, layer_window
from .stability import softcap, softcapped_attention


def kv_cache_numbers_per_token(cfg: ModelConfig) -> int:
    """How many numbers one token occupies in the KV cache of ONE layer."""
    if cfg.mla_kv_rank > 0:
        return cfg.mla_kv_rank + cfg.mla_rope_dim
    return 2 * cfg.n_kv_heads * cfg.head_dim


class MLAttention(nn.Module):
    """Drop-in replacement for model.Attention that caches a compressed latent instead of K/V."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        assert cfg.mla_rope_dim % 2 == 0 and cfg.mla_rope_dim > 0, "mla_rope_dim must be even"
        assert not cfg.qk_norm, "qk_norm is not supported together with MLA"
        self.cfg, self.layer_idx = cfg, layer_idx
        self.H, self.hd = cfg.n_heads, cfg.head_dim
        self.dc, self.dr = cfg.mla_kv_rank, cfg.mla_rope_dim
        # queries: content part (head_dim) + rotary part (rope_dim) for every head
        self.wq = nn.Linear(cfg.d_model, self.H * (self.hd + self.dr), bias=False)
        # down-projection: latent c_kv (d_c) and the shared rotary key k_rope (d_r) in one matmul
        self.w_dkv = nn.Linear(cfg.d_model, self.dc + self.dr, bias=False)
        self.kv_norm = RMSNorm(self.dc)
        # up-projection: latent -> per-head [k_nope ; v]
        self.w_ukv = nn.Linear(self.dc, self.H * 2 * self.hd, bias=False)
        self.wo = nn.Linear(self.H * self.hd, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.window = layer_window(cfg.sliding_window, cfg.global_every, layer_idx)
        self.scale = 1.0 / math.sqrt(self.hd + self.dr)

    def forward(self, x, cos, sin, cache: KVCache | None = None, absorb: bool | None = None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.H, self.hd + self.dr).transpose(1, 2)
        q_nope, q_rope = q.split([self.hd, self.dr], dim=-1)
        q_rope = apply_rope(q_rope, cos, sin)
        c_kv, k_rope = self.w_dkv(x).split([self.dc, self.dr], dim=-1)
        c_kv = self.kv_norm(c_kv).unsqueeze(1)                       # (B, 1, T, d_c)
        k_rope = apply_rope(k_rope.unsqueeze(1), cos, sin)           # (B, 1, T, d_r), shared by heads
        if cache is not None:   # the cache stores the latent and the rotary key, nothing else
            c_kv, k_rope = cache.update(self.layer_idx, c_kv, k_rope)
        if absorb is None:      # default: absorbed weights for generation, naive path for training
            absorb = cache is not None and not self.training
        if absorb:
            y = self._absorbed(q_nope, q_rope, c_kv, k_rope)
        else:
            y = self._naive(q_nope, q_rope, c_kv, k_rope, cache)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))

    def _naive(self, q_nope, q_rope, c_kv, k_rope, cache):
        """Up-project the latents into full per-head keys/values, then ordinary attention."""
        B, _, T, _ = q_nope.shape
        S = c_kv.shape[2]
        kv = self.w_ukv(c_kv.squeeze(1)).view(B, S, self.H, 2 * self.hd).transpose(1, 2)
        k_nope, v = kv.split([self.hd, self.hd], dim=-1)
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope.expand(-1, self.H, -1, -1)], dim=-1)
        drop = self.dropout if self.training else 0.0
        if self.cfg.attn_softcap > 0:
            return softcapped_attention(q, k, v, self.cfg.attn_softcap, dropout_p=drop, window=self.window)
        is_causal = T > 1 and (cache is None or cache.length == T) and self.window == 0
        mask = None if is_causal or (T == 1 and self.window == 0) else attention_mask(T, S, self.window, q.device)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal, dropout_p=drop)

    def _absorbed(self, q_nope, q_rope, c_kv, k_rope):
        """Attention computed directly against the cached latents (no full K/V is ever built)."""
        T, S = q_nope.shape[2], c_kv.shape[2]
        w = self.w_ukv.weight.view(self.H, 2 * self.hd, self.dc)
        w_uk, w_uv = w[:, :self.hd], w[:, self.hd:]                  # (H, head_dim, d_c) each
        q_lat = torch.einsum("bhtd,hdc->bhtc", q_nope, w_uk)        # query moved into latent space
        scores = (q_lat @ c_kv.transpose(-2, -1) + q_rope @ k_rope.transpose(-2, -1)) * self.scale
        scores = softcap(scores, self.cfg.attn_softcap)
        scores = scores.masked_fill(~attention_mask(T, S, self.window, q_nope.device), float("-inf"))
        att = F.softmax(scores.float(), dim=-1).type_as(q_nope)     # (B, H, T, S)
        o_lat = att @ c_kv                                           # weighted sum of latents
        return torch.einsum("bhtc,hdc->bhtd", o_lat, w_uv)          # up-project once, afterwards
