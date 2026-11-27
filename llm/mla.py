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

Query compression (DeepSeek-V2/V3). The queries get a low-rank bottleneck too:

    c_q = RMSNorm(W_dq · x)        q = [q_nope ; q_rope] = W_uq · c_q        (d_q = mla_q_rank)

It saves parameters and training activations, not cache (queries are never cached). DeepSeek-V3
uses d_q = 1536 for d_model = 7168. Opt-in: mla_q_rank > 0; 0 keeps the full-rank query matrix wq.

Precomputed absorbed weights (decoding speed). Absorption still runs two extra einsums per step
(W_uk on the query, W_uv on the output). Both are products of fixed matrices, so for inference
they can be multiplied out once:

    W_qlat[h] = W_uk[h]ᵀ · W_qnope[h]     input (x or c_q)  -> latent query   (one matmul)
    W_ov[h]   = W_o[:, h] · W_uv[h]       latent output o_lat -> d_model       (one matmul)

`GPT.merge_mla_weights()` builds them (non-persistent buffers, so .to(device) moves them and
checkpoints don't store them); calling `.train()` drops them again because the weights will change.
Whether merging is cheaper depends on shapes: W_qlat has H·d_c·d_in entries versus
H·head_dim·(d_in + d_c) unmerged, so it pays when d_c ≲ head_dim (and costs more when d_c ≫ head_dim,
which is why real kernels sometimes keep W_uk separate).

Opt-in: ModelConfig(mla_kv_rank=d_c, mla_rope_dim=d_r, mla_q_rank=d_q). mla_kv_rank = 0 (default)
keeps the GQA attention of Chapter 2. Works with the KV cache, sliding windows and attention soft-capping.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .model import KVCache, RMSNorm, apply_rope
from .batching import padded_attention_mask
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
        self.dq = cfg.mla_q_rank
        # queries: content part (head_dim) + rotary part (rope_dim) for every head
        if self.dq > 0:   # low-rank query: x -> c_q (d_q) -> RMSNorm -> per-head queries
            self.w_dq = nn.Linear(cfg.d_model, self.dq, bias=False)
            self.q_norm = RMSNorm(self.dq)
            self.w_uq = nn.Linear(self.dq, self.H * (self.hd + self.dr), bias=False)
        else:
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
        # precomputed absorbed weights for decoding (None until merge_weights() is called)
        for name in ("w_qlat", "w_qrope", "w_ov"):
            self.register_buffer(name, None, persistent=False)

    @property
    def q_proj(self) -> nn.Module:
        """The matrix producing [q_nope ; q_rope] from the query input."""
        return self.w_uq if self.dq > 0 else self.wq

    def query_input(self, x):
        """x itself, or the normalised low-rank query latent c_q under query compression."""
        return self.q_norm(self.w_dq(x)) if self.dq > 0 else x

    @property
    def merged(self) -> bool:
        return self.w_ov is not None

    @torch.no_grad()
    def merge_weights(self):
        """Multiply out W_uk into the query and W_uv into the output projection (inference only)."""
        for name, mod in (("q_proj", self.q_proj), ("w_ukv", self.w_ukv), ("wo", self.wo)):
            if not isinstance(mod, nn.Linear):
                raise TypeError(f"{name} is a {type(mod).__name__}; merge LoRA / dequantize before merging MLA weights")
        H, hd, dr, dc = self.H, self.hd, self.dr, self.dc
        wq = self.q_proj.weight.view(H, hd + dr, -1)
        w_qnope, w_qrope = wq[:, :hd], wq[:, hd:]                    # (H, hd, d_in), (H, dr, d_in)
        w = self.w_ukv.weight.view(H, 2 * hd, dc)
        w_uk, w_uv = w[:, :hd], w[:, hd:]                            # (H, hd, d_c) each
        wo = self.wo.weight.view(-1, H, hd)                          # (d_model, H, hd)
        self.w_qlat = torch.einsum("hdc,hdi->hci", w_uk, w_qnope).reshape(H * dc, -1)
        self.w_qrope = w_qrope.reshape(H * dr, -1).clone()
        self.w_ov = torch.einsum("mhd,hdc->mhc", wo, w_uv).reshape(wo.shape[0], H * dc)
        return self

    def unmerge_weights(self):
        self.w_qlat = self.w_qrope = self.w_ov = None
        return self

    def train(self, mode: bool = True):
        if mode:                # the weights are about to change: precomputed products go stale
            self.unmerge_weights()
        return super().train(mode)

    def forward(self, x, cos, sin, cache: KVCache | None = None, absorb: bool | None = None):
        B, T, _ = x.shape
        qin = self.query_input(x)
        c_kv, k_rope = self.w_dkv(x).split([self.dc, self.dr], dim=-1)
        c_kv = self.kv_norm(c_kv).unsqueeze(1)                       # (B, 1, T, d_c)
        k_rope = apply_rope(k_rope.unsqueeze(1), cos, sin)           # (B, 1, T, d_r), shared by heads
        if cache is not None:   # the cache stores the latent and the rotary key, nothing else
            c_kv, k_rope = cache.update(self.layer_idx, c_kv, k_rope)
        # left-padded batch: causal/window mask AND "key is not a pad" (None otherwise)
        pad_mask = padded_attention_mask(T, c_kv.shape[2], self.window, cache, device=x.device)
        if absorb is None:      # default: absorbed weights for generation, naive path for training
            absorb = cache is not None and not self.training
        if absorb and self.merged:   # fastest path: two fused matmuls, no per-head einsums
            q_lat = F.linear(qin, self.w_qlat).view(B, T, self.H, self.dc).transpose(1, 2)
            q_rope = F.linear(qin, self.w_qrope).view(B, T, self.H, self.dr).transpose(1, 2)
            o_lat = self._latent_attention(q_lat, apply_rope(q_rope, cos, sin), c_kv, k_rope, pad_mask)
            return F.linear(o_lat.transpose(1, 2).reshape(B, T, -1), self.w_ov)
        q = self.q_proj(qin).view(B, T, self.H, self.hd + self.dr).transpose(1, 2)
        q_nope, q_rope = q.split([self.hd, self.dr], dim=-1)
        q_rope = apply_rope(q_rope, cos, sin)
        if absorb:
            y = self._absorbed(q_nope, q_rope, c_kv, k_rope, pad_mask)
        else:
            y = self._naive(q_nope, q_rope, c_kv, k_rope, cache, pad_mask)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))

    def _naive(self, q_nope, q_rope, c_kv, k_rope, cache, pad_mask=None):
        """Up-project the latents into full per-head keys/values, then ordinary attention."""
        B, _, T, _ = q_nope.shape
        S = c_kv.shape[2]
        kv = self.w_ukv(c_kv.squeeze(1)).view(B, S, self.H, 2 * self.hd).transpose(1, 2)
        k_nope, v = kv.split([self.hd, self.hd], dim=-1)
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope.expand(-1, self.H, -1, -1)], dim=-1)
        drop = self.dropout if self.training else 0.0
        if self.cfg.attn_softcap > 0:
            return softcapped_attention(q, k, v, self.cfg.attn_softcap, dropout_p=drop, window=self.window,
                                        mask=pad_mask)
        is_causal = T > 1 and (cache is None or cache.length == T) and self.window == 0
        mask = None if is_causal or (T == 1 and self.window == 0) else attention_mask(T, S, self.window, q.device)
        if pad_mask is not None:
            is_causal, mask = False, pad_mask
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal, dropout_p=drop)

    def _absorbed(self, q_nope, q_rope, c_kv, k_rope, pad_mask=None):
        """Attention computed directly against the cached latents (no full K/V is ever built)."""
        w = self.w_ukv.weight.view(self.H, 2 * self.hd, self.dc)
        w_uk, w_uv = w[:, :self.hd], w[:, self.hd:]                  # (H, head_dim, d_c) each
        q_lat = torch.einsum("bhtd,hdc->bhtc", q_nope, w_uk)        # query moved into latent space
        o_lat = self._latent_attention(q_lat, q_rope, c_kv, k_rope, pad_mask)
        return torch.einsum("bhtc,hdc->bhtd", o_lat, w_uv)          # up-project once, afterwards

    def _latent_attention(self, q_lat, q_rope, c_kv, k_rope, pad_mask=None):
        """softmax(q_lat·c + q_rope·k_rope) · c  — attention in latent space, returns (B, H, T, d_c)."""
        T, S = q_lat.shape[2], c_kv.shape[2]
        scores = (q_lat @ c_kv.transpose(-2, -1) + q_rope @ k_rope.transpose(-2, -1)) * self.scale
        scores = softcap(scores, self.cfg.attn_softcap)
        allowed = pad_mask if pad_mask is not None else attention_mask(T, S, self.window, q_lat.device)
        scores = scores.masked_fill(~allowed, float("-inf"))
        att = F.softmax(scores.float(), dim=-1).type_as(q_lat)      # (B, H, T, S)
        return att @ c_kv                                            # weighted sum of latents
