"""Chapter 2 — A modern decoder-only Transformer (Llama-style) from scratch.

Components, each a small class you can read top to bottom:
  * RMSNorm                 — cheaper LayerNorm used by Llama, Mistral, Qwen...
  * Rotary embeddings (RoPE) — relative position information baked into Q/K
  * Grouped-Query Attention — fewer K/V heads than Q heads -> smaller KV cache
  * SwiGLU feed-forward     — gated MLP used by nearly every 2024+ open model
  * KV cache                — makes generation O(n) instead of O(n^2)
  * optional Mixture of Experts FFN (cfg.n_experts > 0, see moe.py)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .moe import MoEFFN, expert_hidden


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


def rope_frequencies(head_dim: int, max_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)            # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D). Rotate each (even, odd) pair of dims by a position-dependent angle.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class KVCache:
    """Stores past keys/values per layer so each new token only computes itself."""

    def __init__(self, n_layers: int):
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers

    @property
    def length(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        if self.k[layer] is not None:
            k = torch.cat([self.k[layer], k], dim=2)
            v = torch.cat([self.v[layer], v], dim=2)
        self.k[layer], self.v[layer] = k, v
        return k, v


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.cfg, self.layer_idx = cfg, layer_idx
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, self.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.wo = nn.Linear(self.n_heads * self.hd, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache: KVCache | None = None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)
        # GQA: each K/V head is shared by n_heads / n_kv query heads
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        # causal mask is only needed when processing more than one new token
        is_causal = T > 1 and (cache is None or cache.length == T)
        mask = None
        if T > 1 and not is_causal:   # prefill on top of an existing cache
            S = k.shape[2]
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(S - T)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.ffn_mult * cfg.d_model)
        hidden = 32 * math.ceil(hidden / 32)   # round for hardware efficiency
        self.w_gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, layer_idx)
        self.norm2 = RMSNorm(cfg.d_model)
        if cfg.n_experts > 0:
            hidden = expert_hidden(cfg.d_model, cfg.ffn_mult, cfg.n_experts, cfg.moe_top_k)
            self.ffn = MoEFFN(cfg.d_model, hidden, cfg.n_experts, cfg.moe_top_k)
        else:
            self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)   # pre-norm residual
        x = x + self.ffn(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = rope_frequencies(cfg.head_dim, cfg.context_length * 2, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # GPT-2 trick: scale residual projections by 1/sqrt(2 * n_layers)
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "w_down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                cache: KVCache | None = None, all_logits: bool = False):
        B, T = idx.shape
        start = cache.length if cache is not None else 0
        assert start + T <= self.rope_cos.shape[0], "sequence longer than RoPE table"
        cos, sin = self.rope_cos[start:start + T], self.rope_sin[start:start + T]
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, cos, sin, cache)
        x = self.norm(x)
        if targets is None:
            if all_logits:
                return self.lm_head(x), None
            return self.lm_head(x[:, -1:, :]), None    # only the last position is needed
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        if self.cfg.n_experts > 0 and self.training:
            loss = loss + self.cfg.moe_aux_weight * self.moe_aux_loss()
        return logits, loss

    def moe_aux_loss(self) -> torch.Tensor:
        """Mean load-balancing loss over MoE layers (from the most recent forward)."""
        losses = [b.ffn.aux_loss for b in self.blocks if isinstance(b.ffn, MoEFFN)]
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)

    def configure_optimizer(self, lr: float, weight_decay: float = 0.1, betas=(0.9, 0.95)):
        # no weight decay on norms / 1-D params (standard practice)
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
