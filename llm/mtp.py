"""Multi-Token Prediction (MTP) auxiliary heads — DeepSeek-V3 style.

A normal LM learns only "predict the next token". MTP adds D small extra modules that, at every
position i, also predict the tokens 2, 3, ..., D+1 steps ahead. The extra losses are a denser
training signal: the trunk's hidden state must already "plan" a little further than one token.

DeepSeek-V3 (2024, §2.2) keeps the *causal chain* between depths, unlike Gloeckle et al. 2024
(independent parallel heads). Module k (k = 1..D) at position i combines

    h^{k-1}_i            the previous depth's hidden state (depth 0 = the main trunk output)
    Emb(t_{i+k})         the embedding of the token k steps ahead (known at training time)

    h'_i = M_k [ RMSNorm(h^{k-1}_i) ; RMSNorm(Emb(t_{i+k})) ]      (Linear 2d → d)
    h^k  = TRM_k(h')                                                 (one Transformer block)
    p    = softmax(Head(RMSNorm(h^k)))   →  predicts t_{i+k+1}

The embedding table and output head are *shared* with the main model; each depth owns only a
projection, a block and three RMSNorms. Each depth sees one position fewer (the last k positions
have no target), so depth k runs on length T − k.

    L = L_main + λ · mean_k L_mtp^k           (DeepSeek-V3: D = 1, λ = 0.3 then 0.1)

The loss is only added in training mode; eval/generation run the main model alone, so validation
loss stays comparable to a model without MTP. At inference the modules can be dropped — or used
as a draft for speculative decoding (each depth guesses one more token).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .model import Block, RMSNorm


class MTPModule(nn.Module):
    """One prediction depth: merge (hidden, next-token embedding) → Transformer block → normed hidden."""

    def __init__(self, cfg: ModelConfig, depth: int):
        super().__init__()
        self.depth = depth
        self.hnorm = RMSNorm(cfg.d_model)
        self.enorm = RMSNorm(cfg.d_model)
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.block = Block(cfg, cfg.n_layers + depth - 1)   # its own layer index (never uses the KV cache)
        self.norm = RMSNorm(cfg.d_model)                     # final norm before the shared head

    def forward(self, h: torch.Tensor, emb: torch.Tensor, cos, sin) -> torch.Tensor:
        """h, emb: (B, T', d). Returns the new hidden state h^k (un-normed, fed to the next depth)."""
        x = self.proj(torch.cat([self.hnorm(h), self.enorm(emb)], dim=-1))
        return self.block(x, cos, sin)


class MultiTokenPrediction(nn.Module):
    """D sequential MTP modules sharing the main model's embedding and output head."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.mtp_depth > 0
        self.depths = nn.ModuleList(MTPModule(cfg, k + 1) for k in range(cfg.mtp_depth))

    def __len__(self) -> int:
        return len(self.depths)

    def losses(self, h: torch.Tensor, idx: torch.Tensor, targets: torch.Tensor,
               embed: nn.Module, logits_fn, cos, sin) -> list[torch.Tensor]:
        """Cross-entropy of every depth.

        h:       (B, T, d) main trunk output *before* the final norm
        idx:     (B, T) input tokens t_0..t_{T-1};  targets: (B, T) = t_1..t_T (−100 = ignore)
        Depth k at position i (i < T − k) reads Emb(idx[i + k]) and predicts targets[i + k] = t_{i+k+1}.
        """
        T = idx.shape[1]
        out = []
        for mod in self.depths:
            k = mod.depth
            if T - k <= 0:          # sequence too short for this depth
                break
            n = T - k
            h = mod(h[:, :n], embed(idx[:, k:]), cos[:n], sin[:n])
            logits = logits_fn(mod.norm(h))
            tgt = targets[:, k:]
            out.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100))
        return out
