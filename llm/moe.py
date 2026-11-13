"""Mixture-of-Experts feed-forward (Mixtral / DeepSeek / Qwen-MoE style), from scratch.

Instead of one big SwiGLU, each layer holds `n_experts` small SwiGLU experts and a
router. Every token is sent to its `top_k` highest-scoring experts, and their
outputs are mixed with the (renormalised) router weights. Total parameters grow
with the number of experts, but compute per token only grows with `top_k`.

Without help, the router collapses onto a few favourite experts. We add the
Switch-Transformer load-balancing loss:

    aux = n_experts * sum_e( f_e * P_e )

f_e = fraction of tokens routed to expert e, P_e = mean router probability of e.
It is minimal (= top_k) when load is uniform. The model adds `moe_aux_weight * aux`
to the training loss.

Enable it with `ModelConfig(n_experts=8, moe_top_k=2)`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """One SwiGLU expert — the same block as the dense FFN, just smaller."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoEFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d_model, hidden) for _ in range(n_experts))
        self.aux_loss = torch.tensor(0.0)      # refreshed on every forward
        self.last_load = torch.zeros(n_experts)  # fraction of tokens per expert (for inspection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        probs = F.softmax(self.router(flat).float(), dim=-1)            # (N, E)
        weights, idx = probs.topk(self.top_k, dim=-1)                    # (N, k)
        weights = (weights / weights.sum(-1, keepdim=True)).type_as(flat)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            rows, slot = (idx == e).nonzero(as_tuple=True)               # tokens routed to e
            if rows.numel():
                out.index_add_(0, rows, weights[rows, slot, None] * expert(flat[rows]))

        # Switch-Transformer load-balancing loss
        dispatch = F.one_hot(idx, self.n_experts).sum(1).float()        # (N, E), 1 per chosen expert
        frac_tokens = dispatch.mean(0)                                   # f_e (sums to top_k)
        frac_probs = probs.mean(0)                                       # P_e (sums to 1)
        self.aux_loss = self.n_experts * (frac_tokens * frac_probs).sum()
        self.last_load = (frac_tokens / self.top_k).detach()
        return out.view(B, T, D)


def expert_hidden(d_model: int, ffn_mult: float, n_experts: int, top_k: int) -> int:
    """Size experts so the *active* FFN compute per token matches the dense model."""
    dense = ffn_mult * d_model
    return 32 * max(1, math.ceil(dense / top_k / 32))
