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

DeepSeek-MoE / DeepSeek-V3 additions (both opt-in):

* **Shared experts** (`n_shared`): a few experts that *every* token always uses, added on
  top of the routed ones. They learn the common knowledge, so routed experts can specialise.

* **Aux-loss-free balancing** (`balance="bias"`): instead of a loss term that fights the
  language-model loss, each expert gets a bias b_e that is used *only to pick* experts:

      scores  s = sigmoid(router(x))                      # per-expert affinity in (0, 1)
      chosen  = top_k(s + b)                              # bias decides who gets the token
      weights = s[chosen] / sum(s[chosen])                # but the mix uses the raw scores

  After every training forward, overloaded experts get their bias lowered and underloaded
  ones raised:  b_e -= rate * sign(load_e - mean_load).  No gradient is involved, so the
  balancing pressure never distorts the LM gradient. DeepSeek-V3 uses rate = 0.001.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpointing import is_recomputing


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
    def __init__(self, d_model: int, hidden: int, n_experts: int = 8, top_k: int = 2,
                 n_shared: int = 0, balance: str = "aux", bias_rate: float = 1e-3):
        super().__init__()
        assert 1 <= top_k <= n_experts
        assert balance in ("aux", "bias"), balance
        self.n_experts, self.top_k = n_experts, top_k
        self.balance, self.bias_rate = balance, bias_rate
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d_model, hidden) for _ in range(n_experts))
        self.shared = nn.ModuleList(Expert(d_model, hidden) for _ in range(n_shared))
        # selection-only bias for aux-loss-free balancing; a buffer, not a parameter (no gradient)
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        self.aux_loss = torch.tensor(0.0)      # refreshed on every forward
        self.last_load = torch.zeros(n_experts)  # fraction of tokens per expert (for inspection)

    def route(self, flat: torch.Tensor):
        """Return (probs, weights, idx): router probabilities, mixing weights and chosen experts."""
        logits = self.router(flat).float()
        if self.balance == "bias":
            scores = torch.sigmoid(logits)                                   # (N, E)
            # gradient checkpointing re-runs this forward after update_bias(): route with the old bias
            bias = self._routed_bias if is_recomputing() else self.expert_bias
            idx = (scores + bias).topk(self.top_k, dim=-1).indices
            weights = scores.gather(-1, idx)                                 # bias NOT in the mix
            probs = scores / scores.sum(-1, keepdim=True)
        else:
            probs = F.softmax(logits, dim=-1)                                # (N, E)
            weights, idx = probs.topk(self.top_k, dim=-1)                    # (N, k)
        weights = weights / weights.sum(-1, keepdim=True)
        return probs, weights.type_as(flat), idx

    @torch.no_grad()
    def update_bias(self, load: torch.Tensor):
        """DeepSeek-V3 rule: push each expert's bias toward the mean load by a fixed step."""
        self.expert_bias -= self.bias_rate * torch.sign(load - load.mean())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        probs, weights, idx = self.route(flat)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            rows, slot = (idx == e).nonzero(as_tuple=True)               # tokens routed to e
            if rows.numel():
                out.index_add_(0, rows, weights[rows, slot, None] * expert(flat[rows]))

        for expert in self.shared:                                       # every token, weight 1
            out = out + expert(flat)

        dispatch = F.one_hot(idx, self.n_experts).sum(1).float()        # (N, E), 1 per chosen expert
        frac_tokens = dispatch.mean(0)                                   # f_e (sums to top_k)
        self.last_load = (frac_tokens / self.top_k).detach()
        if self.balance == "bias":
            self.aux_loss = torch.zeros((), device=x.device)             # no loss term at all
            if self.training and not is_recomputing():   # gradient checkpointing re-runs forward
                self._routed_bias = self.expert_bias.clone()
                self.update_bias(self.last_load)
        else:  # Switch-Transformer load-balancing loss
            frac_probs = probs.mean(0)                                   # P_e (sums to 1)
            self.aux_loss = self.n_experts * (frac_tokens * frac_probs).sum()
        return out.view(B, T, D)


def expert_hidden(d_model: int, ffn_mult: float, n_experts: int, top_k: int, n_shared: int = 0) -> int:
    """Size experts so the *active* FFN compute per token (top_k routed + n_shared) matches the dense model."""
    dense = ffn_mult * d_model
    return 32 * max(1, math.ceil(dense / (top_k + n_shared) / 32))
