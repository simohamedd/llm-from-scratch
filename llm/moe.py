"""Mixture-of-Experts feed-forward — starter (see docs/proposals/mixture-of-experts.md)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, hidden, bias=False), nn.SiLU(), nn.Linear(hidden, d_model, bias=False))
            for _ in range(n_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.view(-1, D)
        weights, idx = F.softmax(self.router(flat), -1).topk(self.top_k, -1)
        weights = weights / weights.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):  # TODO: batched dispatch + load-balancing loss
            rows, slot = (idx == e).nonzero(as_tuple=True)
            if rows.numel():
                out[rows] += weights[rows, slot, None] * expert(flat[rows])
        return out.view(B, T, D)
