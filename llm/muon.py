"""Advanced — Muon: momentum + orthogonalisation for matrix weights.

AdamW updates every number on its own: each weight gets its own adaptive step size. But most of a
transformer's parameters live in 2-D matrices (attention and MLP projections), and a matrix update
`ΔW` acts on the hidden state as a *linear map*. The gradient of such a matrix is usually very
low-rank — a few directions dominate and the rest get almost no learning signal.

Muon (MomentUm Orthogonalized by Newton-Schulz, Keller Jordan 2024) fixes that per matrix:

    B  ← μ·B + G                      heavy-ball momentum on the raw gradient
    M  ← G + μ·B                      Nesterov look-ahead
    O  ← NewtonSchulz5(M) ≈ U·Vᵀ      where M = U·S·Vᵀ  (all singular values pushed to ≈1)
    W  ← W·(1 - lr·wd) - lr·scale·O

`U·Vᵀ` keeps the *directions* of the momentum but gives every direction the same strength, so
rare-but-useful directions get learned as fast as the dominant ones. Computing an exact SVD is too
slow, so a 5-step quintic Newton-Schulz iteration approximates it with only matmuls.

Muon is only defined for matrices. Embeddings, the output head and 1-D gains/biases keep AdamW
(Keller's recipe and Moonlight's both do this), so this file implements one optimiser that runs
Muon on "hidden" matrices and AdamW on everything else.

`scale = 0.2·√max(rows, cols)` (Moonlight, Liu et al. 2025) makes the update's RMS match a typical
AdamW update, so Muon can reuse AdamW's learning rate, weight decay and schedule unchanged.
"""
from __future__ import annotations

import math

import torch

# quintic coefficients tuned by Keller Jordan: converge fast for singular values in (0, 1]
NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal factor U·Vᵀ of G (shape [rows, cols]) with matmuls only.

    Each step applies the odd polynomial p(X) = aX + b(XXᵀ)X + c(XXᵀ)²X, which acts on every
    singular value separately: σ ← aσ + bσ³ + cσ⁵. Starting from σ ≤ 1 (we divide by the Frobenius
    norm), a few steps push all σ into roughly [0.7, 1.2] — not exactly 1, but good enough and cheap.
    """
    a, b, c = NS_COEFFS
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:                       # work on the wide side: XXᵀ is the smaller Gram matrix
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return (X.T if transposed else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for groups with `use_muon=True`, AdamW for the rest (embeddings, head, norms).

    Every group reads its own `lr` and `weight_decay`, so the trainer's warm-up + cosine schedule
    (which sets `g["lr"]` on all groups) works unchanged.
    """

    def __init__(self, param_groups, lr: float = 3e-3, weight_decay: float = 0.0, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, betas=(0.9, 0.95), eps: float = 1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, betas=betas, eps=eps, use_muon=False)
        super().__init__(param_groups, defaults)
        for g in self.param_groups:
            if g["use_muon"] and any(p.dim() != 2 for p in g["params"]):
                raise ValueError("Muon groups may only contain 2-D weight matrices")

    @staticmethod
    def update_scale(p: torch.Tensor) -> float:
        """Moonlight's RMS matching: an orthogonal [m, n] matrix has RMS 1/√max(m, n)."""
        return 0.2 * math.sqrt(max(p.shape))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            lr, wd = g["lr"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                p.mul_(1 - lr * wd)                              # decoupled weight decay (both branches)
                if g["use_muon"]:
                    buf = state.setdefault("momentum", torch.zeros_like(p))
                    buf.mul_(g["momentum"]).add_(p.grad)
                    m = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                    p.add_(newton_schulz(m, g["ns_steps"]), alpha=-lr * self.update_scale(p))
                else:                                            # plain AdamW
                    b1, b2 = g["betas"]
                    if not state:
                        state["t"] = 0
                        state["m"], state["v"] = torch.zeros_like(p), torch.zeros_like(p)
                    state["t"] += 1
                    state["m"].lerp_(p.grad, 1 - b1)
                    state["v"].lerp_(p.grad.square(), 1 - b2)
                    m_hat = state["m"] / (1 - b1 ** state["t"])
                    v_hat = state["v"] / (1 - b2 ** state["t"])
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(g["eps"]), value=-lr)
        return loss


def muon_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Split a GPT's parameters: hidden 2-D matrices → Muon, everything else → AdamW.

    The token embedding and the (tied) output head are matrices too, but they map tokens in and
    out rather than hidden → hidden, and Muon is known to work worse on them, so they stay on AdamW.
    """
    io = {id(p) for name, p in model.named_parameters() if name.startswith(("tok_emb", "lm_head"))}
    muon, adam_decay, adam_no_decay = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 2 and id(p) not in io:
            muon.append(p)
        elif p.dim() >= 2:
            adam_decay.append(p)
        else:
            adam_no_decay.append(p)
    groups = [{"params": muon, "use_muon": True, "weight_decay": weight_decay},
              {"params": adam_decay, "use_muon": False, "weight_decay": weight_decay},
              {"params": adam_no_decay, "use_muon": False, "weight_decay": 0.0}]
    return [g for g in groups if g["params"]]
