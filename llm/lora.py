"""Chapter 5a — LoRA: Low-Rank Adaptation for cheap fine-tuning.

Instead of updating a big weight W (d_out x d_in), freeze it and learn
    W' = W + (alpha / r) * B @ A,   A: r x d_in,  B: d_out x r,  r << d
Only A and B are trained -> often <1% of parameters. B starts at zero so the
model is unchanged at step 0. After training, `merge_lora` folds B@A back into W
so inference has zero overhead.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.lora_A.T @ self.lora_B.T) * self.scale


def apply_lora(model: nn.Module, targets=("wq", "wv"), r: int = 8, alpha: int = 16) -> nn.Module:
    for p in model.parameters():
        p.requires_grad = False
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha))
    return model


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                child.base.weight += (child.lora_B @ child.lora_A) * child.scale
                child.base.weight.requires_grad = True
                setattr(module, name, child.base)
    for p in model.parameters():
        p.requires_grad = True
    return model


def trainable_summary(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"trainable {train:,} / {total:,} ({100 * train / total:.2f}%)"
