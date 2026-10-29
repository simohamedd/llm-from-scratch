"""Chapter 6a — Weight-only int8 quantization from scratch.

Each Linear weight row is stored as int8 plus one float scale:
    w_int8 = round(w / scale),  scale = max(|w_row|) / 127
That is ~4x smaller than fp32 with very little quality loss — the same idea
(with fancier grouping) behind GGUF Q8_0, bitsandbytes LLM.int8 and AWQ/GPTQ.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.data.float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        self.register_buffer("w_int8", torch.round(w / scale).to(torch.int8))
        self.register_buffer("scale", scale)
        self.bias = linear.bias

    def forward(self, x):
        w = self.w_int8.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, w, self.bias)


def quantize_int8(model: nn.Module, skip=("lm_head",)) -> nn.Module:
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name not in skip:
                setattr(module, name, Int8Linear(child))
    return model


def model_size_mb(model: nn.Module) -> float:
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total += t.numel() * t.element_size()
    return total / 1e6
