"""Chapter 6a — Weight-only int8 quantization from scratch.

Each Linear weight row is stored as int8 plus one float scale:
    w_int8 = round(w / scale),  scale = max(|w_row|) / 127
That is ~4x smaller than fp32 with very little quality loss — the same idea
(with fancier grouping) behind GGUF Q8_0, bitsandbytes LLM.int8 and AWQ/GPTQ.

Below it, ``Int4Linear`` / ``quantize_int4`` go to 4 bits with one fp16 scale per group of
32-128 weights (GGUF Q4_0 / Q4_1 style) — see docs/advanced/int4-quantization.md.
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


# ---------------------------------------------------------------------------------------------
# int4 group-wise quantization (GGUF Q4_0 / Q4_1 style)
# ---------------------------------------------------------------------------------------------
# int4 with one scale per *group* of weights instead of per row.
#
# With only 16 levels a single scale per row is too coarse: one outlier in a 4096-wide row
# would stretch the grid for every other weight. llama.cpp's Q4_0 / Q4_1 (and GPTQ / AWQ with
# ``group_size=128``) cut each row into groups of ``group_size`` consecutive input weights and
# give each group its own fp16 scale:
#
#     Q4_0 (symmetric):  d = m / -8      where m is the group's value with the largest |w|
#                        q = clamp(round(w / d), -8, 7)       w ≈ d * q
#     Q4_1 (asymmetric): d = (max - min) / 15,  q = clamp(round((w - min) / d), 0, 15)
#                        w ≈ d * q + min
#
# Q4_0's sign trick maps the extreme value exactly onto -8, so all 16 levels are used and the
# largest weight is reconstructed exactly. Two 4-bit codes are packed in each uint8 byte. Storage
# per weight: 4 bits + 16 / group_size (Q4_0, 4.5 bits at g = 32) or + 32 / group_size (Q4_1, 5.0).
# Like ``Int8Linear`` this is *weight-only*: the forward pass dequantizes to the activation dtype
# and runs an ordinary matmul (real kernels fuse the unpacking into the matmul instead).


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack unsigned 4-bit codes (values 0..15, last dim even) two per byte: low nibble = even index."""
    q = q.to(torch.uint8)
    return q[..., 0::2] | (q[..., 1::2] << 4)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of ``pack_int4``: (..., n / 2) uint8 -> (..., n) uint8 codes 0..15."""
    low, high = packed & 0x0F, packed >> 4
    return torch.stack([low, high], dim=-1).flatten(-2)


def quantize_groups_int4(w: torch.Tensor, group_size: int = 32, symmetric: bool = True):
    """Quantize a 2-D weight (out, in) group-wise along ``in``.

    Returns ``(packed, scale, offset)``: packed uint8 (out, in_padded / 2), fp16 scale
    (out, n_groups) and fp16 offset (out, n_groups) — the group minimum for Q4_1, ``None`` for Q4_0.
    ``in`` is zero-padded up to a multiple of ``group_size`` (the pad is dropped on dequantize).
    """
    if group_size < 2 or group_size % 2:
        raise ValueError("group_size must be an even number >= 2")
    out_f, in_f = w.shape
    pad = (-in_f) % group_size
    g = F.pad(w.float(), (0, pad)).view(out_f, -1, group_size)       # (out, n_groups, group_size)
    if symmetric:
        idx = g.abs().argmax(dim=-1, keepdim=True)
        m = g.gather(-1, idx)                                         # signed value with the largest |w|
        d = (m / -8.0).half().float()                                 # round the scale to fp16 *before* use
        d = torch.where(d == 0, torch.ones_like(d), d)                # all-zero group: any scale works
        q = torch.clamp(torch.round(g / d), -8, 7) + 8                # shift to 0..15
        offset = None
    else:
        lo, hi = g.amin(dim=-1, keepdim=True), g.amax(dim=-1, keepdim=True)
        lo = lo.half().float()
        d = ((hi - lo) / 15.0).half().float()
        d = torch.where(d == 0, torch.ones_like(d), d)
        q = torch.clamp(torch.round((g - lo) / d), 0, 15)
        offset = lo.squeeze(-1).half()
    packed = pack_int4(q.view(out_f, -1))
    return packed, d.squeeze(-1).half(), offset


def dequantize_groups_int4(packed, scale, offset, in_features: int, group_size: int) -> torch.Tensor:
    """Rebuild the float32 weight (out, in_features) from ``quantize_groups_int4``'s outputs."""
    out_f = packed.shape[0]
    q = unpack_int4(packed).float().view(out_f, -1, group_size)
    d = scale.float().unsqueeze(-1)
    if offset is None:
        w = (q - 8) * d
    else:
        w = q * d + offset.float().unsqueeze(-1)
    return w.view(out_f, -1)[:, :in_features]


class Int4Linear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` holding group-wise int4 weights (Q4_0 or Q4_1 style)."""

    def __init__(self, linear: nn.Linear, group_size: int = 32, symmetric: bool = True):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        if group_size <= 0:                              # one group per row (int8-style baseline)
            group_size = self.in_features + self.in_features % 2
        self.group_size, self.symmetric = group_size, symmetric
        packed, scale, offset = quantize_groups_int4(linear.weight.data, group_size, symmetric)
        self.register_buffer("w_packed", packed)
        self.register_buffer("scale", scale)
        self.register_buffer("offset", offset)          # None for Q4_0 (not saved)
        self.bias = linear.bias

    def dequantize(self, dtype=torch.float32) -> torch.Tensor:
        w = dequantize_groups_int4(self.w_packed, self.scale, self.offset, self.in_features, self.group_size)
        return w.to(dtype)

    @property
    def weight(self) -> torch.Tensor:
        """Read-only dequantized weight, for code that reads ``.weight`` directly (e.g. MLA's absorbed path)."""
        return self.dequantize()

    def forward(self, x):
        return F.linear(x, self.dequantize(x.dtype), self.bias)

    def extra_repr(self) -> str:
        kind = "Q4_0" if self.symmetric else "Q4_1"
        return f"in={self.in_features}, out={self.out_features}, group_size={self.group_size}, {kind}"


def quantize_int4(model: nn.Module, group_size: int = 32, symmetric: bool = True,
                  skip=("lm_head", "router")) -> nn.Module:
    """Replace every ``nn.Linear`` (except names in ``skip``) by an ``Int4Linear``, in place.

    ``group_size`` = consecutive input weights sharing one scale; ``0`` = one scale per row.

    The output head and MoE routers are kept in full precision by default: the head is often
    tied to the embedding, and a router's tiny logit differences decide which expert runs.
    """
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name not in skip:
                setattr(module, name, Int4Linear(child, group_size, symmetric))
    return model


def bits_per_weight(model: nn.Module) -> float:
    """Average storage bits per weight over all quantized Linear layers (codes + scales + offsets).

    Counts what is actually stored, so zero-padding of a partial last group is included.
    """
    bits = n = 0
    for m in model.modules():
        if isinstance(m, Int4Linear):
            n += m.in_features * m.out_features
            tensors = (m.w_packed, m.scale, m.offset)
        elif isinstance(m, Int8Linear):
            n += m.w_int8.numel()
            tensors = (m.w_int8, m.scale)
        else:
            continue
        bits += sum(t.numel() * t.element_size() * 8 for t in tensors if t is not None)
    return bits / max(n, 1)
