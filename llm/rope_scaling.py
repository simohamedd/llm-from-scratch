"""Advanced — RoPE scaling: run a model on longer sequences than it was trained on.

RoPE rotates each (even, odd) pair i of a query/key by the angle  pos · θ_i,  θ_i = base^(-2i/d).
Fast pairs (small i) spin many times inside the training context; slow pairs (large i) turn less
than one full circle. Past the training length L the slow pairs reach angles the model has never
seen, and quality collapses. Three fixes, all of which only change the frequencies θ_i:

  * linear / Position Interpolation (Chen et al. 2023):  θ_i' = θ_i / s
        squeeze positions back into [0, L). Every pair slows down by s — including the fast ones
        that encode local word order, which blurs neighbouring tokens.
  * NTK-aware (bloc97 2023):  base' = base · s^(d / (d - 2))
        raise the base instead: the slowest pair is slowed by ~s, the fastest pair barely changes.
  * YaRN (Peng et al. 2023): "NTK-by-parts" + attention temperature.
        Count how many full turns r_i = L · θ_i / 2π pair i makes inside the training context:
            r_i > beta_fast (32)  -> keep θ_i       (extrapolate: already seen every angle)
            r_i < beta_slow (1)   -> θ_i / s        (interpolate like PI)
            in between            -> linear blend
        and multiply cos/sin by  m = 0.1 · ln(s) + 1, which scales every attention logit q·k by m²
        — softmax gets sharper, compensating for the flatter scores at long range.

s = factor = new_length / L. The functions here only build the (cos, sin) tables, so the
attention code is untouched. Opt-in: ModelConfig(rope_scaling="yarn", rope_factor=4) or, at
inference on an already trained model, `model.set_rope_scaling("yarn", 4)`.
"""
from __future__ import annotations

import math

import torch

KINDS = ("", "linear", "ntk", "yarn")


def base_inv_freq(dim: int, theta: float) -> torch.Tensor:
    """θ_i = theta^(-2i/dim) for i = 0 .. dim/2 - 1 (the plain RoPE frequencies)."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))


def yarn_correction_range(dim: int, theta: float, original_len: int,
                          beta_fast: float = 32.0, beta_slow: float = 1.0) -> tuple[float, float]:
    """Pair indices (low, high) where pairs make beta_fast resp. beta_slow turns within original_len.

    Solves  original_len · theta^(-2i/dim) / 2π = r  for i.
    """
    def pair_for(turns: float) -> float:
        return dim * math.log(original_len / (turns * 2 * math.pi)) / (2 * math.log(theta))
    low = max(math.floor(pair_for(beta_fast)), 0)
    high = min(math.ceil(pair_for(beta_slow)), dim // 2 - 1)
    return low, high


def yarn_mscale(factor: float) -> float:
    """YaRN attention temperature: cos/sin are multiplied by this (logits by its square)."""
    return 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0


def scaled_inv_freq(dim: int, theta: float, kind: str = "", factor: float = 1.0, original_len: int = 0,
                    beta_fast: float = 32.0, beta_slow: float = 1.0) -> tuple[torch.Tensor, float]:
    """Return (inv_freq, mscale) for the chosen scaling. kind '' or factor 1 = plain RoPE."""
    assert kind in KINDS, f"rope_scaling must be one of {KINDS}"
    inv = base_inv_freq(dim, theta)
    if not kind or factor == 1.0:
        return inv, 1.0
    if kind == "linear":
        return inv / factor, 1.0
    if kind == "ntk":
        return base_inv_freq(dim, theta * factor ** (dim / (dim - 2))), 1.0
    # yarn: blend interpolated (inv / s) and original (inv) frequencies per pair
    assert original_len > 0, "yarn needs the original training context length"
    low, high = yarn_correction_range(dim, theta, original_len, beta_fast, beta_slow)
    ramp = ((torch.arange(dim // 2).float() - low) / max(high - low, 1e-3)).clamp(0, 1)
    # ramp = 0 for fast pairs (keep), 1 for slow pairs (interpolate)
    inv = inv * (1 - ramp) + (inv / factor) * ramp
    return inv, yarn_mscale(factor)


def rope_tables(dim: int, max_len: int, theta: float, kind: str = "", factor: float = 1.0,
                original_len: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) of shape (max_len, dim/2), ready for model.apply_rope."""
    inv, mscale = scaled_inv_freq(dim, theta, kind, factor, original_len)
    freqs = torch.outer(torch.arange(max_len).float(), inv)
    return freqs.cos() * mscale, freqs.sin() * mscale
