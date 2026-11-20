"""Learning-rate schedules: warm-up + cosine, and Warmup-Stable-Decay (WSD).

Cosine decay (the GPT-3 / Llama recipe) ties the *whole* schedule to the total number of steps
chosen before training starts: the LR begins to fall right after warm-up, so you can't stop early
or keep going without the schedule being wrong. Comparing token budgets means one full run each.

Warmup-Stable-Decay (MiniCPM, Hu et al. 2024; "trapezoidal", Hägele et al. 2024) splits it in three:

    lr ▲      ┌──────────────────────── stable ────────────┐
       │     /                                              \\   decay
       │    /  warm-up                                       \\  (last ~10-20 % of steps)
       └───┴────────────────────────────────────────────────┴──▶ step

  * warm-up  — linear 0 → peak, as before
  * stable   — hold the peak LR constant for most of training
  * decay    — anneal quickly to ``min_lr`` over a short final window

The stable phase doesn't depend on when you stop, so you can save a checkpoint at any point, branch
off, run only the short decay, and get a finished model for that budget — or continue the stable run
further. That's how MiniCPM, DeepSeek-V3 (multi-step), Llama 3 and many recent open models run
long pre-training and data-mixture experiments. Most of the loss drop happens *during* the decay,
which is why WSD curves look worse than cosine until the very end.

Decay shapes (``progress`` p goes 0 → 1 over the decay window, LR = min + (peak - min)·f(p)):
  * ``"linear"``  f = 1 - p
  * ``"cosine"``  f = ½(1 + cos πp)
  * ``"sqrt"``    f = 1 - √p      ("1-sqrt", Hägele et al. — drops fast, then gently; their best)
"""
from __future__ import annotations

import math

DECAY_SHAPES = ("linear", "cosine", "sqrt")
SCHEDULES = ("cosine", "wsd")


def warmup_lr(step: int, lr: float, warmup_steps: int) -> float:
    """Linear warm-up: lr·(step+1)/warmup (the first step already gets a small, non-zero LR)."""
    return lr * (step + 1) / warmup_steps


def cosine_lr(step: int, lr: float, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1) -> float:
    """Linear warm-up, then cosine decay from ``lr`` to ``lr·min_lr_ratio`` at ``max_steps``."""
    if step < warmup_steps:
        return warmup_lr(step, lr, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    min_lr = lr * min_lr_ratio
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def decay_factor(progress: float, shape: str = "sqrt") -> float:
    """f(p) in [0, 1] for p in [0, 1]: 1 at the start of the decay window, 0 at its end."""
    p = min(max(progress, 0.0), 1.0)
    if shape == "linear":
        return 1.0 - p
    if shape == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * p))
    if shape == "sqrt":
        return 1.0 - math.sqrt(p)
    raise ValueError(f"unknown decay shape {shape!r}; choose from {DECAY_SHAPES}")


def wsd_lr(step: int, lr: float, warmup_steps: int, max_steps: int, decay_frac: float = 0.2,
           min_lr_ratio: float = 0.0, decay_shape: str = "sqrt") -> float:
    """Warmup-Stable-Decay: warm-up, constant ``lr``, then decay over the last ``decay_frac`` of steps.

    The decay window is ``round(decay_frac · max_steps)`` steps ending at ``max_steps`` (it never
    eats into the warm-up). ``step == max_steps`` returns exactly ``lr·min_lr_ratio``.
    """
    if not 0.0 <= decay_frac <= 1.0:
        raise ValueError(f"decay_frac must be in [0, 1], got {decay_frac}")
    if step < warmup_steps:
        return warmup_lr(step, lr, warmup_steps)
    decay_steps = min(round(decay_frac * max_steps), max_steps - warmup_steps)
    decay_start = max_steps - decay_steps
    if step <= decay_start or decay_steps <= 0:
        return lr
    min_lr = lr * min_lr_ratio
    return min_lr + (lr - min_lr) * decay_factor((step - decay_start) / decay_steps, decay_shape)
