"""Gradient (activation) checkpointing, from scratch — trade compute for memory.

Training memory is dominated not by the weights but by the *activations*: every op in the
forward pass saves its inputs/outputs so backward can compute gradients. For a Transformer that
is roughly  layers × batch × seq_len × d_model × (a dozen tensors per block)  — it grows with the
batch and the context, and is what makes long-context training run out of memory.

Checkpointing a block means: during forward, run it but **throw away** everything it saved for
backward, keeping only the block's input. When backward reaches the block, **re-run its forward**
(from the kept input) to regenerate those tensors, then continue as usual.

    without:  memory ~ L blocks × (all activations)         compute = 1 fwd + 1 bwd
    with:     memory ~ L block inputs + 1 block's activations compute ≈ 2 fwd + 1 bwd (~ +30 %)

How this implementation works (the same idea as `torch.utils.checkpoint(use_reentrant=False)`):
autograd lets us intercept every tensor an op saves for backward with
`torch.autograd.graph.saved_tensors_hooks(pack, unpack)`.

  * forward:  `pack` replaces each saved tensor by a small integer ticket — the tensor is freed.
  * backward: the first `unpack` of a ticket re-runs the block (same RNG state for dropout, same
    autocast/bf16 setting) with a second `pack` hook that *records* the saved tensors in order,
    so ticket i is again the i-th tensor saved. `unpack` hands it to autograd and forgets it.

Because the normal autograd graph is kept (only the saved tensors are swapped out), everything
just works: gradients reach parameters even when the block input needs no grad (LoRA with a
frozen embedding), side outputs such as the MoE aux loss keep their gradient, and both
`loss.backward()` and `torch.autograd.grad()` are supported.

One catch: a block with *side effects* in forward would run them twice. The only one in this
repo is MoE aux-loss-free balancing (`expert_bias` update), which checks `is_recomputing()`.

Enable it with `ModelConfig(grad_checkpoint=True)` (optionally `checkpoint_every=2` to checkpoint
only every second block). It only acts in training mode with gradients on; eval and generation
are untouched.
"""
from __future__ import annotations

import contextlib
from typing import Callable

import torch

_RECOMPUTING = 0


def is_recomputing() -> bool:
    """True while a checkpointed block is being re-run during backward (skip side effects then)."""
    return _RECOMPUTING > 0


def _autocast_state(device_type: str) -> tuple[bool, torch.dtype]:
    return torch.is_autocast_enabled(device_type), torch.get_autocast_dtype(device_type)


def checkpoint(fn: Callable, *args):
    """Run fn(*args) without storing its activations; recompute them during backward.

    fn must be deterministic given its inputs and the RNG state (true for every block here),
    so the recomputation saves exactly the same tensors, in the same order, as the first run.
    """
    device_type = next((a.device.type for a in args if torch.is_tensor(a)), "cpu")
    autocast_on, autocast_dtype = _autocast_state(device_type)
    cpu_rng = torch.get_rng_state()                       # dropout must draw the same masks twice
    cuda_rng = torch.cuda.get_rng_state() if device_type == "cuda" else None
    recomputed: dict[int, torch.Tensor] = {}             # ticket -> tensor, filled by recompute()
    n_saved = 0

    def pack(t: torch.Tensor) -> int:                     # forward: keep a ticket, drop the tensor
        nonlocal n_saved
        n_saved += 1
        return n_saved - 1

    def recompute():
        global _RECOMPUTING
        order = iter(range(n_saved))

        def record(t: torch.Tensor):                       # recompute: collect saved tensors in order
            recomputed[next(order)] = t.detach()

        # inputs are detached: the re-run builds a throw-away graph we never backprop through
        inputs = [a.detach().requires_grad_(a.requires_grad) if torch.is_tensor(a) else a for a in args]
        devices = [torch.cuda.current_device()] if cuda_rng is not None else []
        _RECOMPUTING += 1
        try:
            with torch.random.fork_rng(devices=devices):   # restore RNG, then put it back afterwards
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng)
                with torch.enable_grad(), \
                        torch.autocast(device_type, dtype=autocast_dtype, enabled=autocast_on), \
                        torch.autograd.graph.saved_tensors_hooks(record, lambda _: None):
                    fn(*inputs)
        finally:
            _RECOMPUTING -= 1

    def unpack(ticket: int) -> torch.Tensor:              # backward: recompute on first need
        if ticket not in recomputed:                       # (again on a retain_graph second pass)
            recompute()
        return recomputed.pop(ticket)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        return fn(*args)


def should_checkpoint(model: torch.nn.Module, layer_idx: int, cache) -> bool:
    """Checkpoint this block? Only when enabled, training, building a graph and not decoding."""
    cfg = model.cfg
    if not cfg.grad_checkpoint or not model.training or cache is not None or not torch.is_grad_enabled():
        return False
    return layer_idx % max(1, cfg.checkpoint_every) == 0


@contextlib.contextmanager
def count_saved_bytes():
    """Measure how many bytes autograd keeps for backward inside the block (for docs / tests).

    Yields a one-element list whose value is updated as tensors are saved. Tensors that our
    checkpoint() replaced by tickets are never seen here, because its hook is the innermost.
    """
    total = [0]
    seen: set[int] = set()

    def pack(t: torch.Tensor):
        key = t.untyped_storage().data_ptr()
        if key not in seen and not isinstance(t, torch.nn.Parameter):
            seen.add(key)
            total[0] += t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        yield total
