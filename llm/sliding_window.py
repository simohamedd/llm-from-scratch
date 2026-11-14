"""Advanced — sliding-window (local) attention, as in Mistral 7B and Gemma 2/3.

Plain causal attention lets token i look at *every* earlier token 0..i, so compute per token
and the KV cache both grow with the sequence length. Sliding-window attention (SWA) restricts
each query to the most recent `window` keys:

        token i may attend to j   iff   i - window < j <= i

Example, window = 3 (1 = allowed):

        keys →  0 1 2 3 4 5
    query 0     1 . . . . .
    query 1     1 1 . . . .
    query 2     1 1 1 . . .
    query 3     . 1 1 1 . .
    query 4     . . 1 1 1 .
    query 5     . . . 1 1 1

Information still travels further than `window`: layer L can see tokens up to L * window back,
because each layer forwards what the previous one gathered (like receptive fields in a CNN).

Two consequences that real models exploit:

  * Rolling KV cache (Mistral): a windowed layer never needs keys older than window - 1
    positions, so the cache can be trimmed to a fixed size — memory stops growing with length.
    KVCache(..., windows=...) in model.py (via GPT.new_cache()) does that.
  * Local/global interleaving (Gemma 2: 1 local : 1 global, Gemma 3: 5 local : 1 global): most
    layers are cheap and local, a few full-attention layers keep long-range recall.
    ModelConfig(global_every=N) makes every N-th layer global.

Everything is opt-in: ModelConfig(sliding_window=0) (the default) is the plain Chapter 2 model.
"""
from __future__ import annotations

import torch


def layer_window(sliding_window: int, global_every: int, layer_idx: int) -> int:
    """Window size for one layer: 0 means full causal attention.

    global_every = 0 -> every layer is windowed (Mistral 7B).
    global_every = N -> layers N-1, 2N-1, ... are global, the rest local (Gemma 2: N=2, Gemma 3: N=6).
    """
    if sliding_window <= 0:
        return 0
    if global_every > 0 and (layer_idx + 1) % global_every == 0:
        return 0
    return sliding_window


def attention_mask(T: int, S: int, window: int = 0, device=None) -> torch.Tensor:
    """Boolean (T, S) mask, True = may attend. Causal, optionally limited to the last `window` keys.

    The T queries are the *last* T positions of the S keys (S > T when a KV cache holds earlier
    tokens), so query i sits at key index S - T + i. This stays true for a trimmed rolling cache,
    because the cache always keeps the most recent keys.
    """
    q_pos = torch.arange(T, device=device)[:, None] + (S - T)    # (T, 1) query index among keys
    k_pos = torch.arange(S, device=device)[None, :]              # (1, S)
    allowed = k_pos <= q_pos                                     # causal
    if window > 0:
        allowed = allowed & (q_pos - k_pos < window)             # only the last `window` keys
    return allowed
