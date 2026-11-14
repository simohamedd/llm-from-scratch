# Advanced — Sliding-window attention

**Code:** [`llm/sliding_window.py`](../../llm/sliding_window.py), wired into [`llm/model.py`](../../llm/model.py) · **Test:** `tests/test_09_sliding_window.py` · **Builds on:** [Chapter 2](../02-transformer.md), [Chapter 4](../04-inference.md)
**Used by:** Mistral 7B (window 4096, every layer) · Gemma 2 (local 4096 / global, alternating) · Gemma 3 (local 1024, 5 local : 1 global) · gpt-oss (alternating banded / dense layers)

## The idea
Causal attention lets token `i` look at every earlier token, so the attention cost per token and
the KV cache grow linearly with the sequence. Sliding-window attention (SWA) lets each query see
only the most recent `window` keys:

```
token i may attend to j   iff   i - window < j <= i

window = 3     keys → 0 1 2 3 4 5
       query 3        . 1 1 1 . .
       query 5        . . . 1 1 1
```

Information can still travel further than the window: each layer passes on what the previous one
collected, so after `L` layers a token can be influenced by tokens up to `L · window` back — the
same "receptive field" argument as stacked convolutions.

## How it works here
- `attention_mask(T, S, window)` builds the `(T, S)` boolean band. The `T` new queries are always
  the **last** `T` of the `S` keys, so the same formula covers training, a single decode step and
  chunked prefill on top of a KV cache.
- `layer_window(sliding_window, global_every, i)` decides each layer's window. `global_every = 0`
  windows every layer (Mistral); `global_every = N` keeps every N-th layer global (Gemma 2: `N=2`,
  Gemma 3: `N=6`).
- `Attention` passes the banded mask to `F.scaled_dot_product_attention` (or to the hand-written
  soft-capped attention, which now also takes `window`).
- **Rolling KV cache:** `GPT.new_cache()` creates a `KVCache` that, for a windowed layer, stores
  only the last `window − 1` keys/values — everything a future token can still see. The cache
  stays a fixed size; `cache.length` still counts all tokens because RoPE needs absolute positions.
  Global layers keep their full cache. `llm/generate.py` now uses `model.new_cache()`, which is
  identical to the old cache when the window is off.

## Run it
```bash
python scripts/prepare_data.py --vocab-size 512 --tokenizer-sample 100000
python scripts/train.py --preset nano --steps 400                                   # baseline
python scripts/train.py --preset nano --steps 400 --sliding-window 32               # Mistral-style
python scripts/train.py --preset nano --steps 400 --sliding-window 16 --global-every 2   # Gemma 2-style
```
```python
from llm import GPT, ModelConfig
model = GPT(ModelConfig(sliding_window=32, global_every=2))
cache = model.new_cache()           # rolling buffer for the local layers
```
Both default to `0`, so the Chapter 2 model is unchanged unless you ask for it.

## Measured result
Tiny Shakespeare, nano preset (0.80M params, context 128), vocab 512, 400 steps, lr 3e-3, CPU,
one seed; val loss averaged over 100 fixed validation batches:

| model | val loss | val ppl |
|---|---|---|
| baseline (full attention) | 2.843 | 17.2 |
| window 32, all layers | 2.808 | 16.6 |
| window 16, all layers | **2.796** | 16.4 |
| window 16, every 2nd layer global | 2.829 | 16.9 |

KV cache after decoding 512 tokens (nano dims, 4 layers, 2 KV heads, fp32, context 512):

| cache | stored keys per layer | size |
|---|---|---|
| full attention | 512, 512, 512, 512 | 1024 KiB |
| window 32, all layers | 31, 31, 31, 31 | 62 KiB |
| window 32, every 2nd layer global | 31, 512, 31, 512 | 543 KiB |

What this shows, honestly:
- Restricting attention **did not hurt** at this scale — the windowed models are even slightly
  better (−0.03 to −0.05). Short training on 128-token windows of Shakespeare mostly rewards local
  patterns, and a narrow band is a helpful inductive bias early on. One seed, so differences of a
  few hundredths are close to noise; don't read this as "SWA beats full attention" in general —
  long-range recall is exactly what tiny Shakespeare can't test.
- The real win is **memory**: the rolling cache of an all-local model stays at `window − 1` entries
  per layer, 16× smaller here after 512 tokens and constant beyond. Interleaving global layers
  (Gemma) keeps long-range recall at roughly half the full cost.
- There is **no speed-up** in this implementation: the banded mask is applied to the full `T × S`
  score matrix, so training costs the same (wall times were equal within noise). Production kernels
  (FlashAttention `window_size`, Mistral's xFormers path) skip the masked blocks entirely.

## Exercises
- Measure the receptive field: feed a sequence, take the gradient of the last logit w.r.t. the input
  embeddings and plot which positions get a non-zero gradient for 1, 2, 4 layers with `window=8`.
- Make the decode loop in `llm/generate.py` skip the "rebuild the cache at `context_length`" step
  for an all-local model — with a rolling cache there's nothing to rebuild (you'll need a RoPE
  table that grows, or positions taken modulo a large table).
- Our dense mask still computes the full `T × S` scores. Write a chunked version that only
  multiplies each block of queries with the keys inside its band, and time it at `context_length`
  1024 with `window=64`.
- Try Gemma 3's ratio (`--global-every 6`) on the `micro` preset and compare with all-global.
