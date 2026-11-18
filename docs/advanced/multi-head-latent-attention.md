# Advanced — Multi-head Latent Attention (MLA)

**Code:** [`llm/mla.py`](../../llm/mla.py), wired into [`llm/model.py`](../../llm/model.py) · **Test:** `tests/test_10_mla.py` · **Builds on:** [Chapter 2](../02-transformer.md), [Chapter 4](../04-inference.md)
**Used by:** DeepSeek-V2 (d_c 512, d_r 64) · DeepSeek-V3 / R1 (same, 128 heads) · Kimi K2 (DeepSeek-V3 architecture) · MiniCPM3

## The idea
When a model generates text, most of the memory goes to the **KV cache**: standard attention keeps
`2 · n_kv_heads · head_dim` numbers per token per layer. GQA shrinks that by letting query heads
*share* K/V heads. MLA takes a different route: it caches a small **latent vector** per token and
reconstructs every head's keys and values from it.

```
c_kv = RMSNorm(W_dkv · x)          ← cached   (d_c numbers)
k_h  = W_uk[h] · c_kv              ← rebuilt on the fly, one per head
v_h  = W_uv[h] · c_kv
```

Every head still has its own keys and values (unlike GQA/MQA), they just live in a shared
`d_c`-dimensional subspace — a low-rank factorisation of the K/V projections.

**Decoupled RoPE.** RoPE rotates keys by position, and that rotation sits between `W_uk` and the
query, which would break the absorption trick below. So each query/key gets a small extra part that
carries the position:

```
q_h = [ q_nope_h ; RoPE(q_rope_h) ]       k_h = [ W_uk[h] · c_kv ; RoPE(k_rope) ]
```

`k_rope = W_kr · x` is **one** vector shared by all heads (MQA-style) and is cached too, so the
cache holds `d_c + d_r` numbers per token per layer. DeepSeek-V3: 512 + 64 = 576 instead of
2 · 128 · 128 = 32 768 for multi-head attention.

**Weight absorption.** For the content part of the score,
`q_nope_h · (W_uk[h] c_kv) = (W_uk[h]ᵀ q_nope_h) · c_kv`, so the query can be moved into latent
space once and dotted directly with the cached latents; values are summed as latents and
up-projected once afterwards (`W_uv[h] · Σ a_j c_j`). Full keys/values are never materialised
during decoding.

## How it works here
- `MLAttention` (in `llm/mla.py`) replaces `Attention` in every block when `mla_kv_rank > 0`.
  One linear (`w_dkv`) produces `[c_kv ; k_rope]`, `kv_norm` normalises the latent, `w_ukv`
  up-projects it to per-head `[k_nope ; v]`, and `wq` produces `[q_nope ; q_rope]`.
- The existing `KVCache` is reused unchanged: its `k` slot stores the latent `(B, 1, T, d_c)` and
  its `v` slot stores the shared rotary key `(B, 1, T, d_r)`. Sliding windows still trim it.
- Two attention paths, numerically identical (tested): `_naive` up-projects K/V and calls SDPA
  (used for training), `_absorbed` works in latent space (used by default when a cache is given in
  eval mode, i.e. generation).
- RoPE tables are built for `cfg.rope_dim` (= `mla_rope_dim` under MLA, the head size otherwise).
- Works with sliding-window attention and attention soft-capping; `qk_norm` is not supported with
  MLA. Query compression and precomputed absorbed weights are in the follow-up
  [MLA query compression & merged weights](mla-decoding.md).

## Run it
```bash
python scripts/prepare_data.py --vocab-size 512 --tokenizer-sample 100000
python scripts/train.py --preset nano --steps 400                                   # GQA baseline
python scripts/train.py --preset nano --steps 400 --mla-kv-rank 32 --mla-rope-dim 16
```
```python
from llm import GPT, ModelConfig
from llm.mla import kv_cache_numbers_per_token
cfg = ModelConfig(mla_kv_rank=64, mla_rope_dim=16)
print(kv_cache_numbers_per_token(cfg))      # 80 numbers per token per layer
model = GPT(cfg)
```
`mla_kv_rank` defaults to `0`, so the Chapter 2 model is unchanged unless you ask for it.

## Measured result
Tiny Shakespeare, nano preset (d_model 128, 4 layers, 4 query heads of 32, context 128), vocab
512, 400 steps, lr 3e-3, batch 32, CPU, one seed (1337); val loss averaged over 100 fixed
validation batches. Cache = numbers per token per layer (fp32 size after 120 decoded tokens);
decode speed = greedy decoding with the cache, batch 1, 2 CPU threads.

| attention | params | KV cache / token / layer | cache @120 tok | val loss | val ppl | decode tok/s |
|---|---|---|---|---|---|---|
| GQA, 2 KV heads (baseline) | 0.804M | 128 | 240 KiB | **2.864** | 17.5 | 868 |
| MHA, 4 KV heads | 0.870M | 256 | 480 KiB | **2.856** | 17.4 | 936 |
| MLA d_c 64, d_r 16 | 0.878M | 80 | 150 KiB | 2.906 | 18.3 | 652 |
| MLA d_c 32, d_r 16 | 0.829M | 48 | 90 KiB | 2.942 | 18.9 | 645 |
| MLA d_c 16, d_r 8 | 0.784M | 24 | 45 KiB | 3.027 | 20.6 | 610 |

What this shows, honestly:
- **MLA did not win on quality at this scale.** Every MLA variant is worse than GQA after 400
  steps (+0.04 to +0.16), and the gap grows as the latent shrinks. A rank-16 to 64 bottleneck is a
  real constraint for a 128-wide model; DeepSeek's claim (MLA ≥ MHA) is for d_c = 512 with
  head_dim 128 and long training, where the K/V matrices are far from full-rank anyway. One seed,
  short training — differences of ~0.01 are noise, 0.1 is not.
- **The cache is much smaller**: 1.6× (d_c 64) to 5.3× (d_c 16) smaller than our GQA baseline and
  3–10× smaller than MHA, at roughly the parameter count of the baseline.
- **Decoding got slower here** (~0.7×), because at this size the extra einsums cost more than the
  memory traffic they save. MLA pays off when the cache is large (long contexts, big batches,
  memory-bound GPUs) — not on a 0.8M-parameter model on a CPU.
- Training time was about the same (77–87 s for all runs).

## Exercises
- Add **query compression** (`c_q = RMSNorm(W_dq x)`, `q = W_uq c_q`, DeepSeek-V3 uses rank 1536)
  and check that the val loss / parameter trade-off changes.
- Precompute the absorbed matrices once for inference: merge `W_uv` into `wo` (`W_o[h] · W_uv[h]`)
  so the decode step skips one einsum, and time it.
- Train the `micro` preset with `--mla-kv-rank 128` vs GQA for 2 000+ steps — does the quality gap
  close with a wider latent and more training?
- Plot the singular values of a trained GQA model's `wk`/`wv` weights. How many are needed to keep
  99 % of the energy? That's the rank MLA could have used for free.
- Combine MLA with `--sliding-window 32` and measure the cache: latent compression and windowing
  multiply.
