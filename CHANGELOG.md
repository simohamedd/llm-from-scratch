# Changelog

One entry per daily improvement (newest first). Work lands on `develop`; merge `develop` → `main`
when you're happy with it.

## 2026-11-15 — Multi-head Latent Attention
- `llm/mla.py`: `MLAttention` — cached latent `c_kv` (RMSNorm'd low-rank down-projection), per-head K/V up-projection,
  decoupled RoPE with one shared rotary key, and a weight-absorbed decode path that never builds full K/V (equal to the naive path, tested)
- `ModelConfig(mla_kv_rank, mla_rope_dim)` — off by default (`mla_kv_rank=0`); `cfg.rope_dim` sizes the RoPE table;
  reuses `KVCache` (latent in the `k` slot, rotary key in `v`), works with sliding windows and attention soft-capping
- `kv_cache_numbers_per_token(cfg)`; `scripts/train.py --mla-kv-rank 32 --mla-rope-dim 16`
- `tests/test_10_mla.py` (4 tests); full suite: 26 passed
- Measured on Tiny Shakespeare (nano, vocab 512, 400 steps, CPU, one seed), val loss: GQA 2.864, MHA 2.856,
  MLA d_c 64 2.906, d_c 32 2.942, d_c 16 3.027 — MLA is worse at this scale; KV cache per token per layer
  128 (GQA) → 80 / 48 / 24 numbers (1.6–5.3× smaller); decoding ~0.7× slower on CPU at this size;
  doc: `docs/advanced/multi-head-latent-attention.md`

## 2026-11-14 — Sliding-window attention
- `llm/sliding_window.py`: `attention_mask(T, S, window)` (banded causal mask, KV-cache aware) and `layer_window()` for local/global layer patterns
- `ModelConfig(sliding_window, global_every)` — off by default; works with SDPA and the soft-capped attention path
- Rolling KV cache: `KVCache(n_layers, windows)` / `GPT.new_cache()` keeps only `window − 1` keys per local layer; `generate.py` uses it
- `scripts/train.py --sliding-window 32 --global-every 2`
- `tests/test_09_sliding_window.py` (6 tests); full suite: 22 passed
- Measured on Tiny Shakespeare (nano, vocab 512, 400 steps, CPU, one seed), val loss: full 2.843, window 32 2.808,
  window 16 2.796, window 16 + every-2nd-layer global 2.829 — no quality loss at this scale; KV cache after 512
  tokens 1024 KiB → 62 KiB (all local) / 543 KiB (alternating); no speed-up (dense masked scores); doc:
  `docs/advanced/sliding-window-attention.md`

## 2026-11-13 — QK-norm + logit soft-capping
- `llm/stability.py`: `softcap()` (cap·tanh(x/cap)) and hand-written `softcapped_attention()` with KV-cache-aware causal mask
- `ModelConfig(qk_norm, attn_softcap, final_softcap)` — all off by default; per-head RMSNorm on Q/K before RoPE; `GPT.logits()` applies the final cap
- `scripts/train.py --qk-norm --attn-softcap 50 --final-softcap 30`
- `tests/test_08_stability.py` (5 tests); full suite: 16 passed
- Measured on Tiny Shakespeare (nano, vocab 512, 400 steps, CPU, one seed), val loss: baseline 2.880,
  QK-norm 2.826, soft-cap 2.874, both 2.799. At lr 1e-2 nothing diverges and both is worse
  (3.087 vs 2.966) — no instability to fix at this scale; doc: `docs/advanced/qk-norm-softcap.md`

## 2026-11-13 — Mixture of Experts
- `llm/moe.py`: SwiGLU experts, top-k router, Switch-Transformer load-balancing loss, `last_load` inspection
- `ModelConfig(n_experts, moe_top_k, moe_aux_weight)`; `GPT` adds the aux loss during training
- `scripts/train.py --experts N --top-k K`
- `tests/test_07_moe.py` (4 tests); full suite: 11 passed
- Measured on Tiny Shakespeare (nano, 400 steps, CPU): dense val loss 2.856 vs MoE-4/top-2 2.864 —
  equal quality at this scale; doc: `docs/advanced/mixture-of-experts.md`

## 2026-10-29 — v0.1.0
- Chapters 1-7, LLM Radar weekly bot, chapter and feature branches
