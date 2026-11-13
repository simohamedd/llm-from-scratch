# Changelog

One entry per daily improvement (newest first). Work lands on `develop`; merge `develop` → `main`
when you're happy with it.

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
