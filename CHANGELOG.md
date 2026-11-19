# Changelog

One entry per daily improvement (newest first). Work lands on `develop`; merge `develop` → `main`
when you're happy with it.

## 2026-11-19 — Muon optimizer for matrix weights
- `llm/muon.py`: `newton_schulz()` (5-step quintic iteration, coefficients 3.4445 / −4.7750 / 2.0315, ≈ `U·Vᵀ` with matmuls only),
  `Muon` optimiser (Nesterov momentum 0.95 → orthogonalise → `0.2·√max(m, n)` Moonlight RMS scaling, decoupled weight decay)
  that runs Muon on `use_muon` groups and a built-in AdamW on the rest; `muon_param_groups()` keeps the tied embedding/head
  and 1-D norms on AdamW
- `GPT.configure_optimizer(..., optimizer="muon")`, `TrainConfig(optimizer)`, `scripts/train.py --optimizer muon` — default stays AdamW
- `tests/test_14_muon.py` (7 tests: NS orthogonalises ill-conditioned matrices of 3 shapes, param split, AdamW branch = `torch.optim.AdamW`,
  update RMS ≈ 0.2, Muon training reduces loss, unknown optimizer rejected); full suite: 50 passed
- Measured on Tiny Shakespeare (nano, vocab 512, 400 steps, CPU, one seed), val loss: AdamW lr 1.5e-3 / 3e-3 / 6e-3 = 2.921 / 2.850 / 2.934;
  Muon lr 3e-3 / 6e-3 / 1.2e-2 = 2.870 / 2.803 / 2.778 — same LR no gain, but Muon takes a 4× larger LR and its best beats AdamW's
  best by 0.07; ~7 % slower per step on CPU; doc: `docs/advanced/muon.md`

## 2026-11-18 — MLA query compression + precomputed absorbed weights
- `llm/mla.py`: low-rank queries `c_q = RMSNorm(W_dq x)`, `q = W_uq c_q` (`ModelConfig(mla_q_rank)`, off by default = full-rank `wq`);
  `MLAttention.merge_weights()` multiplies out `W_qlat = W_ukᵀ·W_qnope` and `W_ov = W_o·W_uv` (non-persistent buffers,
  dropped by `.train()`, refused on LoRA/int8 layers) so a decode step skips both per-head einsums; shared `_latent_attention()`
- `GPT.merge_mla_weights()`; `scripts/train.py --mla-q-rank 64`
- `tests/test_13_mla_decoding.py` (5 tests: naive = absorbed = merged incl. windows/soft-cap/q-compression, cached generation,
  train() unmerges, LoRA refused, q-compressed model trains); full suite: 43 passed
- Measured on Tiny Shakespeare (nano, vocab 512, 400 steps, CPU, one seed), val loss: GQA 2.858, MLA d_c 32 2.934,
  + d_q 64 2.898, + d_q 32 2.918 (0.829M → 0.813M / 0.771M params) — query compression shows no measurable cost at this scale.
  Greedy decode tok/s absorbed → merged (2 threads): B=1 565 → 701, 475 → 605, 562 → 672; B=16 4752 → 5393,
  4241 → 5621, 5273 → 6106 — merging is 1.1–1.3× faster with identical outputs; doc: `docs/advanced/mla-decoding.md`

## 2026-11-17 — RoPE scaling: Position Interpolation, NTK-aware, YaRN
- `llm/rope_scaling.py`: `scaled_inv_freq()` / `rope_tables()` — linear (PI) `θ/s`, NTK-aware `base·s^(d/(d-2))`,
  and YaRN (per-pair ramp between keep and `θ/s` by turns-in-context, `beta_fast=32`, `beta_slow=1`, cos/sin × `0.1·ln s + 1`)
- `ModelConfig(rope_scaling, rope_factor, rope_original_context)` — off by default; `cfg.max_seq_len` sizes the RoPE table
  and the context `generate.py` keeps; `GPT.set_rope_scaling(kind, factor)` switches a trained model in place
- `scripts/eval_long_context.py` (val loss per length × method); `scripts/train.py --rope-scaling yarn --rope-factor 4`
- `tests/test_12_rope_scaling.py` (7 tests); full suite: 38 passed
- Measured on Tiny Shakespeare (nano, vocab 512, 400 steps at context 128, CPU, one seed), zero-shot val loss at
  256 / 512 / 1024 tokens: plain 2.927 / 3.075 / 3.364, linear 3.578 / 4.146 / 4.313, NTK 2.886 / 2.999 / 3.307,
  YaRN 2.924 / 2.991 / 3.134 (2.887 at 128 for all). After 100 fine-tune steps at 512 (×4), val loss @512: plain 2.812,
  linear 2.874, NTK 2.800, YaRN 2.797 — YaRN best zero-shot at 4–8×, differences after fine-tuning within noise;
  doc: `docs/advanced/rope-scaling.md`

## 2026-11-16 — Shared experts + aux-loss-free MoE balancing
- `llm/moe.py`: `MoEFFN(n_shared, balance, bias_rate)` — always-on shared experts added to the routed output, and
  DeepSeek-V3 bias routing: sigmoid scores, `top_k(s + b)` for selection only, mixing weights from raw scores,
  `b -= rate * sign(load - mean)` after each training forward (buffer, no gradient, saved in checkpoints, frozen in eval)
- `ModelConfig(moe_shared_experts, moe_balance, moe_bias_rate)` — off by default (`0`, `"aux"`, `1e-3`);
  `expert_hidden(..., n_shared)` keeps active compute at the dense budget
- `scripts/train.py --shared-experts 1 --moe-balance bias --moe-bias-rate 1e-3`
- `tests/test_11_moe_deepseek.py` (5 tests); full suite: 31 passed
- Measured on Tiny Shakespeare (nano, vocab 512, 400 steps, CPU, one seed, MoE 4/top-2), val loss / MaxVio:
  no balancing 2.820 / 0.576, Switch aux 2.795 / 0.142, bias γ=1e-3 2.825 / 0.080, bias γ=1e-2 2.787 / 0.067,
  +shared aux 2.788 / 0.111, +shared bias γ=1e-2 2.822 / 0.047 (dense 2.844) — bias routing balances better,
  loss differences are within noise at this scale; doc: `docs/advanced/deepseek-moe.md`

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
