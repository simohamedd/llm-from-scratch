# Changelog

One entry per daily improvement (newest first). Work lands on `develop`; merge `develop` → `main`
when you're happy with it.

## 2026-11-26 — Batched generation with left padding + key-padding attention masks
- `llm/batching.py`: `left_pad()` (pads on the left so every row ends in the same column), `padded_attention_mask()`
  (causal / sliding-window mask AND `cache.key_valid`, plus "always see yourself" so pad queries never get an all
  -inf row → NaN), `batch_stream()` (one prefill for all rows, one (B, 1) step per token, per-row stop ids, finished
  rows fed a masked pad, repetition penalty ignores pads), `batch_generate_ids()`, `generate_batch()` (text in/out,
  `<|pad|>` as pad). RoPE needs no per-row positions (scores depend only on distances). `KVCache.key_valid` (default
  `None` = no mask code runs); mask threaded through GQA / soft-capped / sliding-window / MLA (naive, absorbed,
  merged) attention. `scripts/bench_batched.py`. Opt-in: `stream()` / `generate()` / training unchanged. Limits:
  static batching, stops at the context length
- `tests/test_21_batched_generation.py` (26 tests: left_pad, mask shape / self-attend rule, padded prefill logits =
  unpadded and batched greedy = one-at-a-time for dense / SWA / local-global / soft-cap / QK-norm / MLA / MLA + q-rank +
  soft-cap / MoE + shared / YaRN, merged MLA weights, pad id irrelevant + repetition penalty, independent stops,
  context limit, seeded sampling, `generate_batch` = `generate`); full suite: 134 passed
- Measured (Tiny Shakespeare, vocab 512, nano 400 steps val 2.843, 32 prompts × 100 greedy tokens, CPU 2 threads, two
  runs): `stream()` one at a time 919–950 tok/s; batch 1 0.79–0.81× (explicit-mask overhead), batch 2 1.32–1.56×,
  4 2.53–2.55×, 8 3.86–4.43×, 16 7.01–7.37×, **32 10.4–11.2×** — generated ids identical to one-at-a-time for 32/32
  prompts at every batch size. Doc: `docs/advanced/batched-generation.md`

## 2026-11-25 — Speculative decoding with a separate draft model
- `llm/speculative.py`: `draft_speculative_stream(target, draft, …)` (drop-in for `stream()`): a small model with the
  same tokenizer drafts K tokens autoregressively on its own KV cache (first step catches up on uncached committed
  tokens, d_K never fed), one target forward verifies `[pending, d_1..d_K]`, same `accept_or_resample()` /
  `filtered_probs()` as the MTP version (exact target distribution; greedy = argmax match), bonus token when all
  accepted, both caches rolled back with `truncate_cache()`. `speculative_generate(..., draft=model)`;
  `scripts/bench_speculative.py --draft nano.pt --n-draft 1 2 4` (also times the draft alone = cost ratio c).
  Opt-in: `stream()` / `generate()` / MTP path unchanged. Limits: batch 1, no repetition penalty, no sliding-window
  layers, stops at the smaller context window. Proposal from `feature/speculative-decoding` copied to
  `docs/proposals/` and ticked
- `tests/test_20_draft_speculative.py` (10 tests: greedy output = target greedy for random dense / smaller / MLA / MoE
  drafts and K ∈ {1, 3, 7}, draft = target accepts 100 % with 1 + 4 target passes for 21 tokens, trained draft is
  accepted + saves passes, cached drafts = cache-free recomputation, end-of-context, sampled 2nd-token marginal = exact,
  errors / stop ids / `speculative_generate`); full suite: 108 passed
- Measured (Tiny Shakespeare, vocab 512, 400 steps each, target micro 4.6 M val 2.677, draft nano 0.8 M val 2.865,
  8 prompts × 100 tokens, CPU 2 threads): greedy K=1 57.9 % acceptance, 1.57 tokens / target pass, 0.82× wall-clock;
  K=2 1.94, 0.78×; K=4 2.20, 0.56×; K=6 2.29, 0.46× — text identical to plain every time. T=0.8: K=2 64.0 %,
  2.24 tokens / pass, 0.93×; K=4 2.72, 0.74×. **No wall-clock gain on CPU at this size**: nano is only ~1.9× cheaper
  than micro (c ≈ 0.52; overhead-dominated), so the Leviathan formula predicts ≈ 1.0× at best for K=1.
  Doc: `docs/advanced/draft-model-speculative-decoding.md`

## 2026-11-24 — Self-speculative decoding with the MTP heads as the draft
- `llm/speculative.py`: `mtp_speculative_stream()` (drop-in for `stream()`): MTP depths 1..D draft the next tokens
  from the trunk's hidden state, one main forward verifies `[pending, d_1..d_D]`, speculative-sampling accept /
  residual resample (`accept_or_resample()`, exact target distribution; greedy = argmax match), bonus token when all
  accepted. Main KV cache rolled back with `truncate_cache()`; one KV cache per MTP depth that only keeps positions
  whose inputs are committed tokens. `filtered_probs()` applies temperature / top-k / top-p to both p and q;
  `SpecStats` (acceptance rate, tokens per main pass, per-depth acceptance); `speculative_generate()`;
  `scripts/bench_speculative.py`. Opt-in: `stream()` / `generate()` unchanged. Limits: batch 1, no repetition penalty,
  no sliding-window layers, stops at a full context
- `tests/test_19_mtp_speculative.py` (12 tests: greedy output = plain greedy for dense / MLA / MoE / QK-norm and every
  n_draft, accepts drafts + fewer main passes on a trained tiny model, cached drafts = uncached recomputation,
  end-of-context, accept/resample reproduces p and Σ min(p, q) acceptance, sampled 2nd-token marginal = exact,
  stop ids / errors); full suite: 98 passed
- Measured (nano, vocab 512, 400 steps, 8 prompts × 100 tokens, CPU 2 threads): D=1 checkpoint greedy — 70.1 %
  acceptance, **1.68 tokens per main pass**, identical text, wall-clock 0.98–1.04× plain (T=0.8: 73.4 %, 1.72, 0.97×);
  D=2 checkpoint, 2 drafts greedy — 54.1 % (64 % / 44 % per depth), **2.04 tokens per pass**, 0.92× (T=0.8: 2.23, 0.92×).
  Fewer main passes but no wall-clock gain at nano size: one MTP block is ~27 % of a 4-block step and a multi-token
  verify costs ~1.17× a single-token one. Doc: `docs/advanced/mtp-speculative-decoding.md`

## 2026-11-23 — Multi-token prediction (MTP) auxiliary heads (DeepSeek-V3)
- `llm/mtp.py`: `MTPModule` (RMSNorm(h) ‖ RMSNorm(Emb(t_{i+k})) → Linear 2d→d → one `Block` → RMSNorm) and
  `MultiTokenPrediction` (D sequential depths sharing the main embedding + output head; depth k runs on length T − k and
  predicts t_{i+k+1}, `ignore_index=-100` respected, depths longer than the sequence skipped)
- `ModelConfig(mtp_depth, mtp_weight)` — off by default; loss = main + λ · mean MTP loss, **training only** (eval loss,
  `generate()` and checkpoints-without-MTP unchanged). Heads are built after the trunk is initialised, so the same seed gives
  an identical trunk with or without MTP. MoE aux losses of MTP blocks included; `num_params(include_mtp=False)`,
  `model.last_mtp_losses`, ` | mtp …` in the train log; `scripts/train.py --mtp-depth 1 --mtp-weight 0.3`
- `tests/test_18_mtp.py` (10 tests: default off + shared weights, identical trunk / eval output, loss = main + λ·mean,
  depth-k targets re-derived by hand, ignore-index + short sequences, gradients reach trunk + heads for dense / MoE / MLA /
  grad-checkpoint, train → generate → reload); full suite: 86 passed
- Measured (nano, vocab 512, 400 steps, CPU 2 threads), main-head val loss @400, baseline → MTP D=1 λ=0.3: seed 1337
  2.854 → 2.823, seed 7 2.832 → 2.847, seed 42 2.835 → 2.844 (means 2.840 vs 2.838 — no measurable gain at this scale, as
  the literature predicts for small models); D=2 2.834, λ=0.1 2.840 (seed 1337). Cost +29 % wall time per depth
  (116 → 150 → 182 s). Doc: `docs/advanced/multi-token-prediction.md`; follow-up added to ROADMAP (MTP as speculative draft)

## 2026-11-22 — Multi-GPU data-parallel training (`torchrun`)
- `llm/distributed.py`: `setup_distributed()` (reads torchrun's `RANK` / `LOCAL_RANK` / `WORLD_SIZE`, NCCL on GPU, gloo on
  CPU, a no-op single-process `DistEnv` otherwise), `broadcast_parameters()`, from-scratch `GradReducer` (flattens gradients
  into ~25 MB buckets in reverse parameter order, one `all_reduce` per bucket, ÷ world size), `all_reduce_mean()`,
  `sync_buffers()` (averages persistent float buffers such as MoE `expert_bias`)
- `train()` goes data-parallel only inside an initialised process group: broadcast rank 0's weights, seed `seed + rank` for
  different batches, reduce once per optimizer step after the last micro-batch (≈ DDP `no_sync`), average eval losses
  over ranks, rank 0 alone logs and saves. Single-process behaviour unchanged (rank 0 seed = old seed).
  `torchrun --nproc_per_node 2 scripts/train.py --batch-size 16` (`--backend gloo|nccl`); `--batch-size` is per rank
- `tests/test_17_distributed.py` (8 tests, 2 gloo processes: reduced grads = single-process full-batch grads for dense /
  tiny buckets / MLA, = `torch.nn.parallel.DistributedDataParallel`, `train()` keeps replicas bit-identical incl. MoE bias
  balancing + only rank 0 logs, buffer sync + mean, single-process no-op); full suite: 76 passed
- Measured (nano, vocab 512, 400 steps, 2-core CPU), val loss @400: single 1×32 2.851 / 2.849 (two seeds, 101 s / 99 s),
  DDP 2×16 2.801 (99 s), DDP 2×32 (global 64) 2.729 (195 s) — same global batch tracks the single run; no speed-up on
  2 CPU cores because one process already uses both through intra-op threads (the gain needs one device per rank).
  Caveat documented: MoE aux-loss statistics are per rank, so that gradient is not the global-batch one;
  doc: `docs/advanced/distributed-training.md`

## 2026-11-21 — Gradient checkpointing (activation recomputation)
- `llm/checkpointing.py`: from-scratch `checkpoint(fn, *args)` built on `torch.autograd.graph.saved_tensors_hooks` — forward
  swaps every saved tensor for a ticket, backward re-runs the block once (RNG state and autocast restored) and hands the
  recomputed tensors back in order (non-reentrant design: works with frozen inputs/LoRA, side outputs and `autograd.grad`);
  `is_recomputing()`, `should_checkpoint()`, `count_saved_bytes()` (bytes autograd keeps for backward)
- `ModelConfig(grad_checkpoint, checkpoint_every)` — off by default; only active in training with grad on and no KV cache;
  `scripts/train.py --grad-checkpoint --checkpoint-every 2`. MoE bias balancing skips its update during recompute and
  routes with the bias the batch originally used
- `tests/test_16_grad_checkpoint.py` (11 tests: gradients = plain backward for dense / dropout / MoE aux / MoE bias + shared /
  MLA / sliding-window + QK-norm + soft-cap, MoE bias updated once, matches `torch.utils.checkpoint`, LoRA with frozen
  embedding gets gradients, less saved memory + eval untouched, training reduces loss); full suite: 68 passed
- Measured (CPU, 2 threads): nano 400 steps on Tiny Shakespeare, same init — val loss 2.8363 for off / every-2nd / every block
  (curves identical to 4 decimals), wall time 88.8 s / 103.2 s (+16 %) / 113.3 s (+28 %). Peak extra RSS for one fwd+bwd,
  micro (T=256): B=32 1914 → 1250 → 726 MiB, B=64 3858 → 2440 → 1751 MiB (2.2–2.6× less); saved activations micro B=32
  1115.5 → 578.1 → 40.7 MiB. Not needed at nano scale; doc: `docs/advanced/gradient-checkpointing.md`

## 2026-11-20 — Warmup-Stable-Decay (WSD) learning-rate schedule
- `llm/schedules.py`: `cosine_lr()` (the existing warm-up + cosine, moved out of the trainer), `wsd_lr()` (linear warm-up,
  constant peak LR, decay over the last `decay_frac` of steps to `lr·min_lr_ratio`) and `decay_factor()` with `linear`,
  `cosine` and `1-sqrt` shapes
- `TrainConfig(schedule, decay_frac, decay_shape)` — default stays `"cosine"` (bit-for-bit the old formula, tested);
  `scripts/train.py --schedule wsd --decay-frac 0.2 --decay-shape sqrt --min-lr-ratio 0.1`
- `tests/test_15_wsd_schedule.py` (7 tests: default unchanged, warm-up / flat plateau / monotone decay to min for 3 shapes,
  shape values, edge cases + bad args rejected, WSD training reduces loss and logs the plateau LR); full suite: 57 passed
- Measured on Tiny Shakespeare (nano, vocab 512, 400 steps, 40 warm-up, AdamW 3e-3, CPU), val loss @400: cosine 2.852
  (min 0 → 2.881), WSD 1-sqrt 20 % 2.804 (min 0 → 2.813), linear 20 % 2.808, 1-sqrt 10 % 2.817; second seed cosine 2.832 vs
  WSD 2.790 — WSD ends ~0.045 lower on both seeds while trailing by ~0.07 before its decay (step 300); same wall time;
  doc: `docs/advanced/wsd-schedule.md`

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
