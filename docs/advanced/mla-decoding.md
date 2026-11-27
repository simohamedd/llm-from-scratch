# Advanced — MLA query compression & precomputed absorbed weights

**Code:** [`llm/mla.py`](../../llm/mla.py) (`mla_q_rank`, `merge_weights()`), [`llm/model.py`](../../llm/model.py) (`GPT.merge_mla_weights()`) · **Test:** `tests/test_13_mla_decoding.py` · **Builds on:** [Multi-head Latent Attention](multi-head-latent-attention.md)
**Used by:** DeepSeek-V2 (d_q 1536) · DeepSeek-V3 / R1 (d_q 1536, d_model 7168) · Kimi K2 · MiniCPM3 — and every serious MLA inference engine (vLLM, SGLang, FlashMLA) precomputes the absorbed matrices

## The idea
Two follow-ups to [MLA](multi-head-latent-attention.md):

**1. Query compression.** DeepSeek squeezes the *queries* through a low rank too:

```
c_q = RMSNorm(W_dq · x)                (d_q numbers)
[q_nope ; q_rope] = W_uq · c_q         (per head: head_dim + rope_dim)
```

Queries are never cached, so this does **not** shrink the KV cache. It saves parameters and the
activation memory of training (`d_model × H·(head_dim + rope_dim)` becomes
`d_model × d_q + d_q × H·(head_dim + rope_dim)`). DeepSeek-V3: d_q = 1536 for d_model = 7168.

**2. Precomputed absorbed weights.** MLA's absorption trick moves the query into latent space
(`W_uk[h]ᵀ q_nope`) and up-projects the attention output afterwards (`W_uv[h] · o_lat`). During
decoding that is still two per-head einsums per layer per token. But every factor is a fixed
matrix, so they can be multiplied out once after training:

```
W_qlat[h] = W_uk[h]ᵀ · W_qnope[h]     input -> latent query      (d_c × d_in per head)
W_ov[h]   = W_o[:, h] · W_uv[h]       latent output -> d_model   (d_model × d_c per head)
```

A decode step then becomes: one matmul for the latent query, one for the rotary query, attention
against the cached latents, one matmul out. Per-head keys/values are never built and the per-head
up/down einsums disappear.

Is it always cheaper? No. `W_qlat` has `H·d_c·d_in` entries versus `H·head_dim·(d_in + d_c)`
unmerged, so merging the query side pays when `d_c ≲ head_dim`. DeepSeek-V3 has d_c 512 and
head_dim 128, so a merged `W_qlat` would be ~3× more FLOPs than keeping `W_uk` separate on the
query side — engines merge the output side (`W_o·W_uv`) and often keep `W_uk` separate. Here
(d_c 32, head_dim 32) merging both is a win.

## How it works here
- `ModelConfig(mla_q_rank=d_q)` (default `0` = the full-rank `wq` of MLA, unchanged). With `d_q > 0`
  `MLAttention` has `w_dq`, `q_norm` (RMSNorm) and `w_uq` instead of `wq`; `q_proj` /
  `query_input()` hide the difference from the rest of the layer.
- `GPT.merge_mla_weights()` switches to eval mode and calls `MLAttention.merge_weights()` in each
  layer: it stores `w_qlat`, `w_qrope`, `w_ov` as **non-persistent buffers** — they follow
  `.to(device)`, but are not written to checkpoints. In absorbed mode (cache + eval, i.e.
  generation) the layer then uses them instead of the einsum path.
- `.train()` drops the merged copies (the weights are about to change); `.eval()` keeps them, so
  `generate()` / `stream()` work as before. Merging on top of LoRA or int8 layers is refused with a
  clear error — merge LoRA / dequantize first.
- All three paths — naive (full K/V + SDPA), absorbed (einsums), merged — give the same output up
  to float error, with sliding windows, soft-capping and query compression (tested).

## Run it
```bash
python scripts/prepare_data.py --vocab-size 512 --tokenizer-sample 100000
python scripts/train.py --preset nano --steps 400 --mla-kv-rank 32 --mla-rope-dim 16 --mla-q-rank 64
```
```python
from llm.trainer import load_checkpoint
model = load_checkpoint("checkpoints/best.pt")
model.merge_mla_weights()                           # inference only; .train() undoes it
```

## Measured result
Tiny Shakespeare, nano preset (d_model 128, 4 layers, 4 query heads of 32, context 128), vocab
512, 400 steps, lr 3e-3, batch 32, CPU, one seed (1337); val loss averaged over 100 validation
batches. Decode speed: greedy decoding with the cache, 8-token prompt + 112 new tokens, best of 3,
2 CPU threads; "absorbed" = einsum path, "merged" = after `merge_mla_weights()`.

| attention | params | val loss | val ppl | tok/s B=1 absorbed → merged | tok/s B=16 absorbed → merged |
|---|---|---|---|---|---|
| GQA, 2 KV heads (baseline) | 0.804M | **2.858** | 17.4 | — | — |
| MLA d_c 32, d_r 16 | 0.829M | 2.934 | 18.8 | 565 → 701 (1.24×) | 4752 → 5393 (1.13×) |
| MLA d_c 32 + d_q 64 | 0.813M | 2.898 | 18.1 | 475 → 605 (1.27×) | 4241 → 5621 (1.33×) |
| MLA d_c 32 + d_q 32 | 0.771M | 2.918 | 18.5 | 562 → 672 (1.20×) | 5273 → 6106 (1.16×) |

What this shows, honestly:
- **Merging speeds decoding up by ~1.1–1.3×** on CPU at this size, with identical outputs.
  Absolute tok/s on a shared 2-core machine are noisy (±10 %); the ratio within a row is the
  meaningful number.
- **Query compression did not hurt here** — d_q 64 / 32 gave 2.898 / 2.918 vs 2.934 without it, while
  removing 2–7 % of the parameters. With one seed and 400 steps, differences of ~0.03 are within
  noise; the safe reading is "no measurable cost", not "it helps".
- MLA is still behind GQA at this scale (see the [MLA doc](multi-head-latent-attention.md)); none of
  this changes the KV cache size (48 numbers / token / layer for d_c 32 + d_r 16).

## Exercises
- Count FLOPs per decode step for the absorbed and merged paths as a function of `d_c` and
  `head_dim`. At what `d_c / head_dim` does merging the *query* side stop paying? Add an option to
  merge only `W_o·W_uv` and time both settings with `--mla-kv-rank 128`.
- Merge `w_qrope` and `w_qlat` into one matrix (one matmul instead of two) and time it.
- Train `micro` with `--mla-kv-rank 128 --mla-q-rank 128` for 2 000 steps — does query compression
  start to cost quality when training is longer?
- Make `merge_weights()` work on top of LoRA by merging the adapters first (`llm/lora.py: merge_lora`).
- Store the merged matrices in bf16 and check how far the logits move.
