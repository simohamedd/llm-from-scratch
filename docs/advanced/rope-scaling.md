# Advanced — RoPE scaling (Position Interpolation, NTK-aware, YaRN)

**Code:** [`llm/rope_scaling.py`](../../llm/rope_scaling.py), wired into [`llm/model.py`](../../llm/model.py) · **Eval:** [`scripts/eval_long_context.py`](../../scripts/eval_long_context.py) · **Test:** `tests/test_12_rope_scaling.py` · **Builds on:** [Chapter 2](../02-transformer.md), [Chapter 4](../04-inference.md)
**Used by:** Qwen2 / Qwen2.5 (YaRN ×4, 32K → 128K) · DeepSeek-V2 / V3 (YaRN ×40, 4K → 128K) · gpt-oss (YaRN to 128K) · Llama 3.1 (its own frequency-dependent "llama3" scaling, same by-parts idea) · Position Interpolation was introduced on LLaMA by Meta

## The idea
RoPE (Chapter 2) rotates each (even, odd) pair `i` of a query/key by the angle `pos · θ_i` with
`θ_i = base^(-2i/d)`. Fast pairs spin many times inside the training context; slow pairs turn less
than once. Ask the model to read past its training length `L` and the slow pairs show it angles it
has never seen — perplexity climbs. Every fix below changes **only the frequencies `θ_i`**; the
attention code stays the same and no weights are added. With `s = new_length / L`:

| method | new frequencies | intuition |
|---|---|---|
| linear (Position Interpolation) | `θ_i / s` | squeeze positions `0 … sL` back into `0 … L` — but fast pairs slow down too, so neighbouring tokens blur |
| NTK-aware | `base' = base · s^(d/(d-2))` | raise the base: the slowest pair is slowed by exactly `s`, the fastest pair is unchanged, smooth in between |
| YaRN ("NTK-by-parts" + temperature) | per pair: keep, interpolate, or blend | pairs that make > 32 turns in `L` keep `θ_i`, pairs that make < 1 turn get `θ_i / s`, a linear ramp in between; then multiply cos/sin by `m = 0.1·ln s + 1` |

The YaRN temperature: multiplying both `q` and `k` rotations by `m` multiplies every attention
logit by `m²`. Long contexts spread the softmax over more keys and flatten it; the slightly larger
logits sharpen it back. For nano (`head_dim 32`, `L = 128`) the ramp runs from pair 0 to pair 6:
pair 0 keeps its frequency, pairs 1–5 blend, pairs 6–15 are fully interpolated.

## How it works here
- `scaled_inv_freq(dim, theta, kind, factor, original_len)` returns the new `θ_i` and the YaRN
  `mscale`; `rope_tables(...)` turns that into the `(cos, sin)` tables `apply_rope` already uses.
- `ModelConfig(rope_scaling="yarn", rope_factor=4, rope_original_context=128)` builds a model with
  scaled RoPE from the start (e.g. for long-context fine-tuning). `cfg.max_seq_len` is then
  `original × factor`, the RoPE table is sized to it, and `llm/generate.py` keeps that many tokens.
- `model.set_rope_scaling("yarn", 4)` switches a **trained** model in place — the zero-shot
  "train short, run long" case. `max_len=` only enlarges the table, so you can also measure plain
  RoPE past its training length.
- Works with GQA, sliding windows and MLA (where only the decoupled `mla_rope_dim` dims rotate).
  KV-cache decoding gives the same logits as a full forward (tested).
- Off by default (`rope_scaling=""`): the Chapter 2 tables are bit-for-bit unchanged.

## Run it
```bash
python scripts/prepare_data.py --vocab-size 512 --tokenizer-sample 100000
python scripts/train.py --preset nano --steps 400                        # trained at 128 tokens
python scripts/eval_long_context.py --ckpt checkpoints/best.pt --lengths 128 256 512 1024
```
```python
from llm.trainer import load_checkpoint
model = load_checkpoint("checkpoints/best.pt").set_rope_scaling("yarn", 4)   # 128 → 512 tokens
```
To fine-tune at the longer length with scaling on, train with
`--rope-scaling yarn --rope-factor 4 --rope-original-context 128` (and a dataset at 512 tokens).

## Measured result
Tiny Shakespeare, nano preset (0.80M params), vocab 512, 400 steps at context 128, lr 3e-3, CPU,
one seed. Val loss (ppl) averaged over **all** positions of 64 fixed validation windows; factor
`s = length / 128`. **Zero-shot** — no training at the longer length:

| RoPE | 128 | 256 (s=2) | 512 (s=4) | 1024 (s=8) |
|---|---|---|---|---|
| plain (extrapolate) | 2.887 (17.9) | 2.927 (18.7) | 3.075 (21.7) | 3.364 (28.9) |
| linear (PI) | 2.887 (17.9) | 3.578 (35.8) | 4.146 (63.2) | 4.313 (74.7) |
| NTK-aware | 2.887 (17.9) | **2.886 (17.9)** | 2.999 (20.1) | 3.307 (27.3) |
| YaRN | 2.887 (17.9) | 2.924 (18.6) | **2.991 (19.9)** | **3.134 (23.0)** |

Then **100 fine-tuning steps at 512 tokens** (factor 4, batch 8, lr 1e-3) from the same
checkpoint, one run per method:

| RoPE during fine-tune + eval | val loss @128 | val loss @512 |
|---|---|---|
| plain | 2.881 | 2.812 |
| linear (PI) | 2.937 | 2.874 |
| NTK-aware | 2.870 | 2.800 |
| YaRN | **2.868** | **2.797** |

What this shows, honestly:
- **Plain RoPE degrades but doesn't collapse** at 2–8×: our model is tiny and Shakespeare is mostly
  local, so most of the loss comes from short-range patterns. Real LLMs fall off much harder.
- **Linear interpolation without fine-tuning is the worst option** — it slows the fast pairs that
  encode local order. That is why the PI paper always fine-tunes afterwards.
- **NTK-aware and YaRN both beat plain RoPE zero-shot**; YaRN wins clearly at 4× and 8×
  (3.134 vs 3.364 at 1024 tokens). NTK is marginally best at 2×.
- After a short fine-tune at the long length all methods land within ~0.02 of each other except PI;
  YaRN is best, but by less than the noise of one seed. The fine-tune also trains on more data, so
  "plain" improves too — compare the methods with each other, not with the zero-shot table.

## Exercises
- Report loss only on positions ≥ 128 (the "new" part of each window) instead of the average — the
  gaps between methods get larger.
- Implement **dynamic NTK** (used by many inference servers): recompute `base'` from the *current*
  sequence length at every step, so short prompts use plain RoPE. What does that do to the KV cache?
- Implement Llama 3.1's `"llama3"` scaling (wavelength thresholds `low_freq_factor=1`,
  `high_freq_factor=4`) and compare it with YaRN — it is YaRN's ramp without the temperature.
- Sweep YaRN's `beta_fast` / `beta_slow` and the temperature coefficient (0.1) at 1024 tokens.
- DeepSeek-V3 also scales the softmax by `mscale²` *inside* MLA. Add that to `llm/mla.py` and check
  whether it changes anything for `--mla-kv-rank 32`.

## References
- Chen et al., *Extending Context Window of Large Language Models via Positional Interpolation* (2023)
- bloc97, *NTK-Aware Scaled RoPE* (r/LocalLLaMA, 2023)
- Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models* (2023)
