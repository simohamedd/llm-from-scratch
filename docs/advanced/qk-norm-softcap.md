# Advanced — QK-norm and logit soft-capping

**Code:** [`llm/stability.py`](../../llm/stability.py), wired into [`llm/model.py`](../../llm/model.py) · **Test:** `tests/test_08_stability.py` · **Builds on:** [Chapter 2](../02-transformer.md)
**Used by:** OLMo 2, Gemma 3, Qwen3 (QK-norm) · Gemma 2 (attention + final soft-capping) · Grok-1 (attention soft-capping)

## The problem
In long or large training runs the attention logits `q·k/√d` can grow without limit. Softmax then
becomes nearly one-hot, its gradient vanishes for every other key, and a few big updates make the
loss spike or diverge. The output logits can drift the same way. Two cheap fixes keep these
numbers in a sane range.

## Fix 1 — QK-norm
Apply an RMSNorm (over `head_dim`, with a learnable gain) to every query head and every key head
**before** RoPE and the dot product:

```
q = RMSNorm_q(Wq x)      k = RMSNorm_k(Wk x)      q, k = RoPE(q), RoPE(k)      softmax(q·kᵀ/√d)
```

Now `|q·k|` is bounded by the norm gains rather than by how big `Wq`/`Wk` have grown. It costs
`2 · head_dim` parameters per layer (one gain shared by all query heads, one by all key heads, the
Qwen3 / Gemma 3 layout; OLMo 2 normalises the full projection instead).

## Fix 2 — logit soft-capping
```
softcap(x) = cap · tanh(x / cap)
```
≈ identity for `|x| ≪ cap`, smoothly saturating at `±cap`. Unlike hard clipping the gradient never
becomes exactly zero. Gemma 2 uses `cap = 50` on attention scores and `cap = 30` on the final
vocabulary logits.

`F.scaled_dot_product_attention` can't run a function between `q·kᵀ` and softmax, so with
`attn_softcap > 0` the model switches to `softcapped_attention()`: the attention formula written out
by hand, with a causal mask that also handles a KV cache (`tril(S - T)`). This is slower and uses
more memory than the fused kernel — one reason Gemma 3 dropped attention soft-capping in favour of
QK-norm.

## Run it
```bash
python scripts/prepare_data.py --vocab-size 512 --tokenizer-sample 100000
python scripts/train.py --preset nano --steps 400                                     # baseline
python scripts/train.py --preset nano --steps 400 --qk-norm                           # QK-norm
python scripts/train.py --preset nano --steps 400 --attn-softcap 50 --final-softcap 30
```
```python
from llm import GPT, ModelConfig
model = GPT(ModelConfig(qk_norm=True, attn_softcap=50.0, final_softcap=30.0))
```
All three default to off, so the Chapter 2 model is unchanged unless you ask for them.

## Measured result (Tiny Shakespeare, nano 0.80M params, vocab 512, 400 steps, CPU, one seed)
Default learning rate 3e-3:

| model | val loss | val ppl | time |
|---|---|---|---|
| baseline | 2.880 | 17.8 | 112 s |
| QK-norm | 2.826 | 16.9 | 117 s |
| soft-cap (attn 50, final 30) | 2.874 | 17.7 | 128 s |
| QK-norm + soft-cap | **2.799** | **16.4** | 136 s |

Stress test at learning rate 1e-2:

| model | val loss | val ppl |
|---|---|---|
| baseline | **2.966** | 19.4 |
| QK-norm + soft-cap | 3.087 | 21.9 |

What this shows, honestly:
- At the default LR, **QK-norm gives a real gain** (−0.05 val loss); soft-capping alone does almost
  nothing, because at this scale logits never get near 50 or 30 — the caps are effectively identity.
- At 1e-2 **neither model diverges**, so there was no instability to fix, and the stabilised model
  is *worse*. A nano model with 400 steps is simply too small and too short for logit growth to
  bite; these tricks pay off in billion-parameter, long-horizon runs. Results are from a single seed,
  so differences of a few hundredths are within noise.
- The hand-written soft-capped attention costs ~10-15 % wall time on CPU versus the fused kernel.

## Exercises
- Log `max |q·k/√d|` per layer during training with and without QK-norm and plot it.
- Find the learning rate at which the baseline diverges; does QK-norm push that limit higher?
- Try OLMo 2's variant: normalise the whole `Wq x` (width `n_heads · head_dim`) before reshaping.
- Use a much smaller `final_softcap` (e.g. 5) and watch it hurt — the cap must stay above the
  logits the model needs.
