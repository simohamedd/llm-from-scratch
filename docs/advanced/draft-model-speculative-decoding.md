# Advanced — Speculative decoding with a draft model

**Code:** [`llm/speculative.py`](../../llm/speculative.py) (`draft_speculative_stream`) · **Benchmark:** [`scripts/bench_speculative.py`](../../scripts/bench_speculative.py) `--draft` · **Test:** `tests/test_20_draft_speculative.py` · **Builds on:** [Chapter 4](../04-inference.md), [MTP self-speculative decoding](mtp-speculative-decoding.md)
**Used by:** Google (the original paper, T5-XXL with a T5-small draft) · DeepMind Chinchilla 70B with a 4B draft (speculative sampling) · llama.cpp `--model-draft` · vLLM / SGLang / TensorRT-LLM "draft model" mode · Hugging Face `assisted generation` (`assistant_model=`) · Llama 3.2 1B / Qwen2.5 0.5B are commonly used as drafts for their big siblings

## The idea
At batch size 1 a decoding step is mostly fixed cost — every weight of the model is read to
produce one token. Scoring K + 1 tokens in one pass costs barely more than scoring 1. So: let a
**small model with the same tokenizer** guess K tokens one by one (cheap), let the **big model**
score all of them in **one** pass, keep the longest prefix it agrees with, and get one extra token
of the big model's own for free (Leviathan et al. 2023; Chen et al. 2023).

The accept / resample rule makes the output distribution **exactly** the big model's, so this is a
pure speed trick: no quality is traded. With greedy decoding the text is token-for-token identical.

The [MTP chapter](mtp-speculative-decoding.md) did the same with the model's own MTP heads as the
draft. Here the draft is an independent language model — any checkpoint trained on the same
tokenizer, no retraining of the target needed. That is the classic, most widely deployed form.

## How it works
Committed tokens t₀..t_N; the last one ("pending") was drawn from the target but is not yet in the
target's KV cache. Each round:

```
draft    feed the draft the committed tokens it hasn't cached yet → q₁ → d₁
         feed d₁ → q₂ → d₂ … up to d_K               (K small forward passes, own KV cache)
verify   target on [t_N, d₁, …, d_K]   — one forward pass —   → p₀ … p_K
accept   d_j with probability min(1, p_{j−1}(d_j) / q_j(d_j))     greedy: iff d_j = argmax p_{j−1}
reject   at the first failure j: next pending ~ normalize(max(0, p_{j−1} − q_j)); drop the rest
bonus    all K accepted: next pending ~ p_K
rollback truncate the target cache to the committed prefix (pending + accepted drafts);
         truncate the draft cache the same way — it catches up on the correction next round
```

Each round commits 1..K + 1 tokens for one target pass. Correctness:
`P(x) = q(x)·min(1, p/q) + P(reject)·resid(x) = min(p, q)(x) + (p − q)⁺(x) = p(x)`
(`accept_or_resample()`, shared with the MTP version, tested statistically).

Bookkeeping is simpler than with MTP: the draft is an ordinary causal LM over the *same* token
sequence, so one KV cache and `truncate_cache()` suffice. The last draft d_K is never fed to the
draft model — it is only needed if accepted, and then the next round's catch-up step feeds it.

**When does it pay off?** With per-token acceptance α, draft length K, and draft cost c (one draft
step / one target step), the expected speed-up is (Leviathan et al., Thm 3.8)

```
speed-up = (1 − α^(K+1)) / ((1 − α) · (c·K + 1))
```

You want α high (draft agrees with the target) and c tiny (draft ≪ target). Real deployments use
c ≈ 0.01–0.05 (a 1B draft for a 70B target) and get 2–3×.

## Run it
```bash
python scripts/prepare_data.py --vocab-size 512 --tokenizer-sample 100000
python scripts/train.py --preset nano --steps 400 && cp checkpoints/best.pt nano.pt     # the draft
python scripts/train.py --preset micro --steps 400 && cp checkpoints/best.pt micro.pt   # the target
python scripts/bench_speculative.py --ckpt micro.pt --draft nano.pt --n-draft 1 2 4
python scripts/bench_speculative.py --ckpt micro.pt --draft nano.pt --temperature 0.8
```
```python
from llm.speculative import SpecStats, draft_speculative_stream, speculative_generate
stats = SpecStats()
ids = list(draft_speculative_stream(target, draft, x, max_new_tokens=100, n_draft=2, stats=stats))
stats.acceptance_rate, stats.tokens_per_call, stats.accepted_at_depth
print(speculative_generate(target, tok, "ROMEO:", draft=draft, max_new_tokens=100))   # greedy
```
The draft must share the tokenizer (`vocab_size` is checked). Same limits as the MTP version:
batch size 1, no repetition penalty, no sliding-window layers, stops at the smaller context window.

## Measured result (Tiny Shakespeare, vocab 512, 400 steps each, CPU 2 threads)
Target = `micro` (4.6 M params, val loss 2.677), draft = `nano` (0.8 M params, val 2.865), both
from `scripts/train.py --steps 400`. 8 prompts × 100 new tokens, best of 3 timings per prompt.

| decoding | temp | tokens / target pass | draft acceptance | accepted at position 1 / 2 / 3 / 4 | tokens/s | vs plain | same text |
|---|---|---|---|---|---|---|---|
| plain (micro) | 0 | 1.00 | — | — | 500 | 1.00× | — |
| nano alone (cost reference) | 0 | — | — | — | 958 | 1.92× | — |
| draft K=1 | 0 | 1.57 | 57.9 % | 58 % | 411 | 0.82× | ✅ identical |
| draft K=2 | 0 | 1.94 | 48.3 % | 60 / 37 % | 390 | 0.78× | ✅ identical |
| draft K=3 | 0 | 2.13 | 39.1 % | 59 / 37 / 21 % | 327 | 0.65× | ✅ identical |
| draft K=4 | 0 | 2.20 | 31.2 % | 58 / 37 / 18 / 12 % | 281 | 0.56× | ✅ identical |
| draft K=6 | 0 | 2.29 | 22.6 % | 56 / 34 / 18 / 11 / 9 / 7 % | 229 | 0.46× | ✅ identical |
| plain (micro) | 0.8 | 1.00 | — | — | 455 | 1.00× | — |
| draft K=1 | 0.8 | 1.62 | 63.8 % | 64 % | 398 | 0.88× | (sampling) |
| draft K=2 | 0.8 | 2.24 | 64.0 % | 74 / 54 % | 423 | 0.93× | (sampling) |
| draft K=4 | 0.8 | 2.72 | 45.7 % | 69 / 51 / 37 / 26 % | 335 | 0.74× | (sampling) |

Cost of one forward pass after a 50-token prefix: micro 1 token 2.04 ms, 2 tokens 2.23 ms,
3 tokens 2.37 ms, 5 tokens 2.68 ms; nano 1 token 1.07 ms.

What this shows, honestly:
- **The algorithm does what it promises:** the target runs 1.6–2.7× fewer times per generated
  token, and greedy output is bit-identical to plain decoding at every K.
- **It is slower in wall-clock here (0.46–0.93×)**, and the formula says why: the nano draft is
  only ~2× cheaper than micro (c ≈ 0.52) because at this size a step is dominated by Python and
  kernel-launch overhead, not by reading weights. With α ≈ 0.58 and c ≈ 0.52 the ideal K=1
  speed-up is (1 + α) / (1 + c) ≈ 1.04 *before* the ~9 % extra cost of a 2-token verify pass and
  the 2-token catch-up step — so ≤ 1. Deeper drafts lose more: each extra draft costs c but is
  accepted less and less (58 → 37 → 18 %).
- The trade flips when c is small: a 70B target on a GPU reads 140 GB of weights per step, a 1B
  draft 2 GB (c ≈ 0.015), and there K = 4–5 is typical.
- Sampling at T = 0.8 is accepted more often than greedy (acceptance is Σ min(p, q), forgiving when
  both distributions are spread out), so it comes closest to break-even (0.93× at K = 2).
- Compared with [MTP drafting](mtp-speculative-decoding.md) on nano (70 % acceptance, 0.92–1.04×):
  the MTP head reads the target's own hidden state and is cheaper, so it agrees more and costs
  less. A separate draft wins on flexibility — any small model with the same tokenizer, no
  retraining of the big one.

## Exercises
1. **Find break-even.** Train a much smaller draft (e.g. `d_model=64, n_layers=1`) and a larger
   target (the `small` preset on a GPU). Plot the measured speed-up against the formula above.
2. **Adaptive K.** Stop drafting as soon as the draft's own top probability falls below a threshold
   (the draft is "unsure"). Does it beat any fixed K? (This is what most production systems do.)
3. **Distil the draft.** Fine-tune nano on the micro model's logits (KL loss) instead of the raw
   text (DistillSpec, Zhou et al. 2023). How much does acceptance go up?
4. **Draft tokenizer mismatch.** Make it work when the draft has a *different* tokenizer by
   re-tokenising the draft's text (Hugging Face "universal assisted generation").
5. **Batching.** Rows accept different numbers of tokens; you need per-row cache lengths (see the
   ROADMAP item on left-padded batched generation).

## References
- Y. Leviathan, M. Kalman, Y. Matias, *Fast Inference from Transformers via Speculative Decoding* (ICML 2023).
- C. Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling* (DeepMind, 2023).
- Y. Zhou et al., *DistillSpec: Improving Speculative Decoding via Knowledge Distillation* (ICLR 2024).
- H. Xia et al., *Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding* (2024).
