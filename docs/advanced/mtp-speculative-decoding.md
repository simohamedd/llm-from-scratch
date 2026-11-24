# Advanced — Self-speculative decoding with the MTP heads

**Code:** [`llm/speculative.py`](../../llm/speculative.py) · **Benchmark:** [`scripts/bench_speculative.py`](../../scripts/bench_speculative.py) · **Test:** `tests/test_19_mtp_speculative.py` · **Builds on:** [Multi-token prediction](multi-token-prediction.md), [Chapter 4](../04-inference.md)
**Used by:** DeepSeek-V3 / R1 (one MTP module as the draft, 85–90 % acceptance, ≈ 1.8× tokens/s) · MiMo-7B · GLM-4.5 · Qwen3-Next · vLLM / SGLang "MTP" speculative modes

## The idea
Plain decoding (Chapter 4) runs the whole model once per token. At batch size 1 that forward pass
is mostly fixed cost (reading the weights, kernel launches, Python), so scoring 3 tokens costs
barely more than scoring 1. **Speculative decoding** (Leviathan et al. 2023; Chen et al. 2023) uses
that: a cheap *draft* guesses the next few tokens, the real model checks them all in **one**
forward pass, and we keep the longest prefix it agrees with — plus one token of its own for free.

The usual draft is a second, smaller model. A model trained with **multi-token prediction** already
has one built in: MTP depth k predicts the token k + 1 steps ahead from the trunk's hidden state.
So drafting costs one small block per guessed token, there is no second model or tokenizer, and the
draft reuses hidden states the main model has already computed. That is *self*-speculative decoding.

## How it works
Committed tokens are t₀..t_N. The last one, t_N ("pending"), was drawn from the main model but is
not in its KV cache yet; h_{N−1} is the trunk's hidden state at the last cached position.

```
draft    depth 1: [h_{N−1} ; Emb(t_N)] → block → head → q₁ → d₁        (guess of t_{N+1})
         depth 2: [h¹_{N−1}; Emb(d₁) ] → block → head → q₂ → d₂        (guess of t_{N+2}) ...
verify   main model on [t_N, d₁, …, d_D]  — one forward pass —  → p₀ … p_D
accept   d_j with probability min(1, p_{j−1}(d_j) / q_j(d_j))      greedy: iff d_j = argmax p_{j−1}
reject   at the first failure j: next pending ~ normalize(max(0, p_{j−1} − q_j)); drop d_{j+1..}
bonus    all D accepted:          next pending ~ p_D
```

Every round emits between 1 and D + 1 tokens for one main-model forward pass. The accept/resample
rule makes the output distribution **exactly** the main model's:
`P(x) = q(x)·min(1, p(x)/q(x)) + P(reject)·resid(x) = min(p, q)(x) + (p − q)⁺(x) = p(x)`.
With greedy decoding the output is token-for-token identical to plain greedy decoding (tested for
dense, MLA, MoE and QK-norm models).

Two pieces of bookkeeping are the whole difficulty:

- **Main cache rollback.** The verify pass writes K/V for all D + 1 inputs; rejected drafts are cut
  off again with `truncate_cache(cache, L + a)`. (A rolling sliding-window buffer can't be rolled
  back, so those models are refused.)
- **One KV cache per MTP depth.** Each depth is a causal Transformer block that attends over its own
  past. Depth k at position i reads `Emb(t_{i+k})`, so its cache may only keep positions whose inputs
  were all *committed* tokens (i ≤ N − k). Each round a depth runs on the positions it has not cached
  yet (ending at N − 1, where the draft is read), then truncates back to N − k + 1 positions and
  carries the one hidden state the next depth still needs. A test re-computes every draft from
  scratch without caches and checks it is identical.

The sampling filters (temperature, top-k, top-p) are applied to **both** p and q via
`filtered_probs()`, because the guarantee only holds for the distributions actually sampled from.
Repetition penalty is not supported (it changes p after every token, inside the round).

## Run it
```bash
python scripts/prepare_data.py --vocab-size 512 --tokenizer-sample 100000
python scripts/train.py --preset nano --steps 400 --mtp-depth 2      # a model with MTP heads
python scripts/bench_speculative.py --ckpt checkpoints/best.pt                    # greedy
python scripts/bench_speculative.py --ckpt checkpoints/best.pt --temperature 0.8  # sampling
```
```python
from llm.speculative import SpecStats, mtp_speculative_stream, speculative_generate
stats = SpecStats()
ids = list(mtp_speculative_stream(model, x, max_new_tokens=100, n_draft=2, stats=stats))
stats.acceptance_rate, stats.tokens_per_call, stats.accepted_at_depth
print(speculative_generate(model, tok, "ROMEO:", max_new_tokens=100))   # greedy by default
```

## Measured result (Tiny Shakespeare, nano, vocab 512, 400 steps, CPU 2 threads)
Two checkpoints from `scripts/train.py --preset nano --steps 400`: `--mtp-depth 1` (val 2.840) and
`--mtp-depth 2` (val 2.898). 8 prompts × 100 new tokens, best of 3 timings per prompt. Plain = the
Chapter 4 `stream()` with the same sampling settings.

| checkpoint | decoding | temp | tokens / main pass | draft acceptance | per depth 1 / 2 | tokens/s | wall-clock vs plain | same text as plain |
|---|---|---|---|---|---|---|---|---|
| D=1 | plain | 0 | 1.00 | — | — | 857–890 | 1.00× | — |
| D=1 | MTP draft, 1 token | 0 | **1.68** | 70.1 % | 70 % | 875–888 | 0.98–1.04× | ✅ identical |
| D=1 | MTP draft, 1 token | 0.8 | 1.72 | 73.4 % | 73 % | 823 | 0.97× | (sampling) |
| D=2 | MTP draft, 1 token | 0 | 1.63 | 64.5 % | 64 % | 853 | 0.96× | ✅ identical |
| D=2 | MTP draft, 2 tokens | 0 | **2.04** | 54.1 % | 64 % / 44 % | 817 | 0.92× | ✅ identical |
| D=2 | MTP draft, 2 tokens | 0.8 | 2.23 | 63.8 % | 71 % / 56 % | 791 | 0.92× | (sampling) |

Cost of the pieces (nano, one token after a 50-token prefix): main forward 1 token 1.20 ms,
2 tokens 1.40 ms, 3 tokens 1.36 ms; one MTP draft step 0.32 ms.

What this shows, honestly:
- **The algorithm works as advertised:** the main model runs 1.6–2.2× fewer times per generated
  token, and greedy output is bit-identical to plain decoding. Even a 400-step nano MTP head is
  accepted 64–78 % of the time (DeepSeek-V3 reports 85–90 % for its second token).
- **No wall-clock speed-up at this scale (0.92–1.04×).** The nano trunk has only 4 blocks, so one
  MTP block is already ~27 % of a decoding step, and a multi-token verify pass costs ~1.17× a
  single-token one (explicit attention masks, more Python per round). The savings and the overhead
  cancel. The trade turns positive when the main model is large and memory-bound (a GPU reading
  billions of weights per pass: DeepSeek-V3 has 1 MTP block next to 61 trunk blocks).
- Deeper drafts get accepted less (64 % → 44 % for the second guess), so D = 2 adds tokens per
  pass but costs more drafting than it saves here — the usual reason production systems use 1–3.
- Sampling at T = 0.8 is accepted *more* often than greedy: acceptance is Σ min(p, q), which is
  forgiving when both distributions are spread out, while greedy needs the exact argmax.

## Exercises
1. **Tree drafts.** Instead of one chain, take the top-2 tokens at depth 1 and verify both branches
   in one pass with a tree attention mask (Medusa / EAGLE). How much does acceptance improve?
2. **Adaptive D.** Track the running acceptance rate and stop drafting deeper when depth k's rate
   falls below a threshold. Does it beat fixed D on wall-clock?
3. **Sliding windows.** Make the rolling KV cache roll back (keep D extra slots) so models with
   `sliding_window > 0` can be used.
4. **Batching.** Extend to batch size > 1: each row accepts a different number of tokens — you
   need per-row cache lengths (see the ROADMAP item on left-padded batched generation).
5. Repeat the benchmark with the `micro` preset on a GPU. At which model size does MTP speculative
   decoding start to pay off in tokens/s?

## References
- Y. Leviathan, M. Kalman, Y. Matias, *Fast Inference from Transformers via Speculative Decoding* (ICML 2023).
- C. Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling* (2023).
- DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024), §2.2 and §5.4.3 (MTP as speculative draft).
- T. Cai et al., *Medusa* (2024) and Y. Li et al., *EAGLE* (2024) — draft heads on the main model's features.
