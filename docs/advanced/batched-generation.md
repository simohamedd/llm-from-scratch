# Advanced — Batched generation with left padding and attention masks

**Code:** [`llm/batching.py`](../../llm/batching.py) (`batch_stream`, `generate_batch`, `left_pad`, `padded_attention_mask`) · **Benchmark:** [`scripts/bench_batched.py`](../../scripts/bench_batched.py) · **Test:** `tests/test_21_batched_generation.py` · **Builds on:** [Chapter 4](../04-inference.md), [sliding-window attention](sliding-window-attention.md)
**Used by:** every inference server — Hugging Face `generate()` with `padding_side="left"`, vLLM / TGI / SGLang (continuous batching on top of this), llama.cpp `--parallel`, the OpenAI / Anthropic APIs behind the scenes

## The idea
`stream()` from Chapter 4 generates for **one** prompt. At batch size 1 a decoding step multiplies
a single vector by every weight matrix. That is limited by **reading the weights from memory**,
not by arithmetic. Stack B prompts into a (B, d) matrix and the weights are read once for B tokens,
so tokens per second grow almost linearly with B until compute becomes the bottleneck.

The problem is that prompts have different lengths and a tensor has to be rectangular. The fix
comes in two parts: **left padding**, so every row ends in the same column, and a **key-padding
mask**, so the pad tokens can't be seen.

## How it works
**Left padding.** Pads go on the left, so the last column holds every row's last real token:

```
col →   0      1      2     3     4
row 0  <pad>  <pad>   To    be    ,        ← short prompt
row 1   O     Romeo   ,    Romeo  !        ← long prompt
```

The next token of every row is read from the last column, and each decoding step appends one
column to all rows. The KV cache stays one rectangular `(B, H, S, D)` tensor. With right padding,
each row's last real token would sit in a different column and new tokens would land after the pads.

**Key-padding mask.** `cache.key_valid` is a `(B, S_total)` boolean tensor stored next to the KV
cache (`False` = pad). Every attention layer (GQA, soft-capped, sliding-window, MLA naive /
absorbed / merged) combines it with its usual mask:

```
allowed[b, i, j] = causal(i, j) [and in-window(i, j)]  and  key_valid[b, j]      or  j == i
```

The `or j == i` matters. A pad *query* in the first columns has no real key to its left, and a
softmax over a row that is all `-inf` gives NaN. Once a NaN is in a pad position's values, it
spreads to real tokens in later layers, because `0 · NaN = NaN`. Letting every position attend to
itself avoids that. A real query never sits on a pad key, so its result doesn't change.

**Why RoPE needs no fix.** Row 0's first real token sits at absolute position 2, not 0. RoPE
rotates q by mθ and k by nθ, so q·k depends only on m − n. Shifting a whole prompt right by p pads
leaves every attention score unchanged. Padded output equals unpadded output up to rounding (the
RoPE tables are float32, so ≈1e-7 relative). Models with *learned absolute* positions (GPT-2) have
to pass per-row position ids instead.

**Repetition penalty** looks at each row's previous tokens. Pads are replaced by the row's first
real token before the penalty is applied, so the pad id is never penalised. Penalising a token
twice is the same as penalising it once, so the duplicates are harmless.

**Finished rows.** A row that samples a stop token is marked done. It keeps riding along in the
batch, fed a masked-out pad, until every row is done or the budget runs out. This is *static
batching*. *Continuous batching* (vLLM, Orca) instead evicts finished rows and admits new requests
mid-flight (see the exercises).

Everything is opt-in. `cache.key_valid` is `None` by default, and then no mask code runs, so
`stream()` / `generate()` and training are bit-for-bit unchanged.

## Run it
```python
from llm.batching import generate_batch
texts = generate_batch(model, tok, ["ROMEO:", "JULIET:\nO", "HAMLET:\nTo be"], max_new_tokens=80)
```

```bash
python scripts/prepare_data.py --vocab-size 512 --tokenizer-sample 100000
python scripts/train.py --preset nano --steps 400
python scripts/bench_batched.py --ckpt checkpoints/best.pt --batch-sizes 1 2 4 8 16 32
```

## Measured result
Tiny Shakespeare, vocab 512, nano (0.8 M params, val loss 2.843 @ 400 steps), 32 prompts
(16 Shakespeare openings × 2, 2–8 tokens), 100 greedy tokens each, CPU with 2 threads, float32.
"same" = generated ids identical to running `stream()` on each prompt alone. Two runs:

| mode | tokens / s (run 1 / run 2) | speed-up | same |
|---|---|---|---|
| `stream()` one prompt at a time | 919 / 950 | 1.00× | — |
| batch 1 | 729 / 772 | 0.79–0.81× | 32/32 |
| batch 2 | 1429 / 1253 | 1.32–1.56× | 32/32 |
| batch 4 | 2339 / 2400 | 2.53–2.55× | 32/32 |
| batch 8 | 4068 / 3672 | 3.86–4.43× | 32/32 |
| batch 16 | 6771 / 6659 | 7.01–7.37× | 32/32 |
| batch 32 | 10297 / 9898 | 10.4–11.2× | 32/32 |

- **Greedy output was identical to one-at-a-time decoding at every batch size** (all 32 prompts).
  As a check that the mask matters: without it, a tiny random test model's padded-row prefill logits
  are off by up to 0.12.
- At this size, a step is almost entirely per-call overhead (Python, kernel launches), so batching
  amortises it very well: 32 rows cost about 3× one row.
- **Batch 1 is ~20 % slower than `stream()`**. That is the cost of building the explicit
  `(B, 1, T, S)` mask and the pad bookkeeping every step, where `stream()` uses SDPA's mask-free
  path. A real server would fall back to `stream()` for a single request, or use a fused kernel
  (FlashAttention's varlen API takes sequence lengths instead of a mask).
- Padding wastes compute in proportion to the length spread. Here the prompts differ by only a
  few tokens, compared with 100 generated tokens, so the waste is negligible. With very different
  prompt lengths, *packing* (varlen attention) or *paged attention* avoids it.

## Limits
- Static batch: finished rows still use a slot until the whole batch ends.
- Stops when the padded sequence reaches the context length (no rolling rebuild like `stream()`).
- Prefill is padded to the longest prompt, and the mask is rebuilt every step (O(B·S) memory).

## Exercises
1. Remove the `| self_key` term in `padded_attention_mask` and run the tests. Which configurations
   (if any) get NaN, and does it depend on the attention kernel (SDPA vs the hand-written soft-capped one)?
2. Switch to *right* padding. What breaks in decoding, and what would you have to track per row to
   fix it (hint: per-row positions and per-row write slots in the cache)?
3. Plot tokens/s against batch size for `--preset micro`. Where does the curve bend, and why
   (memory-bound vs compute-bound)?
4. Use `generate_batch` in `llm/serve.py`: collect requests for up to 20 ms, then run them as one
   batch (a "micro-batching" server).
5. Continuous batching: when a row finishes, replace it with a new prompt. You need to prefill the
   new prompt and splice its K/V into the cache row, with its own pad count.

## References
- Hugging Face Transformers, "Generation with LLMs: padding side" (left padding for decoder-only models)
- Yu et al. 2022, *Orca: A Distributed Serving System for Transformer-Based Generative Models* (iteration-level / continuous batching)
- Kwon et al. 2023, *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM)
- Su et al. 2021, *RoFormer* (RoPE's relative-position property)
