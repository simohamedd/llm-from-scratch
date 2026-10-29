# Chapter 4 — Inference & sampling

**Code:** [`llm/generate.py`](../llm/generate.py), [`scripts/chat.py`](../scripts/chat.py) · **Branch:** `chapter/04-inference`

Generation = repeatedly run the model, pick the next token, append it.

1. **Prefill:** process the whole prompt once, filling the KV cache.
2. **Decode:** feed one token at a time; the cache supplies the past.
3. **Sliding window:** when the context is full, keep the last `context_length` tokens.

| Knob | Effect |
|---|---|
| `temperature=0` | greedy, deterministic |
| `temperature` 0.7-1.0 | more creative |
| `top_k` | sample only from the k likeliest tokens |
| `top_p` (nucleus) | sample from the smallest set covering p of the probability |
| `repetition_penalty` | discourages loops |

```bash
python scripts/chat.py --ckpt checkpoints/best.pt
```

## Evaluating
- Validation perplexity (printed during training).
- For real benchmarks: EleutherAI `lm-evaluation-harness` (tracked in the radar).

## Exercises
- Implement **speculative decoding** with nano as draft and micro as target.
- Add beam search and compare with nucleus sampling.
