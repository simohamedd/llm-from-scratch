# Chapter 2 — The Transformer

**Code:** [`llm/model.py`](../llm/model.py), [`llm/config.py`](../llm/config.py) · **Branch:** `chapter/02-transformer`

We build the same architecture family as Llama 3, Mistral and Qwen — not the 2017 original.

```
tokens → Embedding → [ Block × N ] → RMSNorm → Linear (tied) → logits
Block:   x + Attention(RMSNorm(x))   then   x + SwiGLU(RMSNorm(x))
```

| Component | What it does | Why modern models use it |
|---|---|---|
| **Pre-norm RMSNorm** | rescales activations by their RMS | cheaper than LayerNorm, stable deep training |
| **RoPE** | rotates Q and K by a position-dependent angle | relative positions, extrapolates to longer contexts |
| **Causal self-attention** | each token mixes information from earlier tokens | the core of the model |
| **Grouped-Query Attention** | several Q heads share one K/V head | KV cache 2-8× smaller → faster inference |
| **SwiGLU FFN** | gated MLP: `down(silu(gate(x)) * up(x))` | better quality per parameter than ReLU/GELU |
| **Weight tying** | input embedding = output projection | fewer params for small models |
| **KV cache** | stores past K/V during generation | each new token costs O(n) not O(n²) |

`F.scaled_dot_product_attention` automatically uses FlashAttention kernels when available.

## Sizes (`PRESETS`)
| preset | params | hardware |
|---|---|---|
| nano | ~0.9M | laptop CPU, minutes |
| micro | ~4.7M | laptop / free Colab GPU |
| small | ~27M | single GPU |
| base | ~100M | GPT-2-small scale, a good GPU |

## Exercises
- Set `n_kv_heads = n_heads` (plain multi-head) and `n_kv_heads = 1` (multi-query). Compare loss and speed.
- Replace RoPE with learned absolute position embeddings.
- Add a Mixture-of-Experts FFN (see roadmap).
