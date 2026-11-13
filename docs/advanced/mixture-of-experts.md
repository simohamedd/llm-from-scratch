# Advanced — Mixture of Experts (MoE)

**Code:** [`llm/moe.py`](../../llm/moe.py) · **Test:** `tests/test_07_moe.py` · **Builds on:** [Chapter 2](../02-transformer.md)
**Used by:** Mixtral, DeepSeek-V3, Qwen3-MoE, Llama 4, gpt-oss

## The idea
A dense model runs every parameter for every token. An MoE layer replaces the single SwiGLU
feed-forward with **E experts** and a **router**:

```
token ─► router (linear → softmax) ─► pick top-k experts ─► weighted sum of their outputs
```

Total parameters grow with E, but each token only pays for k experts. That is how DeepSeek-V3
has 671B parameters but only ~37B active per token.

## The catch: router collapse
Left alone, the router sends almost everything to a few experts and the rest never learn. We add
the Switch-Transformer **load-balancing loss**:

```
aux = E · Σ_e  f_e · P_e      f_e = share of tokens sent to e,  P_e = mean router prob of e
```

It is smallest when load is uniform; the model adds `moe_aux_weight · aux` (default 0.01) to the
training loss. `MoEFFN.last_load` lets you inspect how many tokens each expert received.

## Sizing
`expert_hidden()` makes each expert `1/top_k` the width of the dense FFN, so *active* FFN compute per
token stays roughly equal to the dense model — a fair comparison.

## Run it
```bash
python scripts/train.py --preset nano --steps 400                          # dense
python scripts/train.py --preset nano --steps 400 --experts 4 --top-k 2    # MoE
```
```python
from llm import GPT, ModelConfig
model = GPT(ModelConfig(n_experts=8, moe_top_k=2))
```

## Measured result (Tiny Shakespeare, nano, 400 steps, CPU)
| model | total params | val loss | val ppl | time |
|---|---|---|---|---|
| dense | 0.80M | 2.856 | 17.4 | 133 s |
| MoE 4 experts, top-2 | 1.44M | 2.864 | 17.5 | 179 s |

At this tiny scale and short run, MoE **matches** dense quality but doesn't beat it, and the
Python loop over experts is slower on CPU. MoE gains show up with more data, longer training and
efficient batched expert kernels on GPU — so treat this as a correct, readable implementation to
learn from, not a speed-up for nano models.

## Exercises
- Try `top_k=1` (Switch Transformer) and `n_experts=8`.
- Add a **shared expert** that every token uses (DeepSeek-MoE).
- Replace the aux loss with DeepSeek-V3's **aux-loss-free** per-expert bias.
- Plot `last_load` per layer during training to watch the router balance.
