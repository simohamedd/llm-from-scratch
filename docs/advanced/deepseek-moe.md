# Advanced — Shared experts & aux-loss-free load balancing (DeepSeek-MoE / V3)

**Code:** [`llm/moe.py`](../../llm/moe.py) · **Test:** `tests/test_11_moe_deepseek.py` · **Builds on:** [Mixture of Experts](mixture-of-experts.md)
**Used by:** DeepSeek-MoE, DeepSeek-V2/V3/R1 (shared experts + bias balancing), Qwen2-MoE / Qwen1.5-MoE (shared experts), Kimi K2, GLM-4.5 (V3-style bias routing)

## The idea
Two small changes to the plain MoE layer, both from DeepSeek:

**1. Shared experts.** A few experts that *every* token always runs, added to the routed output:

```
y = Σ_shared expert_s(x)  +  Σ_{e in top_k} w_e · expert_e(x)
```

Common knowledge (syntax, frequent words) lives in the shared experts once, instead of being
copied into every routed expert, so routed experts are free to specialise.

**2. Aux-loss-free balancing.** The Switch load-balancing loss works, but it is a second
objective whose gradient pulls against the language-model loss. DeepSeek-V3 drops it and gives
each expert a **bias that is only used to choose experts**:

```
s       = sigmoid(router(x))           affinity of each expert, in (0, 1)
chosen  = top_k(s + b)                 bias decides who gets the token
w       = s[chosen] / Σ s[chosen]      mixing weights use the raw scores — no bias
after each training step:  b_e ← b_e − γ · sign(load_e − mean load)
```

Overloaded experts become a little less attractive, idle ones a little more, by a fixed step γ
(DeepSeek-V3: 0.001). The bias is a buffer, not a parameter: no gradient, no extra loss term,
and it is saved in checkpoints. Eval-mode forwards never change it.

## How it is built
- `MoEFFN(..., n_shared=1)` adds `self.shared`, a list of ordinary SwiGLU experts.
- `MoEFFN(..., balance="bias", bias_rate=1e-3)` switches `route()` to sigmoid scores + biased
  selection, sets `aux_loss` to 0 and calls `update_bias(last_load)` after every training forward.
- `expert_hidden(..., n_shared)` sizes each expert at `1/(top_k + n_shared)` of the dense FFN so
  active compute per token stays at the dense budget (DeepSeek's "fine-grained" experts).

Everything is off by default; the plain MoE and dense model behave exactly as before.

## Run it
```bash
python scripts/train.py --preset nano --steps 400 --experts 4 --top-k 2                                  # Switch aux loss
python scripts/train.py --preset nano --steps 400 --experts 4 --top-k 2 --moe-balance bias               # aux-loss-free
python scripts/train.py --preset nano --steps 400 --experts 4 --top-k 2 --shared-experts 1 --moe-balance bias
```
```python
from llm import GPT, ModelConfig
model = GPT(ModelConfig(n_experts=8, moe_top_k=2, moe_shared_experts=1, moe_balance="bias"))
```

## Measured result (Tiny Shakespeare, nano, vocab 512, 400 steps, CPU, one seed)
MaxVio = `max_e load_e / mean_load − 1` on 20 validation batches, averaged over layers (0 = perfectly balanced).

| model | total params | val loss | MaxVio | time |
|---|---|---|---|---|
| dense | 0.80M | 2.844 | — | 148 s |
| MoE 4/top-2, no balancing | 1.44M | 2.820 | 0.576 | 175 s |
| MoE 4/top-2, Switch aux loss | 1.44M | 2.795 | 0.142 | 178 s |
| MoE 4/top-2, bias γ=0.001 | 1.44M | 2.825 | 0.080 | 176 s |
| MoE 4/top-2, bias γ=0.01 | 1.44M | 2.787 | 0.067 | 175 s |
| + 1 shared, Switch aux loss | 1.25M | 2.788 | 0.111 | 178 s |
| + 1 shared, bias γ=0.001 | 1.25M | 2.820 | 0.068 | 177 s |
| + 1 shared, bias γ=0.01 | 1.25M | 2.822 | 0.047 | 175 s |

What this shows, honestly:
- **Balance:** bias routing balances load better than the aux loss (MaxVio 0.05–0.08 vs 0.11–0.14),
  and without any balancing one expert gets ~1.6× its fair share. This part reproduces the paper.
- **Quality:** all MoE variants land within ~0.04 of each other (2.79–2.83). With one seed and 400
  steps that spread is noise; we can't claim the bias method or shared experts improve loss here.
  DeepSeek's gains were measured at billions of tokens.
- γ = 0.001 is tuned for long runs; in a 400-step run the bias can only move 0.4 in total, so the
  larger γ = 0.01 is a better fit at this scale.
- Shared experts make experts narrower (hidden 192 → 128) and the model 0.2M params smaller at equal
  active compute.

## Exercises
- Log `expert_bias` per layer during training and plot it: which experts are being held back?
- Add DeepSeek-V3's tiny *sequence-wise* balance loss (weight ~1e-4) on top of the bias routing.
- Try 8 fine-grained experts, top-4, 1 shared vs 4 experts, top-2 at the same active compute.
- Compare `top_k=1` with and without a shared expert — does the shared expert rescue top-1 routing?
- Make the bias update use the load summed over gradient-accumulation micro-batches instead of each one.
