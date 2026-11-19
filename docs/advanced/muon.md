# Advanced — the Muon optimizer

**Code:** [`llm/muon.py`](../../llm/muon.py), wired into `GPT.configure_optimizer()` and [`llm/trainer.py`](../../llm/trainer.py) · **Test:** `tests/test_14_muon.py` · **Builds on:** [Chapter 3](../03-pretraining.md)
**Used by:** Kimi K2 (MuonClip, 1T params) · Moonlight (Moonshot, 16B MoE) · GLM-4.5 · the modded-nanoGPT speedrun records

## The idea
AdamW treats every weight as an independent number. But ~all of a transformer's parameters are
2-D matrices (`wq`, `wk`, `wv`, `wo`, the SwiGLU projections), and an update `ΔW` to a matrix is a
linear map applied to the hidden state. Gradients of these matrices are typically *very low-rank*:
a handful of singular directions carry almost all the magnitude, so a plain momentum update mostly
pushes along those few directions and barely learns the rest.

Muon (**M**oment**U**m **O**rthogonalized by **N**ewton-Schulz, Keller Jordan 2024) takes the
momentum matrix `M = U·S·Vᵀ` and replaces it by its orthogonal factor `U·Vᵀ` — same directions,
but every direction gets strength 1:

```
B ← μ·B + G                     momentum (μ = 0.95)
M ← G + μ·B                     Nesterov look-ahead
O ← NewtonSchulz5(M) ≈ U·Vᵀ     orthogonalise
W ← W·(1 − lr·wd) − lr · 0.2·√max(rows, cols) · O
```

## How it works
**Newton-Schulz instead of SVD.** An exact SVD per matrix per step is far too slow on a GPU. After
normalising `X = M / ‖M‖_F` (so every singular value is ≤ 1), five iterations of

```
A = X·Xᵀ
X ← a·X + (b·A + c·A²)·X        (a, b, c) = (3.4445, −4.7750, 2.0315)
```

apply the polynomial `σ ← aσ + bσ³ + cσ⁵` to every singular value at once — using only matmuls. The
coefficients are tuned to blow small σ up fast, so after 5 steps they all land in roughly
`[0.7, 1.2]`. Not exactly orthogonal, but empirically that doesn't matter. `newton_schulz()` works
on the wide orientation (transposes tall matrices) so `X·Xᵀ` is the small Gram matrix.

**Which parameters.** Muon is only defined for matrices, and it's known to work worse on the
embedding and the output head (they map token ids in/out, not hidden → hidden). So
`muon_param_groups()` sends hidden 2-D weights to Muon and keeps the (tied) embedding/head and all
1-D norm gains on AdamW — one `Muon` optimiser object runs both branches. Its AdamW branch is
checked against `torch.optim.AdamW` in the tests.

**The scale.** An orthogonal `[m, n]` matrix has RMS `1/√max(m, n)`; multiplying by
`0.2·√max(m, n)` (Moonlight, Liu et al. 2025) gives updates with RMS ≈ 0.2, about what AdamW
produces. That lets Muon reuse AdamW's learning rate, weight decay and warm-up + cosine schedule —
the trainer sets `g["lr"]` on every group exactly as before. (Keller's original uses
`√max(1, rows/cols)` with a ~0.02 learning rate instead.)

## Run it
```bash
python scripts/prepare_data.py --vocab-size 512 --tokenizer-sample 100000
python scripts/train.py --preset nano --steps 400                           # AdamW baseline
python scripts/train.py --preset nano --steps 400 --optimizer muon --lr 6e-3
```
```python
opt = model.configure_optimizer(lr=6e-3, weight_decay=0.1, optimizer="muon")
# or: train(model, ds, TrainConfig(optimizer="muon", lr=6e-3))
```
Default stays `optimizer="adamw"`, so nothing changes unless you ask for it.

## Measured result (Tiny Shakespeare, nano 0.80M params, vocab 512, 400 steps, batch 32, CPU, one seed)
Same schedule (100 warm-up, cosine to 10 %), weight decay 0.1. With Muon, `--lr` applies to both
the Muon matrices and the AdamW-managed embedding/norms.

| optimizer | lr | val loss @200 | val loss @400 | val ppl | time |
|---|---|---|---|---|---|
| AdamW | 1.5e-3 | 3.176 | 2.921 | 18.6 | 104 s |
| AdamW | 3e-3 (default) | 3.106 | 2.850 | 17.3 | 104 s |
| AdamW | 6e-3 | 3.275 | 2.934 | 18.8 | 105 s |
| Muon | 3e-3 | 3.240 | 2.870 | 17.6 | 111 s |
| Muon | 6e-3 | 3.088 | 2.803 | 16.5 | 112 s |
| Muon | 1.2e-2 | 3.038 | **2.778** | **16.1** | 112 s |

What this shows, honestly:
- At the **same** learning rate (3e-3) Muon is slightly *worse* than AdamW (+0.02, within single-seed noise).
- AdamW's best LR in this sweep is the default 3e-3; going to 6e-3 hurts it. **Muon keeps improving
  at 2× and 4× that LR**, and its best run beats AdamW's best by **0.07 val loss** (2.778 vs 2.850).
  So the gain here comes from Muon tolerating — and wanting — a larger step, which matches what
  Moonlight and the speedrun report (~1.3–2× fewer tokens to a target loss at scale).
- At 1.2e-2 the embedding/norm AdamW groups also get the big LR; a separate LR for the two branches
  (as most recipes use) might do better still — see the exercises.
- Newton-Schulz costs ~7 % wall time on CPU for this tiny model; on a GPU, with bf16 matmuls and
  larger batches, it's typically ~1 %.

## Exercises
- Give the AdamW branch its own learning rate (e.g. keep it at 3e-3 while Muon uses 1.2e-2).
- Replace `newton_schulz()` by an exact `U·Vᵀ` from `torch.linalg.svd` and compare loss and speed.
- Plot the singular values of one `wq` gradient, its momentum, and the Newton-Schulz output.
- Try Keller's scaling `√max(1, rows/cols)` with lr ≈ 0.02 and no weight decay.
- Kimi K2's *MuonClip*: after each step, if a head's max `q·k` logit exceeds a threshold `τ`, rescale
  its `wq` and `wk` by `√(τ / max_logit)`. Implement it and compare with QK-norm
  ([qk-norm-softcap.md](qk-norm-softcap.md)).
