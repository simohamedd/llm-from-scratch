# Advanced — Warmup-Stable-Decay (WSD) learning-rate schedule

**Code:** [`llm/schedules.py`](../../llm/schedules.py), selected by `TrainConfig(schedule=...)` in [`llm/trainer.py`](../../llm/trainer.py) · **Test:** `tests/test_15_wsd_schedule.py` · **Builds on:** [Chapter 3](../03-pretraining.md)
**Used by:** MiniCPM (where the name comes from) · SmolLM2 · Kimi K2 · DeepSeek LLM (multi-step variant) and DeepSeek-V3 (constant, then decay)

## The idea
The course's default schedule is warm-up + cosine decay: the LR starts falling right after warm-up
and reaches its minimum exactly at `max_steps`. That couples the whole run to a step count you
must pick up front. Stop early and the model never got its low-LR "cool-down"; want to train
longer and the schedule is already spent. Comparing budgets (1B vs 2B vs 4B tokens) means a full
separate run for each.

**Warmup-Stable-Decay** (Hu et al. 2024, *MiniCPM*; Hägele et al. 2024, *Scaling Laws and
Compute-Optimal Training Beyond Fixed Training Durations*) splits the schedule in three:

```
lr ▲    ┌──────────────── stable (constant peak LR) ────────┐
   │   /                                                    \   decay
   │  / warm-up                                              \  (last 10-20 %)
   └─┴──────────────────────────────────────────────────────┴─▶ step
```

The stable phase doesn't know when training ends. You can save a checkpoint anywhere on the
plateau, branch off, run only the short decay, and get a finished model for that budget — while the
main run keeps going. One long run gives you a whole family of models (and a scaling-law curve) for
the price of the extra decay branches.

## How it works
```
step < warmup              lr = peak · (step + 1) / warmup
step ≤ max_steps − D       lr = peak                               (stable)
otherwise                  lr = min + (peak − min) · f(p),  p = (step − (max_steps − D)) / D
```
with `D = round(decay_frac · max_steps)` (clamped so it never overlaps warm-up) and three decay shapes:

| `decay_shape` | f(p) | notes |
|---|---|---|
| `linear` | 1 − p | MiniCPM-style |
| `cosine` | ½(1 + cos πp) | a cosine squeezed into the window |
| `sqrt` (default) | 1 − √p | "1-sqrt": drops fast then flattens — best in Hägele et al. |

Most of the loss improvement of a WSD run happens *during* the decay: on the plateau the high LR
keeps the weights bouncing around a wide valley; annealing lets them settle into it. So a WSD loss
curve looks worse than cosine for most of training and then overtakes it at the end — see below.

`llm/schedules.py` also holds the original cosine formula as `cosine_lr()`; the trainer's `lr_at()`
just dispatches on `tc.schedule`. The default is still `"cosine"`, and a test checks it is
bit-for-bit the old formula.

## Run it
```bash
python scripts/prepare_data.py --vocab-size 512 --tokenizer-sample 100000
python scripts/train.py --preset nano --steps 400                                    # cosine baseline
python scripts/train.py --preset nano --steps 400 --schedule wsd                     # 1-sqrt, last 20 %
python scripts/train.py --preset nano --steps 400 --schedule wsd --decay-shape linear --decay-frac 0.1
```
```python
train(model, ds, TrainConfig(schedule="wsd", decay_frac=0.2, decay_shape="sqrt", min_lr_ratio=0.1))
```

## Measured result (Tiny Shakespeare, nano 0.80M params, vocab 512, 400 steps, batch 32, AdamW lr 3e-3, CPU)
40 warm-up steps (what `scripts/train.py` uses for 400 steps), weight decay 0.1. Val loss, 20 eval batches.
Seed 1337 unless noted; "min" is the final LR as a fraction of the peak.

| schedule | decay | min | val @200 | val @300 | val @350 | val @400 |
|---|---|---|---|---|---|---|
| cosine (default) | — | 0.1 | **3.120** | **2.913** | **2.874** | 2.852 |
| cosine | — | 0.0 | 3.115 | 2.909 | 2.882 | 2.881 |
| WSD 1-sqrt | last 20 % | 0.1 | 3.151 | 2.985 | 2.885 | **2.804** |
| WSD 1-sqrt | last 20 % | 0.0 | 3.151 | 2.985 | 2.880 | 2.813 |
| WSD linear | last 20 % | 0.1 | 3.151 | 2.985 | 2.922 | 2.808 |
| WSD 1-sqrt | last 10 % | 0.1 | 3.151 | 2.985 | 2.963 | 2.817 |
| cosine, seed 7 | — | 0.1 | 3.118 | 2.915 | 2.902 | 2.832 |
| WSD 1-sqrt, seed 7 | last 20 % | 0.1 | 3.122 | 2.977 | 2.905 | **2.790** |

Wall time is identical (~195 s per run, 1 thread) — the schedule is free.

What this shows, honestly:
- **WSD ends lower**: −0.047 val loss on seed 1337 and −0.042 on seed 7 versus the default cosine.
  Two seeds agreeing in sign and size makes it more than noise, but it's still a tiny model and a
  400-step run; at scale the literature reports WSD *matching* a tuned cosine rather than beating it.
  Part of the gain here is likely that cosine already spends half the run below ~55 % of the peak
  LR, which is too cautious for such a short run.
- **Until the decay starts, WSD is behind** (+0.07 at step 300) — exactly the expected curve shape.
  Don't judge a WSD run from its plateau loss.
- The three WSD variants all land within 0.013 of each other; 1-sqrt is the best and a 20 % window
  beats 10 %, matching Hägele et al.'s ranking, but the differences are within noise here.
- Decaying to exactly 0 doesn't help either schedule at this scale.

## Exercises
- **Branch from the plateau.** Save model + optimizer state at step 200 of a WSD run, then run a
  40-step decay from there. Compare with a full 240-step cosine run: you get a "200-step model"
  almost for free. (You'll need to add optimizer state to `save_checkpoint()`.)
- Plot `lr_at()` for all schedules side by side, and the val-loss curves from the table.
- Implement DeepSeek LLM's multi-step variant: hold the peak, drop to 31.6 % of it at 80 % of training and to 10 % at 90 %.
- Increase the peak LR: does WSD tolerate a larger plateau LR than cosine tolerates a peak?
- Try WSD together with Muon ([muon.md](muon.md)) — Moonlight and Kimi K2 use exactly that pair.
