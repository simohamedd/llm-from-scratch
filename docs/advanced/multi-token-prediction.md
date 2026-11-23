# Advanced — Multi-Token Prediction (MTP) auxiliary heads

**Code:** [`llm/mtp.py`](../../llm/mtp.py), enabled by `ModelConfig(mtp_depth=...)` in [`llm/model.py`](../../llm/model.py) · **Test:** `tests/test_18_mtp.py` · **Builds on:** [Chapter 2](../02-transformer.md), [Chapter 3](../03-pretraining.md)
**Used by:** DeepSeek-V3 / R1 (D = 1, also reused for speculative decoding) · MiMo-7B · GLM-4.5 · Qwen3-Next · Meta's multi-token models (Gloeckle et al. 2024, parallel heads)

## The idea
A plain language model is trained on one question per position: *what is the next token?* Every
position gives one bit of supervision about one step of the future. **Multi-token prediction**
asks a few more questions at the same position — what comes 2, 3, … D + 1 tokens ahead — and adds
those losses to the main one. Two motivations:

- **Denser signal.** Each training token contributes D + 1 losses instead of 1, and the trunk's
  hidden state is pushed to encode a short "plan", not only the immediate next token.
- **Free draft model.** At inference the extra heads can propose the next few tokens; the main
  model verifies them in one forward pass (speculative decoding). DeepSeek-V3 reports 85–90 %
  acceptance of the second token, ≈ 1.8× decoding throughput.

Gloeckle et al. (2024, *Better & Faster LLMs via Multi-token Prediction*) used D independent output
heads on a shared trunk. **DeepSeek-V3** (2024, §2.2) instead chains them *sequentially*, so each
depth still conditions on everything before the token it predicts. That is what this repo builds.

## How it works
Depth k (k = 1..D) at position i:

```
h⁰ = trunk output (before the final norm)                     ← main model, sees t₀..tᵢ
h'ᵢ = M_k · [ RMSNorm(h^{k−1}ᵢ) ; RMSNorm(Emb(t_{i+k})) ]       Linear 2d → d
h^k = TRM_k(h')                                                 one ordinary Transformer block (causal)
p^k_{i} = softmax(Head(RMSNorm(h^kᵢ)))     → predicts t_{i+k+1}

L = L_main + λ · (1/D) Σ_k CE(p^k, t_{·+k+1})
```

```
tokens   t0   t1   t2   t3   t4   t5
trunk    h0   h1   h2   h3   h4         → main head predicts t1..t5
depth 1  [h0,e(t1)] [h1,e(t2)] [h2,e(t3)] [h3,e(t4)]   → predicts t2..t5
depth 2  [h¹0,e(t2)] [h¹1,e(t3)] [h¹2,e(t4)]           → predicts t3..t5
```

- **Shared** with the main model: the token embedding and the output head (and the tied weight
  between them). Each depth owns only `proj` (2d × d), one `Block` and three RMSNorms
  (`hnorm`, `enorm`, `norm`). For nano that is +218 k parameters per depth.
- Depth k runs on length T − k: the last k positions have no target.
- The trunk receives gradient from every depth (through `h⁰`), which is the point.
- **Training only.** The MTP loss is added only when `model.training`; `eval()` / generation run the
  main model alone, so validation loss and perplexity stay directly comparable with a non-MTP model
  and `generate()` is unchanged. MoE aux losses of MTP blocks are included in training.
- The heads are built *after* the trunk is initialised, so the same seed gives a bit-identical
  trunk with or without MTP (tested) — that makes the comparison below controlled.

A subtle point about what the depths actually learn: depth k at position i is given `Emb(t_{i+k})`,
so it predicts `t_{i+k+1}` from tokens ≤ i + k — it is still a *next-token* problem, but the newest
k tokens must be absorbed by just k small blocks instead of the full trunk. That is why its loss is
close to (not far above) the main loss, and why it is a useful speculative draft. The "lookahead"
pressure lands on the trunk: `h⁰ᵢ` must already carry what the cheap modules need about the future.

## Run it
```bash
python scripts/prepare_data.py --vocab-size 512 --tokenizer-sample 100000
python scripts/train.py --preset nano --steps 400                                 # baseline
python scripts/train.py --preset nano --steps 400 --mtp-depth 1 --mtp-weight 0.3  # DeepSeek-V3 setting
```
```python
model = GPT(ModelConfig(..., mtp_depth=1, mtp_weight=0.3))
model.num_params(include_mtp=False)   # what you keep for inference
model.last_mtp_losses                 # per-depth losses of the last training step (also in the train log)
```
The training log shows ` | mtp 2.61` (one number per depth: that depth's loss on the last training
batch, in train mode — noisy, and not the same estimator as the eval-mode `train`/`val` columns).

## Measured result (Tiny Shakespeare, nano 0.80M trunk params, vocab 512, 400 steps, batch 32, AdamW 3e-3, cosine, CPU 2 threads)
Val loss of the **main** next-token head (eval mode, 20 batches). Same seed ⇒ identical trunk init and batches.

| run | seed | val @200 | val @300 | val @400 | Δ vs baseline @400 | wall |
|---|---|---|---|---|---|---|
| baseline | 1337 | 3.121 | 2.914 | 2.854 | — | 116 s |
| MTP D=1, λ=0.3 | 1337 | 3.095 | 2.885 | **2.823** | −0.031 | 150 s |
| MTP D=2, λ=0.3 | 1337 | 3.090 | 2.903 | 2.834 | −0.020 | 182 s |
| MTP D=1, λ=0.1 | 1337 | 3.089 | 2.896 | 2.840 | −0.014 | 150 s |
| baseline | 7 | 3.118 | 2.915 | **2.832** | — | 117 s |
| MTP D=1, λ=0.3 | 7 | 3.135 | 2.921 | 2.847 | +0.016 | 149 s |
| baseline | 42 | 3.122 | 2.920 | **2.835** | — | 115 s |
| MTP D=1, λ=0.3 | 42 | 3.123 | 2.936 | 2.844 | +0.010 | 151 s |

Mean over the three seeds (D = 1, λ = 0.3): baseline **2.840**, MTP **2.838**.

What this shows, honestly:
- **No measurable gain at this scale.** One seed improves by 0.03, two get slightly worse; the
  average difference (0.002) is far below the seed-to-seed spread (~0.02). This matches the
  literature: Gloeckle et al. found MTP *hurts* small models and only pays off from ~1–3B
  parameters; DeepSeek-V3's gains are reported at 16B–671B MoE scale.
- **It costs ~30 % wall time per depth** at nano size: one extra Transformer block (the trunk has
  4) plus an extra vocabulary projection per depth. At large scale the relative cost is small
  (DeepSeek-V3: 1 extra block on 61).
- The MTP heads themselves learn well: the depth-1 training loss tracks the main loss closely
  (≈2.6 vs ≈2.5 at step 400), which is what makes them usable as a speculative-decoding draft.

## Exercises
1. **Speculative decoding with the MTP head.** Keep the trunk's hidden states for the whole prefix,
   run depth 1 on `[h⁰; Emb(next token)]` to draft one extra token, then verify both with one main
   forward. Measure the acceptance rate on Tiny Shakespeare (DeepSeek-V3: 85–90 %).
2. Implement Gloeckle-style *parallel* heads (no chaining, each head reads only `h⁰`) and compare
   the head losses with the sequential design at depth 2–3.
3. Schedule λ: 0.3 for the first 70 % of steps, then 0.1 (DeepSeek-V3's recipe). Does it help the
   final main loss?
4. Repeat the table with the `micro` preset on a GPU for 3–5k steps. Does the sign of Δ become
   consistent as the model grows?
5. Measure the code-specific claim: on a code corpus (e.g. this repo's `.py` files) MTP is reported
   to help more than on prose. Is that visible at nano scale?

## References
- DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024), §2.2 Multi-Token Prediction.
- F. Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction* (ICML 2024).
- Xiaomi, *MiMo: Unlocking the Reasoning Potential of Language Model* (2025) — MTP layers for decoding.
