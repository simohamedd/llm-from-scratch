# Advanced — int4 group-wise quantization (GGUF Q4_0 / Q4_1 style)

**Code:** [`llm/quantize.py`](../../llm/quantize.py) (`Int4Linear`, `quantize_int4`, `quantize_groups_int4`, `pack_int4`, `bits_per_weight`) · **Benchmark:** [`scripts/eval_quant.py`](../../scripts/eval_quant.py) · **Test:** `tests/test_22_int4_quantization.py` · **Builds on:** [Chapter 6 — int8 quantization](../06-serving.md)
**Used by:** llama.cpp / Ollama / LM Studio (GGUF `Q4_0`, `Q4_1`, and the `Q4_K` k-quants built on the same idea), GPTQ and AWQ (`group_size=128`), bitsandbytes NF4 (QLoRA, block size 64), MLX 4-bit (group size 64)

## The idea
Chapter 6 stored every Linear weight as int8 with **one scale per row**: ~4× smaller than fp32,
almost no quality loss. Going to 4 bits halves the size again, but 16 levels are far fewer than 256.
With one scale per row, a single large weight in a 4096-wide row stretches the grid for all 4095
others, and most of them get rounded to 0 or ±1 step.

The fix used by every practical 4-bit format: cut each row into **groups** of `group_size`
consecutive input weights (32 in GGUF, 64–128 in GPTQ / AWQ / MLX) and give each group its own
fp16 scale. An outlier now only hurts its own 32 neighbours. The price is the extra scales:
16 bits / 32 weights = 0.5 extra bits per weight.

## How it works
**Q4_0 (symmetric).** For each group, take `m` = the weight with the largest |w|, *keeping its sign*.

```
d = m / -8                      (stored as fp16)
q = clamp(round(w / d), -8, 7)  (stored as q + 8, a 4-bit code 0..15)
w ≈ d · q
```

Dividing by **−8** rather than 7 is the llama.cpp trick: the extreme weight lands exactly on code
−8 and is reconstructed (nearly) exactly, and all 16 codes are used. The cost: a weight close to
−m would need code +8, which doesn't exist, so it is clipped to +7 — off by up to one full step.
The test checks exactly this bound.

**Q4_1 (asymmetric).** Store the group minimum too:

```
d = (max − min) / 15,  q = clamp(round((w − min) / d), 0, 15),  w ≈ d · q + min
```

No clipping, and it wins when a group isn't centred on zero, at 0.5 more bits per weight at g = 32.

**Packing.** Two 4-bit codes go in one `uint8` (`low nibble = even index`, `high = odd`), so a
(out, in) weight becomes a (out, in / 2) byte tensor plus (out, in / group_size) fp16 scales.
If `in` isn't a multiple of `group_size` the row is zero-padded and the pad is dropped on
dequantize. `group_size=0` means one group per row (the int8-style baseline, for comparison).

| format | bits / weight |
|---|---|
| fp32 | 32 |
| int8 per-row (Chapter 6) | 8 + 32 / in_features ≈ 8.1 |
| Q4_0, g = 32 | 4 + 16 / 32 = **4.5** |
| Q4_1, g = 32 | 4 + 32 / 32 = 5.0 |
| Q4_0, g = 128 | 4.125 (+ padding) |

**Forward pass.** Like `Int8Linear`, this is *weight-only* quantization: `forward()` unpacks and
dequantizes the weight to the activation dtype, then runs an ordinary `F.linear`. Real 4-bit
kernels (llama.cpp, Marlin, ExLlama) fuse the unpacking into the matmul so the full-precision
weight never exists in memory. That's where the speed-up comes from; this readable version is
*slower* than fp32 (see below). `Int4Linear.weight` is a read-only property returning the
dequantized weight, so code that reads `.weight` directly (MLA's absorbed decoding path) still works.

`quantize_int4()` skips `lm_head` (often tied to the embedding) and MoE `router`s (their small
logit differences pick the experts) by default. Nothing changes unless you call it.

## Run it
```python
from llm.quantize import quantize_int4, bits_per_weight, model_size_mb
quantize_int4(model, group_size=32, symmetric=True)    # Q4_0; symmetric=False → Q4_1
print(bits_per_weight(model), model_size_mb(model))
```

```bash
python scripts/prepare_data.py --vocab-size 512 --tokenizer-sample 100000
python scripts/train.py --preset micro --steps 400
python scripts/eval_quant.py --ckpt checkpoints/best.pt --group-sizes 16 32 64 128
```

## Measured result
Tiny Shakespeare, vocab 512, 400 training steps, CPU. 128 fixed validation windows of the
model's context length. "weight error" = ‖W − Ŵ‖ / ‖W‖ over the quantized Linears; "top-1 agrees"
= fraction of positions where the quantized model's argmax equals the fp32 model's.
"model MB" includes the fp32 embedding and norms, which aren't quantized.

**micro** (4.6 M params, d_model 256, best val 2.700 during training):

| variant | bits / weight | model MB | weight error | val loss | ppl | top-1 agrees |
|---|---|---|---|---|---|---|
| fp32 | 32 | 18.30 | — | 2.723 | 15.23 | 100 % |
| int8 per-row | 8.11 | 5.09 | 0.007 | 2.723 | 15.22 | 99.5 % |
| Q4_0 g=16 | 5.00 | 3.37 | 0.076 | 2.724 | 15.24 | 94.5 % |
| Q4_1 g=16 | 6.00 | 3.92 | 0.065 | 2.725 | 15.25 | 95.1 % |
| **Q4_0 g=32** | **4.50** | **3.09** | 0.086 | **2.723** | 15.23 | 93.7 % |
| Q4_1 g=32 | 5.00 | 3.37 | 0.078 | 2.728 | 15.30 | 94.4 % |
| Q4_0 g=64 | 4.25 | 2.95 | 0.095 | 2.724 | 15.24 | 93.1 % |
| Q4_1 g=64 | 4.50 | 3.09 | 0.090 | 2.728 | 15.30 | 93.4 % |
| Q4_0 g=128 | 4.22 | 2.93 | 0.103 | 2.726 | 15.28 | 92.6 % |
| Q4_1 g=128 | 4.34 | 3.01 | 0.100 | 2.729 | 15.31 | 92.7 % |
| Q4_0 per-row | 4.05 | 2.84 | 0.114 | 2.725 | 15.25 | 91.9 % |

**nano** (0.8 M params, d_model 128): fp32 2.899 · int8 2.899 · Q4_0 g=32 2.901 (4.50 bits,
3.25 → 0.71 MB) · Q4_1 g=32 2.905 · Q4_0 per-row 2.905 · top-1 agreement 91–95 % for all int4 variants.

**Decoding speed** (greedy, batch 1, 100 tokens, 2 CPU threads): micro fp32 473 tok/s,
int8 298, Q4_0 g=32 127; nano 1006 / 637 / 402. The on-the-fly dequantize costs more than the
smaller weights save, because nothing here is fused.

What this shows, honestly:
- **Q4_0 g=32 makes the Linear weights 7.1× smaller than fp32 and costs nothing measurable in
  validation loss at this scale** (≤ 0.006 nats for every int4 variant). About 6 % of next-token
  argmaxes flip, yet the loss doesn't move, so the flips are between candidates the model rates almost equally.
- Smaller groups do reduce the weight error steadily (0.114 per-row → 0.076 at g=16), and Q4_1 has
  lower weight error than Q4_0 at every group size. **Neither shows up in the loss here.** Q4_1 is
  even slightly *worse* in loss. Lower weight MSE isn't the same as lower output error, and
  these are noise-level differences.
- The reason per-row int4 is fine here: tiny models trained for 400 steps have Gaussian-looking
  weights with no large outliers. Grouping matters for big models, whose activations and weights
  develop outlier channels in the billions-of-parameters range (LLM.int8(), Dettmers et al. 2022).
  `test_smaller_groups_lower_error` plants such outliers and shows the effect directly.
- It's slower on CPU, as explained above. The win is memory: a 7 B model goes from 28 GB (fp32) /
  14 GB (fp16) to ~4 GB at 4.5 bits per weight.

## Exercises
1. Cache the dequantized weight on the first forward (an `Int4Linear(cache=True)` flag) and re-measure
   tokens/s. You keep the disk / download savings but lose the RAM savings. Which one matters for a server?
2. Plant outliers in a trained checkpoint (multiply 1 % of input columns of every `w_up` by 20) and
   rerun `eval_quant.py`. At what group size does per-row int4 fall apart?
3. **Double quantization (QLoRA):** quantize the fp16 scales themselves to int8 with one fp32 scale
   per 256 groups. How many bits per weight does Q4_0 g=64 drop to?
4. **NF4:** replace the 16 uniform levels with the 16 quantiles of a standard normal (bitsandbytes NF4)
   and compare the weight error with Q4_0 at the same group size.
5. **K-quants:** GGUF `Q4_K` groups 8 blocks of 32 into a 256-weight super-block and stores each
   block's scale and min as 6-bit integers relative to one fp16 super-scale. Implement it and compute its bits per weight (4.5).
6. Combine with LoRA: freeze an `Int4Linear` base and train a LoRA adapter on top. That's QLoRA,
   the next item on the roadmap.

## Further reading
- llama.cpp `ggml-quants.c` — `quantize_row_q4_0_ref`, `quantize_row_q4_1_ref`
- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (2022)
- Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (2023)
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023) — NF4, double quantization
