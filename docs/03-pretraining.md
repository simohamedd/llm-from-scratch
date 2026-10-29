# Chapter 3 — Pre-training

**Code:** [`llm/data.py`](../llm/data.py), [`llm/trainer.py`](../llm/trainer.py), [`scripts/train.py`](../scripts/train.py) · **Branch:** `chapter/03-pretraining`

Pre-training = next-token prediction on lots of text. Loss is cross-entropy; perplexity = `exp(loss)`.

## The recipe
- **Data:** one long stream of token ids; sample random windows `x`, targets `y = x shifted by 1`.
- **Optimizer:** AdamW (β = 0.9, 0.95), weight decay 0.1 on matrices only.
- **Schedule:** linear warm-up then cosine decay to 10% of the peak LR.
- **Stability:** gradient clipping at 1.0, scaled init for residual projections.
- **Efficiency:** bf16 autocast on GPU, `--compile` for `torch.compile`, gradient accumulation for big effective batches.

## Run it
```bash
python scripts/prepare_data.py                 # downloads Tiny Shakespeare, trains tokenizer
python scripts/train.py --preset nano --steps 600
```
Reference CPU run (nano, 600 steps, ~2.5 min): validation perplexity **512 → 15**.

## Scaling up
- Use a bigger corpus (FineWeb-Edu, Wikipedia dumps, your own domain text) with `--file`.
- Chinchilla rule of thumb: ~20 training tokens per parameter.
- Multi-GPU: wrap the model in `torch.nn.parallel.DistributedDataParallel` or FSDP (see roadmap).

**Real-world equivalents:** nanoGPT / nanochat, torchtitan, Megatron-LM, DeepSpeed.
