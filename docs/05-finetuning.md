# Chapter 5 — Fine-tuning: from base model to assistant

**Code:** [`llm/lora.py`](../llm/lora.py), [`llm/finetune.py`](../llm/finetune.py), [`scripts/finetune.py`](../scripts/finetune.py) · **Branch:** `chapter/05-finetuning`

A pre-trained model only continues text. To make it *answer*, we fine-tune.

## 1. Chat template
```
<|bos|><|user|>What is LoRA?<|assistant|>LoRA fine-tunes ...<|eos|>
```

## 2. SFT — supervised fine-tuning
Train on (prompt, response) pairs. The loss is **masked** on the prompt (`-100`) so the model
only learns to produce answers.

## 3. LoRA — fine-tune 1-10% of the parameters
`W' = W + (α/r)·B·A` with `B` initialised to zero. After training, `merge_lora` folds the
adapter into `W` → no extra inference cost.

## 4. DPO — preference alignment
For each prompt: a `chosen` and a `rejected` answer. DPO raises
`log π(chosen) − log π(rejected)` relative to a frozen reference model — RLHF without a reward
model. Keep it short and gentle on small datasets: it overfits fast.

```bash
python scripts/finetune.py --sft data/sft_example.jsonl --dpo data/dpo_example.jsonl --steps 200
python scripts/chat.py --ckpt checkpoints/sft.pt --chat
```

Reference result: after SFT the nano model answers *"Say hello."* with *"Hello! How can I help you today?"*.

## Next steps (roadmap)
- **GRPO** (DeepSeek-R1-style RL with verifiable rewards)
- QLoRA (LoRA on a 4-bit base model)

**Real-world equivalents:** Hugging Face PEFT & TRL, Unsloth, Axolotl.
