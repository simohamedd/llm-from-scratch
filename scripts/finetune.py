"""Turn the pre-trained base model into a (tiny) assistant: LoRA-SFT, then DPO.

    python scripts/finetune.py --ckpt checkpoints/best.pt --sft data/sft_example.jsonl \
                               --dpo data/dpo_example.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.finetune import dpo, format_chat, sft  # noqa: E402
from llm.generate import generate  # noqa: E402
from llm.lora import apply_lora, merge_lora, trainable_summary  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from llm.trainer import TrainConfig, load_checkpoint, pick_device, save_checkpoint  # noqa: E402


def read_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    ap.add_argument("--sft", default="data/sft_example.jsonl")
    ap.add_argument("--dpo", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/sft.pt")
    a = ap.parse_args()
    device = pick_device()
    tok, model = BPETokenizer.load(a.tokenizer), load_checkpoint(a.ckpt, device)

    apply_lora(model, targets=("wq", "wk", "wv", "wo"), r=a.lora_rank)
    print("LoRA:", trainable_summary(model))
    sft(model, tok, read_jsonl(a.sft), steps=a.steps, device=device)
    if a.dpo:
        # DPO overfits fast on small preference sets: keep it short and gentle
        dpo(model, tok, read_jsonl(a.dpo), steps=30, device=device)
    merge_lora(model)
    save_checkpoint(Path(a.out), model, model.cfg, TrainConfig(), a.steps, float("nan"))
    print(f"saved {a.out}")
    prompt = format_chat([{"role": "user", "content": "Who are you?"}], add_generation_prompt=True)
    print("sample:", generate(model, tok, prompt, max_new_tokens=60, temperature=0.3))


if __name__ == "__main__":
    main()
