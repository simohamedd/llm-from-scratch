"""Talk to your model in the terminal (streams tokens as they are generated).

    python scripts/chat.py --ckpt checkpoints/best.pt            # completion mode
    python scripts/chat.py --ckpt checkpoints/sft.pt --chat      # chat template (after SFT)
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.finetune import format_chat  # noqa: E402
from llm.generate import stream  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from llm.trainer import load_checkpoint, pick_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=200)
    a = ap.parse_args()
    device = pick_device()
    model, tok = load_checkpoint(a.ckpt, device), BPETokenizer.load(a.tokenizer)
    eos = tok.token_id("<|eos|>")
    print("Type a prompt (Ctrl-C to quit).")
    while True:
        try:
            prompt = input("\n> ")
        except (KeyboardInterrupt, EOFError):
            break
        text = format_chat([{"role": "user", "content": prompt}], True) if a.chat else prompt
        x = torch.tensor([tok.encode(text)], dtype=torch.long, device=device)
        for t in stream(model, x, a.max_tokens, a.temperature, stop_ids={eos}):
            print(tok.decode([t]), end="", flush=True)
        print()


if __name__ == "__main__":
    main()
