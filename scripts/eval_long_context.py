"""Evaluate a trained checkpoint on sequences longer than it was trained on, with and without RoPE scaling.

    python scripts/train.py --preset nano --steps 400
    python scripts/eval_long_context.py --ckpt checkpoints/best.pt --lengths 128 256 512

For every length and every scaling kind (none / linear / ntk / yarn) it prints the validation loss
averaged over fixed random windows. The factor is length / training length (minimum 1).
"""
import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.data import TokenDataset  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from llm.trainer import load_checkpoint  # noqa: E402


@torch.no_grad()
def val_loss(model, ids: torch.Tensor, length: int, n_windows: int, batch_size: int, seed: int = 0) -> float:
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(len(ids) - length - 1, (n_windows,), generator=g)
    total = 0.0
    for i in range(0, n_windows, batch_size):
        s = starts[i:i + batch_size]
        x = torch.stack([ids[j:j + length] for j in s])
        y = torch.stack([ids[j + 1:j + 1 + length] for j in s])
        _, loss = model(x, y)
        total += loss.item() * len(s)
    return total / n_windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--data", default="data/corpus.txt")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--kinds", nargs="+", default=["none", "linear", "ntk", "yarn"])
    ap.add_argument("--windows", type=int, default=64, help="validation windows per length")
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()

    model = load_checkpoint(a.ckpt).eval()
    train_len = model.cfg.rope_original_context or model.cfg.context_length
    ds = TokenDataset.from_text_file(a.data, BPETokenizer.load(a.tokenizer), train_len)
    ids = ds.splits["val"]
    print(f"trained at {train_len} tokens; val loss (ppl) per evaluation length")
    print("| kind | " + " | ".join(str(n) for n in a.lengths) + " |")
    print("|---|" + "---|" * len(a.lengths))
    for kind in a.kinds:
        cells = []
        for n in a.lengths:
            factor = max(1.0, n / train_len)
            model.set_rope_scaling("" if kind == "none" else kind, factor, original_len=train_len, max_len=n)
            loss = val_loss(model, ids, n, a.windows, a.batch_size)
            cells.append(f"{loss:.3f} ({math.exp(loss):.1f})")
        print(f"| {kind} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
