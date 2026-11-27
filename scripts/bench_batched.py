"""Throughput of batched generation (llm/batching.py) versus one prompt at a time.

    python scripts/train.py --preset nano --steps 400
    python scripts/bench_batched.py --ckpt checkpoints/best.pt --batch-sizes 1 2 4 8 16 32

Greedy by default, so every batch size must reproduce the one-at-a-time text exactly
(the "same" column) — any mismatch means a padding / masking bug.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.batching import batch_generate_ids  # noqa: E402
from llm.generate import stream  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from llm.trainer import load_checkpoint  # noqa: E402

PROMPTS = ["ROMEO:", "JULIET:\nO", "KING HENRY:\nWhat", "First Citizen:\nWe are", "MENENIUS:\nI tell you",
           "LADY MACBETH:", "HAMLET:\nTo be", "Second Servant:", "GLOUCESTER:\nNow is the winter",
           "Nurse:\nFaith,", "BRUTUS:", "QUEEN MARGARET:\nAh, but", "PETRUCHIO:\nGood morrow", "Clown:",
           "DUKE VINCENTIO:\nWhat news", "ISABELLA:\nI am"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    ap.add_argument("--n-prompts", type=int, default=32, help="total prompts (the list above, repeated)")
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    a = ap.parse_args()
    model, tok = load_checkpoint(a.ckpt).eval(), BPETokenizer.load(a.tokenizer)
    prompts = [tok.encode(PROMPTS[i % len(PROMPTS)]) for i in range(a.n_prompts)]
    kw = dict(temperature=0, top_k=None, top_p=None, repetition_penalty=1.0)
    pad = tok.token_id("<|pad|>")

    def sequential():
        return [list(stream(model, torch.tensor([p]), a.max_tokens, **kw)) for p in prompts]

    def batched(bs):
        out = []
        for i in range(0, len(prompts), bs):
            out += batch_generate_ids(model, prompts[i:i + bs], max_new_tokens=a.max_tokens, pad_id=pad, **kw)
        return out

    sequential()                                            # warm-up
    t = time.perf_counter()
    ref = sequential()
    base = sum(map(len, ref)) / (time.perf_counter() - t)
    print(f"{'mode':<16}{'tokens/s':>10}{'speed-up':>10}  same")
    print(f"{'stream() x N':<16}{base:>10.0f}{1.0:>9.2f}×  —")
    for bs in a.batch_sizes:
        batched(bs)                                         # warm-up
        t = time.perf_counter()
        out = batched(bs)
        tps = sum(map(len, out)) / (time.perf_counter() - t)
        same = sum(o == r for o, r in zip(out, ref))
        print(f"{f'batch {bs}':<16}{tps:>10.0f}{tps / base:>9.2f}×  {same}/{len(ref)}")


if __name__ == "__main__":
    main()
