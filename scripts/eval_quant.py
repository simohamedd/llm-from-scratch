"""Compare fp32, int8 and group-wise int4 (Q4_0 / Q4_1) versions of a trained checkpoint.

    python scripts/train.py --preset nano --steps 400
    python scripts/eval_quant.py --ckpt checkpoints/best.pt --group-sizes 32 128

For each variant it prints the validation loss on fixed random windows, the size of the
quantized Linear weights (bits per weight) and of the whole model, and how often greedy
next-token predictions agree with the fp32 model.
"""
import argparse
import copy
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.data import TokenDataset  # noqa: E402
from llm.quantize import bits_per_weight, model_size_mb, quantize_int4, quantize_int8  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from llm.trainer import load_checkpoint  # noqa: E402


@torch.no_grad()
def evaluate(model, x, y, ref_pred=None, batch_size=16):
    total, preds = 0.0, []
    for i in range(0, len(x), batch_size):
        logits, loss = model(x[i:i + batch_size], y[i:i + batch_size])
        total += loss.item() * len(x[i:i + batch_size])
        preds.append(logits.argmax(-1))
    pred = torch.cat(preds)
    agree = 1.0 if ref_pred is None else (pred == ref_pred).float().mean().item()
    return total / len(x), pred, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--data", default="data/corpus.txt")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    ap.add_argument("--group-sizes", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--windows", type=int, default=128)
    a = ap.parse_args()

    base = load_checkpoint(a.ckpt).eval()
    n = base.cfg.context_length
    ids = TokenDataset.from_text_file(a.data, BPETokenizer.load(a.tokenizer), n).splits["val"]
    g = torch.Generator().manual_seed(0)
    starts = torch.randint(len(ids) - n - 1, (a.windows,), generator=g)
    x = torch.stack([ids[s:s + n] for s in starts])
    y = torch.stack([ids[s + 1:s + 1 + n] for s in starts])

    variants = [("fp32", lambda m: m), ("int8 per-row", quantize_int8)]
    for gs in a.group_sizes:
        variants.append((f"int4 Q4_0 g={gs}", lambda m, gs=gs: quantize_int4(m, gs, symmetric=True)))
        variants.append((f"int4 Q4_1 g={gs}", lambda m, gs=gs: quantize_int4(m, gs, symmetric=False)))
    variants.append(("int4 Q4_0 per-row", lambda m: quantize_int4(m, 0, symmetric=True)))

    print("| variant | bits / weight (Linear) | model MB | weight rel. RMS error | val loss | ppl | top-1 agrees with fp32 |")
    print("|---|---|---|---|---|---|---|")
    ref = None
    for name, fn in variants:
        m = fn(copy.deepcopy(base)).eval()
        loss, pred, agree = evaluate(m, x, y, ref)
        ref = pred if ref is None else ref
        bpw = 32.0 if name == "fp32" else bits_per_weight(m)
        err = weight_error(base, m)
        print(f"| {name} | {bpw:.2f} | {model_size_mb(m):.2f} | {err:.4f} | {loss:.3f} | {math.exp(loss):.2f} "
              f"| {agree:.1%} |")


def weight_error(base, quant) -> float:
    """Relative RMS error ||W - W_hat|| / ||W|| over every Linear that was quantized."""
    num = den = 0.0
    qmods = dict(quant.named_modules())
    for name, mod in base.named_modules():
        q = qmods.get(name)
        if isinstance(mod, torch.nn.Linear) and q is not None and not isinstance(q, torch.nn.Linear):
            w = mod.weight.float()
            w_hat = q.weight if hasattr(q, "dequantize") else q.w_int8.float() * q.scale
            num += (w - w_hat).pow(2).sum().item()
            den += w.pow(2).sum().item()
    return math.sqrt(num / den) if den else 0.0


if __name__ == "__main__":
    main()
