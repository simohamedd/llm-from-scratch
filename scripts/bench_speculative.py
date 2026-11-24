"""Benchmark MTP self-speculative decoding against plain decoding on a checkpoint trained with MTP.

    python scripts/train.py --preset nano --steps 400 --mtp-depth 2
    python scripts/bench_speculative.py --ckpt checkpoints/best.pt
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.generate import stream  # noqa: E402
from llm.speculative import SpecStats, mtp_speculative_stream  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from llm.trainer import load_checkpoint  # noqa: E402

PROMPTS = ["ROMEO:", "JULIET:\nO", "KING HENRY:\nWhat", "First Citizen:\nWe are", "MENENIUS:\nI tell you",
           "LADY MACBETH:", "HAMLET:\nTo be", "Second Servant:"]


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (output identical to plain)")
    ap.add_argument("--repeats", type=int, default=3, help="timing repeats per prompt (best of)")
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model, tok = load_checkpoint(a.ckpt).eval(), BPETokenizer.load(a.tokenizer)
    assert model.mtp is not None, "checkpoint has no MTP heads (train with --mtp-depth >= 1)"
    xs = [torch.tensor([tok.encode(p)]) for p in PROMPTS]
    n = a.max_tokens

    def run_plain(x):
        torch.manual_seed(0)
        return list(stream(model, x, n, temperature=a.temperature, top_k=None, top_p=None, repetition_penalty=1.0))

    def run_spec(x, d, st):
        torch.manual_seed(0)
        return list(mtp_speculative_stream(model, x, n, temperature=a.temperature, n_draft=d, stats=st))

    rows = []
    base_tok, base_t = 0, 0.0
    for x in xs:
        run_plain(x)                                           # warm-up
        outs = [timed(lambda: run_plain(x)) for _ in range(a.repeats)]
        base_tok += len(outs[0][0])
        base_t += min(t for _, t in outs)
    rows.append(("plain", base_tok / base_t, 1.0, float("nan"), "—", True))
    for d in range(1, len(model.mtp) + 1):
        st, tokens, secs, same = SpecStats(), 0, 0.0, True
        for x in xs:
            ref = run_plain(x)
            outs = [timed(lambda: run_spec(x, d, SpecStats())) for _ in range(a.repeats)]
            out = run_spec(x, d, st)                           # counted run
            same &= (out == ref)
            tokens += len(out)
            secs += min(t for _, t in outs)
        depth = " / ".join(f"{c / max(1, st.rounds):.0%}" for c in st.accepted_at_depth)
        rows.append((f"MTP draft D={d}", tokens / secs, st.tokens_per_call, st.acceptance_rate, depth, same))
    print(f"\n{len(PROMPTS)} prompts × {n} tokens, temperature {a.temperature}, {torch.get_num_threads()} threads")
    print(f"| decoding | tokens/s | speed-up | tokens / main pass | draft acceptance | accepted at depth 1 / 2 / … | "
          f"{'same text as plain' if a.temperature <= 0 else 'same text (n/a when sampling)'} |")
    print("|---|---|---|---|---|---|---|")
    for name, tps, tpc, acc, depth, same in rows:
        acc_s = "—" if acc != acc else f"{acc:.1%}"
        print(f"| {name} | {tps:.0f} | {tps / rows[0][1]:.2f}× | {tpc:.2f} | {acc_s} | {depth} | {same} |")


if __name__ == "__main__":
    main()
