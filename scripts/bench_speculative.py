"""Benchmark MTP self-speculative decoding against plain decoding on a checkpoint trained with MTP.

    python scripts/train.py --preset nano --steps 400 --mtp-depth 2
    python scripts/bench_speculative.py --ckpt checkpoints/best.pt

Draft-model speculative decoding (a separate small model with the same tokenizer drafts):

    python scripts/bench_speculative.py --ckpt micro.pt --draft nano.pt --n-draft 1 2 4
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.generate import stream  # noqa: E402
from llm.speculative import SpecStats, draft_speculative_stream, mtp_speculative_stream  # noqa: E402
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
    ap.add_argument("--draft", default=None, help="checkpoint of a small draft model (same tokenizer)")
    ap.add_argument("--n-draft", type=int, nargs="+", default=[1, 2, 4], help="draft lengths K to try with --draft")
    ap.add_argument("--repeats", type=int, default=3, help="timing repeats per prompt (best of)")
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model, tok = load_checkpoint(a.ckpt).eval(), BPETokenizer.load(a.tokenizer)
    draft = load_checkpoint(a.draft).eval() if a.draft else None
    assert draft is not None or model.mtp is not None, "checkpoint has no MTP heads (train with --mtp-depth >= 1)"
    xs = [torch.tensor([tok.encode(p)]) for p in PROMPTS]
    n = a.max_tokens

    def run_plain(x):
        torch.manual_seed(0)
        return list(stream(model, x, n, temperature=a.temperature, top_k=None, top_p=None, repetition_penalty=1.0))

    def run_spec(x, d, st):
        torch.manual_seed(0)
        if draft is not None:
            return list(draft_speculative_stream(model, draft, x, n, temperature=a.temperature, n_draft=d, stats=st))
        return list(mtp_speculative_stream(model, x, n, temperature=a.temperature, n_draft=d, stats=st))

    rows = []
    base_tok, base_t = 0, 0.0
    for x in xs:
        run_plain(x)                                           # warm-up
        outs = [timed(lambda: run_plain(x)) for _ in range(a.repeats)]
        base_tok += len(outs[0][0])
        base_t += min(t for _, t in outs)
    rows.append(("plain", base_tok / base_t, 1.0, float("nan"), "—", True))
    if draft is not None:   # cost of the draft alone, relative to the target (c in the speed-up formula)
        d_tok, d_t = 0, 0.0
        for x in xs:
            gen = lambda: list(stream(draft, x, n, temperature=0, top_k=None, top_p=None,  # noqa: E731
                                      repetition_penalty=1.0))
            gen()
            outs = [timed(gen) for _ in range(a.repeats)]
            d_tok += len(outs[0][0])
            d_t += min(t for _, t in outs)
        rows.append(("draft alone", d_tok / d_t, float("nan"), float("nan"), "—", "—"))
    for d in (a.n_draft if draft is not None else range(1, len(model.mtp) + 1)):
        st, tokens, secs, same = SpecStats(), 0, 0.0, True
        for x in xs:
            ref = run_plain(x)
            outs = [timed(lambda: run_spec(x, d, SpecStats())) for _ in range(a.repeats)]
            out = run_spec(x, d, st)                           # counted run
            same &= (out == ref)
            tokens += len(out)
            secs += min(t for _, t in outs)
        depth = " / ".join(f"{c / max(1, st.rounds):.0%}" for c in st.accepted_at_depth)
        rows.append((f"draft model K={d}" if draft is not None else f"MTP draft D={d}", tokens / secs, st.tokens_per_call, st.acceptance_rate, depth, same))
    print(f"\n{len(PROMPTS)} prompts × {n} tokens, temperature {a.temperature}, {torch.get_num_threads()} threads")
    print(f"| decoding | tokens/s | speed-up | tokens / main pass | draft acceptance | accepted at depth 1 / 2 / … | "
          f"{'same text as plain' if a.temperature <= 0 else 'same text (n/a when sampling)'} |")
    print("|---|---|---|---|---|---|---|")
    for name, tps, tpc, acc, depth, same in rows:
        acc_s = "—" if acc != acc else f"{acc:.1%}"
        tpc_s = "—" if tpc != tpc else f"{tpc:.2f}"
        print(f"| {name} | {tps:.0f} | {tps / rows[0][1]:.2f}× | {tpc_s} | {acc_s} | {depth} | {same} |")


if __name__ == "__main__":
    main()
