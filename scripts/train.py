"""Pre-train a model from scratch.

    python scripts/train.py --preset nano --steps 1500        # CPU, a few minutes
    python scripts/train.py --preset micro --steps 5000       # GPU / Colab
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.config import PRESETS  # noqa: E402
from llm.data import TokenDataset  # noqa: E402
from llm.model import GPT  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from llm.trainer import TrainConfig, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="nano", choices=list(PRESETS))
    ap.add_argument("--data", default="data/corpus.txt")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--experts", type=int, default=0, help="Mixture of Experts: number of experts (0 = dense)")
    ap.add_argument("--top-k", type=int, default=2, help="experts used per token")
    a = ap.parse_args()

    tok = BPETokenizer.load(a.tokenizer)
    cfg = PRESETS[a.preset]
    cfg.vocab_size = tok.vocab_size
    cfg.n_experts, cfg.moe_top_k = a.experts, a.top_k
    ds = TokenDataset.from_text_file(a.data, tok, cfg.context_length)
    model = GPT(cfg)
    tc = TrainConfig(max_steps=a.steps, batch_size=a.batch_size, lr=a.lr, compile=a.compile,
                     eval_every=max(50, a.steps // 10), warmup_steps=min(100, a.steps // 10))
    model = train(model, ds, tc)
    try:  # sampling is built in chapter 4
        from llm.generate import generate
    except ImportError:
        return
    print("\n--- sample ---")
    print(generate(model, tok, "ROMEO:", max_new_tokens=120))


if __name__ == "__main__":
    main()
