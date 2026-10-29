"""Download a training corpus and train the BPE tokenizer on it.

    python scripts/prepare_data.py                 # Tiny Shakespeare (1 MB) — default
    python scripts/prepare_data.py --file my.txt   # your own text
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.tokenizer import BPETokenizer  # noqa: E402

TINY_SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="use a local text file instead of downloading")
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--tokenizer-sample", type=int, default=300_000, help="chars used to train BPE")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    corpus = out / "corpus.txt"
    if a.file:
        corpus.write_text(Path(a.file).read_text(encoding="utf-8"), encoding="utf-8")
    elif not corpus.exists():
        print(f"downloading {TINY_SHAKESPEARE}")
        urllib.request.urlretrieve(TINY_SHAKESPEARE, corpus)
    text = corpus.read_text(encoding="utf-8")
    print(f"corpus: {len(text):,} characters")

    tok = BPETokenizer()
    tok.train(text[: a.tokenizer_sample], vocab_size=a.vocab_size, verbose=True)
    Path("checkpoints").mkdir(exist_ok=True)
    tok.save("checkpoints/tokenizer.json")
    sample = text[:200]
    ids = tok.encode(sample)
    print(f"vocab={tok.vocab_size}  compression={len(sample.encode()) / len(ids):.2f} bytes/token")
    assert tok.decode(ids) == sample, "round-trip failed"
    print("saved checkpoints/tokenizer.json")


if __name__ == "__main__":
    main()
