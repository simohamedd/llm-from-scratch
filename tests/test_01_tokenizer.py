import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_model, tiny_tok  # noqa: E402,F401
from llm.tokenizer import BPETokenizer  # noqa: E402


def test_tokenizer_roundtrip_and_save(tmp_path):
    t = tiny_tok()
    for s in [TEXT, "unseen wörds ✨", "<|user|>hi<|assistant|>yo<|eos|>"]:
        assert t.decode(t.encode(s)) == s
    assert len(t.encode(TEXT)) < len(TEXT.encode())          # it actually compresses
    t.save(tmp_path / "tok.json")
    t2 = BPETokenizer.load(tmp_path / "tok.json")
    assert t2.encode(TEXT) == t.encode(TEXT) and t2.vocab_size == t.vocab_size
