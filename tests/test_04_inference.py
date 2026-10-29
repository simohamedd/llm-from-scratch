import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_model, tiny_tok  # noqa: E402,F401
from llm.generate import stream  # noqa: E402


def test_generation_streams_tokens():
    m = tiny_model(64)
    out = list(stream(m, torch.tensor([[1, 2, 3]]), max_new_tokens=40))  # > context: exercises sliding window
    assert len(out) == 40 and all(0 <= t < 64 for t in out)
