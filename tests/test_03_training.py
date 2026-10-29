import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_model, tiny_tok  # noqa: E402,F401
from llm.data import TokenDataset  # noqa: E402
from llm.trainer import TrainConfig, train  # noqa: E402


def test_training_reduces_loss(tmp_path):
    t = tiny_tok()
    m = tiny_model(t.vocab_size)
    ds = TokenDataset(torch.tensor(t.encode(TEXT * 3)), context_length=32)
    x, y = ds.batch("val", 8)
    before = m(x, y)[1].item()
    train(m, ds, TrainConfig(max_steps=60, batch_size=8, eval_every=1000, eval_iters=1,
                             warmup_steps=5, out_dir=str(tmp_path)), device="cpu", log=lambda *_: None)
    assert m(x, y)[1].item() < before * 0.7
    assert (tmp_path / "last.pt").exists()
