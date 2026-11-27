import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_model, tiny_tok  # noqa: E402
from llm.data import TokenDataset  # noqa: E402
from llm.schedules import cosine_lr, decay_factor, wsd_lr  # noqa: E402
from llm.trainer import TrainConfig, lr_at, train  # noqa: E402


def _old_lr_at(step, tc):
    """The trainer's original warm-up + cosine formula, kept verbatim as a regression reference."""
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    progress = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
    min_lr = tc.lr * tc.min_lr_ratio
    return min_lr + 0.5 * (tc.lr - min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def test_default_schedule_is_unchanged_cosine():
    tc = TrainConfig(max_steps=300, warmup_steps=30, lr=3e-3)
    assert tc.schedule == "cosine"
    for step in range(0, 320, 7):
        assert lr_at(step, tc) == pytest.approx(_old_lr_at(step, tc), rel=1e-12)
    assert cosine_lr(300, 1.0, 30, 300, 0.1) == pytest.approx(0.1)


@pytest.mark.parametrize("shape", ["linear", "cosine", "sqrt"])
def test_wsd_has_warmup_flat_plateau_and_decays_to_min(shape):
    lr, warm, total, frac, mn = 1e-2, 10, 100, 0.2, 0.1
    lrs = [wsd_lr(s, lr, warm, total, frac, mn, shape) for s in range(total + 1)]
    # warm-up is the same linear ramp as the cosine schedule
    assert lrs[:warm] == pytest.approx([cosine_lr(s, lr, warm, total) for s in range(warm)])
    # stable: exactly the peak LR from the end of warm-up until the decay window (steps 10..80)
    assert all(v == lr for v in lrs[warm:81])
    # decay: monotone non-increasing, ends exactly at min_lr
    tail = lrs[80:]
    assert all(a >= b for a, b in zip(tail, tail[1:]))
    assert lrs[-1] == pytest.approx(lr * mn)
    assert lrs[90] < lr and lrs[90] > lr * mn


def test_decay_shapes_and_1_sqrt_drops_fastest_early():
    for shape in ("linear", "cosine", "sqrt"):
        assert decay_factor(0.0, shape) == 1.0 and decay_factor(1.0, shape) == pytest.approx(0.0)
    assert decay_factor(0.5, "linear") == pytest.approx(0.5)
    assert decay_factor(0.5, "cosine") == pytest.approx(0.5)
    assert decay_factor(0.25, "sqrt") == pytest.approx(0.5)
    assert decay_factor(0.1, "sqrt") < decay_factor(0.1, "linear") < decay_factor(0.1, "cosine")
    with pytest.raises(ValueError):
        decay_factor(0.5, "exp")


def test_wsd_edge_cases():
    # decay window can't overlap warm-up; decay_frac=0 = constant LR after warm-up
    assert wsd_lr(10, 1.0, 10, 20, decay_frac=0.9) == pytest.approx(1.0)
    assert wsd_lr(20, 1.0, 10, 20, decay_frac=0.9, min_lr_ratio=0.0) == pytest.approx(0.0)
    assert all(wsd_lr(s, 1.0, 5, 50, decay_frac=0.0) == 1.0 for s in range(5, 51))
    with pytest.raises(ValueError):
        wsd_lr(0, 1.0, 5, 50, decay_frac=1.5)
    with pytest.raises(ValueError):
        lr_at(0, TrainConfig(schedule="step"))


def test_training_with_wsd_reduces_loss_and_uses_the_schedule(tmp_path):
    t = tiny_tok()
    m = tiny_model(t.vocab_size)
    ds = TokenDataset(torch.tensor(t.encode(TEXT * 3)), context_length=32)
    x, y = ds.batch("val", 8)
    before = m(x, y)[1].item()
    tc = TrainConfig(max_steps=60, batch_size=8, eval_every=30, eval_iters=1, warmup_steps=5,
                     schedule="wsd", decay_frac=0.25, min_lr_ratio=0.0, out_dir=str(tmp_path))
    logs = []
    train(m, ds, tc, device="cpu", log=logs.append)
    assert m(x, y)[1].item() < before * 0.7
    assert "schedule=wsd" in logs[0]
    assert "lr 3.00e-03" in logs[2]     # step 30: still on the plateau (cosine would be ~1.9e-3 here)
    assert "lr 0.00e+00" in logs[3]     # final step: decayed to min_lr
