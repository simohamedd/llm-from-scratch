import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.config import ModelConfig  # noqa: E402
from llm.generate import stream  # noqa: E402
from llm.model import GPT, rope_frequencies  # noqa: E402
from llm.rope_scaling import (base_inv_freq, rope_tables, scaled_inv_freq,  # noqa: E402
                              yarn_correction_range, yarn_mscale)


def small(**kw):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                           context_length=16, **kw))


def test_off_is_plain_rope_and_default_model_unchanged():
    cos, sin = rope_frequencies(8, 32, 10000.0)
    for kind, factor in [("", 1.0), ("", 4.0), ("yarn", 1.0)]:
        c, s = rope_tables(8, 32, 10000.0, kind, factor, original_len=16)
        assert torch.allclose(c, cos) and torch.allclose(s, sin)
    m = small()
    assert m.cfg.max_seq_len == 16 and m.rope_cos.shape[0] == 32
    assert torch.allclose(m.rope_cos, rope_frequencies(8, 32, 10000.0)[0])


def test_linear_and_ntk_frequencies():
    d, base, s = 32, 10000.0, 4.0
    inv = base_inv_freq(d, base)
    lin, m = scaled_inv_freq(d, base, "linear", s)
    assert torch.allclose(lin, inv / s) and m == 1.0
    ntk, _ = scaled_inv_freq(d, base, "ntk", s)
    assert ntk[0] == inv[0] == 1.0                           # fastest pair untouched
    assert torch.isclose(ntk[-1], inv[-1] / s)               # slowest pair interpolated by exactly s
    assert torch.all(ntk[1:] < inv[1:]) and torch.all(ntk >= inv / s - 1e-9)


def test_yarn_keeps_fast_pairs_interpolates_slow_pairs():
    d, base, s, L = 32, 10000.0, 4.0, 128
    low, high = yarn_correction_range(d, base, L)
    assert (low, high) == (0, 6)                             # nano: pairs 0-6 make > 1 turn in 128 tokens
    inv = base_inv_freq(d, base)
    y, mscale = scaled_inv_freq(d, base, "yarn", s, original_len=L)
    assert y[0] == inv[0]                                    # ramp starts at 0 -> extrapolate
    assert torch.allclose(y[high:], inv[high:] / s)          # beyond high -> pure interpolation
    assert torch.all((y <= inv) & (y >= inv / s - 1e-9))
    assert mscale == pytest.approx(0.1 * math.log(4) + 1) and yarn_mscale(1.0) == 1.0
    c, _ = rope_tables(d, 8, base, "yarn", s, original_len=L)
    assert torch.allclose(c[0], torch.full((d // 2,), mscale))   # cos(0) * mscale


def test_set_rope_scaling_matches_config_and_runs_longer_sequences():
    m = small()
    with pytest.raises(AssertionError):
        m(torch.zeros(1, 40, dtype=torch.long))              # past the plain table (2 × 16)
    m.set_rope_scaling("yarn", 4)
    assert m.cfg.max_seq_len == 64 and m.rope_cos.shape[0] == 64
    logits, _ = m(torch.randint(0, 64, (2, 64)), all_logits=True)
    assert logits.shape == (2, 64, 64) and torch.isfinite(logits).all()
    fresh = small(rope_scaling="yarn", rope_factor=4.0, rope_original_context=16)
    assert torch.allclose(fresh.rope_cos, m.rope_cos) and torch.allclose(fresh.rope_sin, m.rope_sin)
    m.set_rope_scaling("", 1.0, max_len=48)                  # back to plain RoPE, bigger table only
    assert m.cfg.max_seq_len == 16 and m.rope_cos.shape[0] == 48
    assert torch.allclose(m.rope_cos[:32], small().rope_cos)


@pytest.mark.parametrize("extra", [{}, {"mla_kv_rank": 16, "mla_rope_dim": 8}])
def test_kv_cache_decoding_matches_full_forward_with_yarn(extra):
    m = small(**extra).set_rope_scaling("yarn", 4).eval()
    x = torch.randint(0, 64, (1, 50))
    full, _ = m(x, all_logits=True)
    cache = m.new_cache()
    step, _ = m(x[:, :30], cache=cache, all_logits=True)
    outs = [step]
    for t in range(30, 50):
        o, _ = m(x[:, t:t + 1], cache=cache)
        outs.append(o)
    assert torch.allclose(torch.cat(outs, 1), full, atol=1e-4)


def test_generation_uses_extended_context():
    m = small().set_rope_scaling("ntk", 4)
    x = torch.randint(0, 64, (1, 40))
    toks = list(stream(m, x, max_new_tokens=10, temperature=0, repetition_penalty=1.0))
    assert len(toks) == 10
    # the whole 40-token prompt (> context_length 16) is used, not just its last 16 tokens
    assert toks[0] == m(x)[0][0, -1].argmax().item()
