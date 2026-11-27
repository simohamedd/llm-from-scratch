import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import tiny_model  # noqa: E402
from llm.config import ModelConfig  # noqa: E402
from llm.generate import stream  # noqa: E402
from llm.model import GPT  # noqa: E402
from llm.quantize import (Int4Linear, bits_per_weight, dequantize_groups_int4, model_size_mb, pack_int4,  # noqa: E402
                          quantize_groups_int4, quantize_int4, quantize_int8, unpack_int4)


def test_pack_unpack_roundtrip():
    q = torch.randint(0, 16, (5, 64), dtype=torch.uint8)
    packed = pack_int4(q)
    assert packed.shape == (5, 32) and packed.dtype == torch.uint8
    assert torch.equal(unpack_int4(packed), q)
    assert pack_int4(torch.tensor([[3, 12]])).item() == 3 | (12 << 4)   # low nibble = even index


@pytest.mark.parametrize("symmetric", [True, False])
def test_codes_and_reconstruction_error_bound(symmetric):
    torch.manual_seed(0)
    w = torch.randn(16, 96)
    packed, scale, offset = quantize_groups_int4(w, 32, symmetric)
    assert packed.shape == (16, 48) and scale.shape == (16, 3) and scale.dtype == torch.float16
    assert (offset is None) == symmetric
    w_hat = dequantize_groups_int4(packed, scale, offset, 96, 32)
    # rounding error is at most half a quantization step per group (plus fp16 slack); Q4_0 can be off by
    # up to one full step for values near -m, which round to +8 and are clipped to code +7
    step = scale.float().abs().repeat_interleave(32, dim=1)
    bound = 1.0 if symmetric else 0.5
    assert ((w - w_hat).abs() <= bound * step * 1.01 + 1e-3).all()
    assert (w - w_hat).abs().mean() < 0.3 * step.mean()


def test_q4_0_reconstructs_group_extreme_exactly():
    w = torch.tensor([[0.1, -0.3, 2.0, 0.5, -1.0, 0.0, 0.25, -0.75]])
    packed, scale, _ = quantize_groups_int4(w, 8, symmetric=True)
    codes = unpack_int4(packed)[0].int() - 8
    assert codes.min() == -8 and codes.max() <= 7          # the extreme maps onto -8
    assert dequantize_groups_int4(packed, scale, None, 8, 8)[0, 2].item() == pytest.approx(2.0, rel=1e-3)


def test_q4_1_handles_shifted_weights_better():
    torch.manual_seed(0)
    w = torch.rand(32, 128) * 0.5 + 1.0          # all positive, far from zero: offset helps
    err = {}
    for sym in (True, False):
        p, s, o = quantize_groups_int4(w, 32, sym)
        err[sym] = (w - dequantize_groups_int4(p, s, o, 128, 32)).pow(2).mean()
    assert err[False] < err[True]


def test_smaller_groups_lower_error():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    w[:, ::50] *= 20                               # a few outliers per row
    errs = []
    for gs in (16, 64, 256):
        p, s, o = quantize_groups_int4(w, gs, True)
        errs.append((w - dequantize_groups_int4(p, s, o, 256, gs)).pow(2).mean().item())
    assert errs[0] < errs[1] < errs[2]


def test_padding_when_in_features_not_multiple_of_group():
    torch.manual_seed(0)
    lin = nn.Linear(40, 8, bias=True)
    q = Int4Linear(lin, group_size=32)
    assert q.w_packed.shape == (8, 32) and q.scale.shape == (8, 2)
    x = torch.randn(3, 40)
    assert torch.nn.functional.cosine_similarity(q(x).flatten(), lin(x).flatten(), dim=0) > 0.98
    assert q.dequantize().shape == (8, 40)


def test_all_zero_group_is_safe():
    w = torch.zeros(4, 32)
    w[0, 0] = 1.0
    for sym in (True, False):
        p, s, o = quantize_groups_int4(w, 16, sym)
        w_hat = dequantize_groups_int4(p, s, o, 32, 16)
        assert torch.isfinite(w_hat).all() and w_hat[1:].abs().max() == 0


def test_bad_group_size():
    with pytest.raises(ValueError):
        quantize_groups_int4(torch.randn(4, 8), group_size=3)


@pytest.mark.parametrize("symmetric", [True, False])
def test_quantized_model_close_and_smaller(symmetric):
    m = tiny_model(64).eval()
    x = torch.randint(0, 64, (2, 12))
    ref = m(x)[0]
    size = model_size_mb(m)
    q8 = quantize_int8(copy.deepcopy(m))
    quantize_int4(m, group_size=16, symmetric=symmetric)
    n_q = sum(isinstance(mod, Int4Linear) for mod in m.modules())
    assert n_q == 2 * 7                             # wq wk wv wo + w_gate w_up w_down per block
    assert isinstance(m.lm_head, nn.Linear)         # head skipped by default
    assert model_size_mb(m) < model_size_mb(q8) < size
    out = m(x)[0]
    assert torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0) > 0.97


def test_bits_per_weight():
    m = quantize_int4(tiny_model(64), group_size=32, symmetric=True)
    assert bits_per_weight(m) == pytest.approx(4 + 16 / 32)
    m = quantize_int4(tiny_model(64), group_size=32, symmetric=False)
    assert bits_per_weight(m) == pytest.approx(4 + 32 / 32)


def test_state_dict_roundtrip():
    m = quantize_int4(tiny_model(64), group_size=16).eval()
    m2 = quantize_int4(tiny_model(64), group_size=16).eval()
    for mod in m2.modules():                       # scramble, then reload
        if isinstance(mod, Int4Linear):
            mod.w_packed.zero_()
    m2.load_state_dict(m.state_dict())
    x = torch.randint(0, 64, (1, 8))
    assert torch.equal(m(x)[0], m2(x)[0])


@pytest.mark.parametrize("kw", [
    {"n_experts": 4, "moe_top_k": 2, "moe_shared_experts": 1},
    {"mla_kv_rank": 16, "mla_rope_dim": 4},
    {"mtp_depth": 1},
])
def test_works_with_other_architectures_and_generation(kw):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, context_length=32, **kw)
    m = GPT(cfg).eval()
    x = torch.randint(0, 64, (1, 10))
    ref = m(x)[0]
    quantize_int4(m, group_size=16)
    if kw.get("n_experts"):
        assert all(isinstance(mod.router, nn.Linear) for mod in m.modules() if hasattr(mod, "router"))
    assert torch.nn.functional.cosine_similarity(m(x)[0].flatten(), ref.flatten(), dim=0) > 0.95
    toks = list(stream(m, x, 5, temperature=0))
    assert len(toks) == 5
