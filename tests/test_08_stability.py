import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_tok  # noqa: E402
from llm.config import ModelConfig  # noqa: E402
from llm.data import TokenDataset  # noqa: E402
from llm.generate import stream  # noqa: E402
from llm.model import GPT, KVCache  # noqa: E402
from llm.stability import softcap, softcapped_attention  # noqa: E402
from llm.trainer import TrainConfig, train  # noqa: E402


def stable_model(vocab, **kw):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=vocab, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                           context_length=32, **kw))


def test_softcap_bounds_and_near_identity():
    x = torch.tensor([-1e4, -100.0, -0.5, 0.0, 0.5, 100.0, 1e4], requires_grad=True)
    y = softcap(x, 30.0)
    assert y.abs().max() <= 30.0
    torch.testing.assert_close(y[2:5], x[2:5], atol=1e-3, rtol=0)     # ~identity near 0
    y.sum().backward()
    assert (x.grad[1:-1] > 0).all()          # unlike clipping, gradient survives past the cap
    assert softcap(x, 0.0) is x                                         # cap <= 0 means off


def test_softcapped_attention_matches_sdpa_with_huge_cap():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 8)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(softcapped_attention(q, k, v, cap=1e6), ref, atol=1e-5, rtol=1e-4)


def test_defaults_unchanged_and_qk_norm_adds_only_norm_weights():
    base, qk = stable_model(64), stable_model(64, qk_norm=True)
    extra = qk.num_params() - base.num_params()
    assert extra == 2 * 2 * base.cfg.head_dim          # q_norm + k_norm per layer, 2 layers
    assert base.blocks[0].attn.q_norm is None


def test_stable_model_kv_cache_generation_and_capped_logits():
    m = stable_model(64, qk_norm=True, attn_softcap=5.0, final_softcap=2.0).eval()
    x = torch.randint(0, 64, (1, 10))
    full, _ = m(x, all_logits=True)
    assert full.abs().max() <= 2.0
    cache = KVCache(m.cfg.n_layers)
    m(x[:, :6], cache=cache)
    steps = torch.cat([m(x[:, i:i + 1], cache=cache)[0] for i in range(6, 10)], 1)
    torch.testing.assert_close(steps, full[:, 6:10], atol=1e-4, rtol=1e-4)
    # prefill of several tokens on top of a cache uses the offset causal mask
    cache = KVCache(m.cfg.n_layers)
    m(x[:, :4], cache=cache)
    chunk, _ = m(x[:, 4:10], cache=cache, all_logits=True)
    torch.testing.assert_close(chunk, full[:, 4:10], atol=1e-4, rtol=1e-4)
    assert len(list(stream(m, x[:, :3], max_new_tokens=8))) == 8


def test_stable_model_trains(tmp_path):
    t = tiny_tok()
    m = stable_model(t.vocab_size, qk_norm=True, attn_softcap=50.0, final_softcap=30.0)
    ds = TokenDataset(torch.tensor(t.encode(TEXT * 3)), context_length=32)
    x, y = ds.batch("val", 8)
    m.eval()
    before = m(x, y)[1].item()
    train(m, ds, TrainConfig(max_steps=60, batch_size=8, eval_every=1000, eval_iters=1,
                             warmup_steps=5, out_dir=str(tmp_path)), device="cpu", log=lambda *_: None)
    m.eval()
    assert m(x, y)[1].item() < before * 0.7
