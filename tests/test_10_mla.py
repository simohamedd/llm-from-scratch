import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_tok  # noqa: E402
from llm.config import ModelConfig  # noqa: E402
from llm.data import TokenDataset  # noqa: E402
from llm.generate import stream  # noqa: E402
from llm.mla import MLAttention, kv_cache_numbers_per_token  # noqa: E402
from llm.model import GPT  # noqa: E402
from llm.trainer import TrainConfig, train  # noqa: E402


def mla_model(vocab, **kw):
    torch.manual_seed(0)
    kw.setdefault("mla_kv_rank", 16)
    kw.setdefault("mla_rope_dim", 4)
    return GPT(ModelConfig(vocab_size=vocab, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                           context_length=32, **kw))


def test_default_is_gqa_and_mla_replaces_attention():
    assert ModelConfig().mla_kv_rank == 0
    m = mla_model(64)
    assert all(isinstance(b.attn, MLAttention) for b in m.blocks)
    assert m.rope_cos.shape[1] == 2                                   # rope_dim / 2 frequencies
    cfg = m.cfg
    assert kv_cache_numbers_per_token(cfg) == 16 + 4
    assert kv_cache_numbers_per_token(ModelConfig(d_model=32, n_heads=4, n_kv_heads=2)) == 2 * 2 * 8


def test_absorbed_attention_equals_naive():
    m = mla_model(64).eval()
    attn = m.blocks[0].attn
    x = torch.randn(2, 12, 32)
    cos, sin = m.rope_cos[:12], m.rope_sin[:12]
    for kw in ({}, {"window": 5}):
        attn.window = kw.get("window", 0)
        naive = attn(x, cos, sin, absorb=False)
        absorbed = attn(x, cos, sin, absorb=True)
        torch.testing.assert_close(absorbed, naive, atol=1e-5, rtol=1e-4)


def test_latent_cache_matches_full_forward_and_is_small():
    for kw in ({}, {"sliding_window": 4}, {"sliding_window": 4, "global_every": 2},
               {"attn_softcap": 5.0}):
        m = mla_model(64, **kw).eval()
        x = torch.randint(0, 64, (1, 20))
        full, _ = m(x, all_logits=True)
        cache = m.new_cache()
        m(x[:, :7], cache=cache)                                      # prefill
        steps = torch.cat([m(x[:, i:i + 1], cache=cache)[0] for i in range(7, 20)], 1)
        torch.testing.assert_close(steps, full[:, 7:20], atol=1e-4, rtol=1e-4)
        # the cache holds the latent (d_c=16) and the shared rotary key (d_r=4) — no full K/V
        assert cache.k[1].shape[-1] == 16 and cache.v[1].shape[-1] == 4 and cache.k[1].shape[1] == 1
        # chunked prefill on top of a cache
        cache = m.new_cache()
        m(x[:, :9], cache=cache)
        chunk, _ = m(x[:, 9:20], cache=cache, all_logits=True)
        torch.testing.assert_close(chunk, full[:, 9:20], atol=1e-4, rtol=1e-4)
        assert len(list(stream(m, x[:, :3], max_new_tokens=40))) == 40


def test_mla_model_trains(tmp_path):
    t = tiny_tok()
    m = mla_model(t.vocab_size)
    ds = TokenDataset(torch.tensor(t.encode(TEXT * 3)), context_length=32)
    x, y = ds.batch("val", 8)
    m.eval()
    before = m(x, y)[1].item()
    train(m, ds, TrainConfig(max_steps=60, batch_size=8, eval_every=1000, eval_iters=1,
                             warmup_steps=5, out_dir=str(tmp_path)), device="cpu", log=lambda *_: None)
    m.eval()
    assert m(x, y)[1].item() < before * 0.7
