import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_tok  # noqa: E402
from llm.config import ModelConfig  # noqa: E402
from llm.data import TokenDataset  # noqa: E402
from llm.generate import stream  # noqa: E402
from llm.model import GPT, KVCache  # noqa: E402
from llm.moe import MoEFFN  # noqa: E402
from llm.trainer import TrainConfig, train  # noqa: E402


def moe_model(vocab, n_experts=4, top_k=2):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=vocab, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                           context_length=32, n_experts=n_experts, moe_top_k=top_k))


def test_moe_layer_shape_and_aux_loss():
    torch.manual_seed(0)
    layer = MoEFFN(d_model=16, hidden=32, n_experts=4, top_k=2)
    y = layer(torch.randn(3, 7, 16))
    assert y.shape == (3, 7, 16)
    # aux loss is >= its uniform-load minimum (= top_k) and the load is a distribution
    assert layer.aux_loss.item() >= 2.0 - 1e-4
    torch.testing.assert_close(layer.last_load.sum(), torch.tensor(1.0))


def test_top1_with_one_expert_equals_that_expert():
    layer = MoEFFN(d_model=8, hidden=16, n_experts=1, top_k=1)
    x = torch.randn(2, 5, 8)
    torch.testing.assert_close(layer(x), layer.experts[0](x))


def test_moe_model_kv_cache_and_generation():
    m = moe_model(64).eval()
    x = torch.randint(0, 64, (1, 10))
    full, _ = m(x, all_logits=True)
    cache = KVCache(m.cfg.n_layers)
    m(x[:, :6], cache=cache)
    steps = torch.cat([m(x[:, i:i + 1], cache=cache)[0] for i in range(6, 10)], 1)
    torch.testing.assert_close(steps, full[:, 6:10], atol=1e-4, rtol=1e-4)
    assert len(list(stream(m, x[:, :3], max_new_tokens=10))) == 10


def test_moe_trains_and_uses_all_experts(tmp_path):
    t = tiny_tok()
    m = moe_model(t.vocab_size)
    ds = TokenDataset(torch.tensor(t.encode(TEXT * 3)), context_length=32)
    x, y = ds.batch("val", 8)
    m.eval()
    before = m(x, y)[1].item()
    train(m, ds, TrainConfig(max_steps=60, batch_size=8, eval_every=1000, eval_iters=1,
                             warmup_steps=5, out_dir=str(tmp_path)), device="cpu", log=lambda *_: None)
    m.eval()
    assert m(x, y)[1].item() < before * 0.7
    m.train()
    m(x, y)
    for block in m.blocks:
        assert (block.ffn.last_load > 0).all(), "an expert received no tokens (router collapse)"
