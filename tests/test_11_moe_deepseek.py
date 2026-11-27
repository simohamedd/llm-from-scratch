import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_tok  # noqa: E402
from llm.config import ModelConfig  # noqa: E402
from llm.data import TokenDataset  # noqa: E402
from llm.model import GPT, KVCache  # noqa: E402
from llm.moe import MoEFFN, expert_hidden  # noqa: E402
from llm.trainer import TrainConfig, train  # noqa: E402


def ds_model(vocab, **kw):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=vocab, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                           context_length=32, n_experts=4, moe_top_k=2, **kw))


def test_shared_expert_is_added_to_every_token():
    torch.manual_seed(0)
    layer = MoEFFN(d_model=8, hidden=16, n_experts=2, top_k=2, n_shared=1)
    x = torch.randn(2, 5, 8)
    routed_only = MoEFFN(d_model=8, hidden=16, n_experts=2, top_k=2)
    routed_only.load_state_dict({k: v for k, v in layer.state_dict().items() if not k.startswith("shared")})
    torch.testing.assert_close(layer(x), routed_only(x) + layer.shared[0](x))
    # active compute stays at the dense budget: (top_k + shared) experts of 1/(top_k + shared) width
    assert expert_hidden(96, 8 / 3, 8, 2, n_shared=2) == 64


def test_bias_changes_selection_but_not_mixing_weights():
    torch.manual_seed(0)
    layer = MoEFFN(d_model=8, hidden=16, n_experts=4, top_k=1, balance="bias").eval()
    x = torch.randn(1, 6, 8)
    layer.expert_bias[:] = torch.tensor([0.0, 0.0, 0.0, 10.0])   # huge bias forces expert 3
    _, weights, idx = layer.route(x.view(-1, 8))
    assert (idx == 3).all()
    torch.testing.assert_close(weights, torch.ones_like(weights))  # top-1 weight is s/s = 1, bias-free
    torch.testing.assert_close(layer(x), layer.experts[3](x))
    assert layer.aux_loss.item() == 0.0


def test_bias_update_pushes_toward_uniform_load():
    layer = MoEFFN(d_model=8, hidden=16, n_experts=4, top_k=1, balance="bias", bias_rate=0.1)
    layer.update_bias(torch.tensor([0.7, 0.1, 0.1, 0.1]))
    torch.testing.assert_close(layer.expert_bias, torch.tensor([-0.1, 0.1, 0.1, 0.1]))
    # eval-mode forwards never touch the bias
    layer.eval()(torch.randn(1, 4, 8))
    torch.testing.assert_close(layer.expert_bias, torch.tensor([-0.1, 0.1, 0.1, 0.1]))


def test_bias_balancing_fixes_a_skewed_router():
    torch.manual_seed(0)
    layer = MoEFFN(d_model=8, hidden=16, n_experts=4, top_k=1, balance="bias", bias_rate=0.05)
    with torch.no_grad():
        layer.router.weight[0] += 3.0          # expert 0 wins for most inputs
    x = torch.randn(4, 64, 8).abs()
    layer(x)
    skewed = layer.last_load.max().item()
    for _ in range(60):
        layer(x)
    assert skewed > 0.6 and layer.last_load.max().item() < 0.4


def test_deepseek_moe_model_trains_and_caches(tmp_path):
    t = tiny_tok()
    m = ds_model(t.vocab_size, moe_shared_experts=1, moe_balance="bias", moe_bias_rate=0.01)
    ds = TokenDataset(torch.tensor(t.encode(TEXT * 3)), context_length=32)
    x, y = ds.batch("val", 8)
    m.eval()
    before = m(x, y)[1].item()
    train(m, ds, TrainConfig(max_steps=60, batch_size=8, eval_every=1000, eval_iters=1,
                             warmup_steps=5, out_dir=str(tmp_path)), device="cpu", log=lambda *_: None)
    m.eval()
    assert m(x, y)[1].item() < before * 0.7
    assert any(b.ffn.expert_bias.abs().sum() > 0 for b in m.blocks)   # biases moved during training
    assert "blocks.0.ffn.expert_bias" in m.state_dict()                # and are saved in checkpoints
    seq = torch.randint(0, t.vocab_size, (1, 10))
    full, _ = m(seq, all_logits=True)
    cache = KVCache(m.cfg.n_layers)
    m(seq[:, :6], cache=cache)
    steps = torch.cat([m(seq[:, i:i + 1], cache=cache)[0] for i in range(6, 10)], 1)
    torch.testing.assert_close(steps, full[:, 6:10], atol=1e-4, rtol=1e-4)
