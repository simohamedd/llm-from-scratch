import sys
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_model, tiny_tok  # noqa: E402
from llm.checkpointing import checkpoint, count_saved_bytes  # noqa: E402
from llm.config import ModelConfig  # noqa: E402
from llm.data import TokenDataset  # noqa: E402
from llm.lora import apply_lora  # noqa: E402
from llm.model import GPT  # noqa: E402
from llm.trainer import TrainConfig, train  # noqa: E402

VOCAB = 64


def _model(**kw):
    torch.manual_seed(0)
    base = dict(vocab_size=VOCAB, d_model=32, n_layers=3, n_heads=4, n_kv_heads=2, context_length=32)
    return GPT(ModelConfig(**{**base, **kw}))


def _batch(B=2, T=24):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, VOCAB, (B, T), generator=g), torch.randint(0, VOCAB, (B, T), generator=g)


def _grads(model, x, y, seed=0):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)                     # same dropout masks for both runs
    _, loss = model(x, y)
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


VARIANTS = [
    {},
    {"dropout": 0.2},                                            # RNG must be replayed exactly
    {"n_experts": 4, "moe_top_k": 2},                            # aux loss is a side output with a gradient
    {"n_experts": 4, "moe_balance": "bias", "moe_shared_experts": 1},
    {"mla_kv_rank": 16, "mla_rope_dim": 4, "qk_norm": False},
    {"sliding_window": 8, "global_every": 2, "qk_norm": True, "attn_softcap": 30.0},
]


@pytest.mark.parametrize("kw", VARIANTS)
def test_checkpointed_gradients_match_plain_backward(kw):
    x, y = _batch()
    plain = _model(**kw).train()
    ckpt = _model(**kw, grad_checkpoint=True).train()
    ckpt.load_state_dict(plain.state_dict())
    loss_a, g_a = _grads(plain, x, y)
    loss_b, g_b = _grads(ckpt, x, y)
    assert torch.allclose(loss_a, loss_b, atol=1e-6)
    assert g_a.keys() == g_b.keys() and len(g_a) > 0
    for n in g_a:
        assert torch.allclose(g_a[n], g_b[n], atol=1e-6, rtol=1e-5), n


def test_moe_bias_update_runs_once_per_step():
    x, y = _batch()
    plain = _model(n_experts=4, moe_balance="bias", moe_bias_rate=0.01).train()
    ckpt = _model(n_experts=4, moe_balance="bias", moe_bias_rate=0.01, grad_checkpoint=True).train()
    ckpt.load_state_dict(plain.state_dict())
    _grads(plain, x, y)
    _grads(ckpt, x, y)                          # backward re-runs forward: must NOT update twice
    for a, b in zip(plain.blocks, ckpt.blocks):
        assert torch.equal(a.ffn.expert_bias, b.ffn.expert_bias)
        assert a.ffn.expert_bias.abs().max() > 0


def test_matches_torch_checkpoint_and_supports_autograd_grad():
    torch.manual_seed(0)
    f = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(16, 4))
    x = torch.randn(5, 8, requires_grad=True)
    torch.manual_seed(3)
    ours = torch.autograd.grad(checkpoint(f, x).pow(2).sum(), [x, *f.parameters()])
    torch.manual_seed(3)
    ref = torch.autograd.grad(torch.utils.checkpoint.checkpoint(f, x, use_reentrant=False).pow(2).sum(),
                              [x, *f.parameters()])
    for a, b in zip(ours, ref):
        assert torch.allclose(a, b, atol=1e-6)


def test_frozen_input_still_trains_lora_weights():
    """LoRA freezes the embedding, so block inputs need no grad — the classic reentrant pitfall."""
    x, y = _batch()
    m = apply_lora(_model(grad_checkpoint=True)).train()
    _, loss = m(x, y)
    loss.backward()
    lora = [p for n, p in m.named_parameters() if "lora_" in n]
    assert lora and all(p.grad is not None for p in lora)
    assert any(p.grad.abs().sum() > 0 for p in lora)


def test_saves_less_activation_memory_and_skips_eval():
    x, y = _batch(B=4, T=32)
    sizes = {}
    for name, kw in {"off": {}, "every2": {"grad_checkpoint": True, "checkpoint_every": 2},
                     "all": {"grad_checkpoint": True}}.items():
        m = _model(n_layers=4, **kw).train()
        with count_saved_bytes() as saved:
            _, loss = m(x, y)
        sizes[name] = saved[0]
        loss.backward()
    assert sizes["all"] < sizes["every2"] < sizes["off"]
    assert sizes["all"] < 0.5 * sizes["off"]
    # eval / generation never takes the checkpoint path, and outputs are identical
    a, b = _model().eval(), _model(grad_checkpoint=True).eval()
    with torch.no_grad():
        assert torch.equal(a(x, all_logits=True)[0], b(x, all_logits=True)[0])


def test_default_is_off_and_training_reduces_loss(tmp_path):
    assert ModelConfig().grad_checkpoint is False
    tok = tiny_tok()
    ds = TokenDataset(torch.tensor(tok.encode(TEXT * 3)), context_length=32)
    model = tiny_model(tok.vocab_size)
    model.cfg.grad_checkpoint = True
    tc = TrainConfig(max_steps=30, batch_size=8, lr=3e-3, warmup_steps=3, eval_every=30, eval_iters=2,
                     out_dir=str(tmp_path))
    logs = []
    train(model, ds, tc, device="cpu", log=logs.append)
    first = float(logs[1].split("val ")[1].split()[0])
    last = float(logs[-1].split("val ")[1].split()[0])
    assert last < first
