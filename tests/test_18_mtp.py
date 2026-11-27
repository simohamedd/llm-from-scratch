import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_tok  # noqa: E402
from llm.config import ModelConfig  # noqa: E402
from llm.data import TokenDataset  # noqa: E402
from llm.generate import generate  # noqa: E402
from llm.model import GPT  # noqa: E402
from llm.trainer import TrainConfig, load_checkpoint, train  # noqa: E402

VOCAB = 64


def _model(**kw):
    torch.manual_seed(0)
    base = dict(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, context_length=32)
    return GPT(ModelConfig(**{**base, **kw}))


def _batch(B=2, T=20):
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, VOCAB, (B, T + 1), generator=g)
    return x[:, :-1].contiguous(), x[:, 1:].contiguous()


def test_default_has_no_mtp_and_same_params():
    plain, mtp = _model(), _model(mtp_depth=2)
    assert plain.mtp is None
    assert mtp.num_params(include_mtp=False) == plain.num_params()
    assert mtp.num_params() > plain.num_params()
    # embedding and head are shared, not copied
    assert all(p is not mtp.tok_emb.weight for p in mtp.mtp.parameters())


def test_eval_loss_and_logits_unchanged_by_mtp_heads():
    """MTP heads are built after the trunk is initialised: same seed → same trunk → same eval output."""
    plain, mtp = _model().eval(), _model(mtp_depth=1).eval()
    for k, v in plain.state_dict().items():
        assert torch.equal(v, mtp.state_dict()[k]), k
    x, y = _batch()
    la, lb = plain(x, y), mtp(x, y)
    assert torch.allclose(la[0], lb[0]) and torch.allclose(la[1], lb[1])


def test_train_loss_is_main_plus_weighted_mtp():
    model = _model(mtp_depth=2, mtp_weight=0.5).train()
    x, y = _batch()
    _, loss = model(x, y)
    assert len(model.last_mtp_losses) == 2
    model.eval()
    _, main = model(x, y)
    expected = main + 0.5 * torch.stack(model.last_mtp_losses).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_depth_k_targets_are_tokens_k_plus_1_ahead():
    """Re-derive depth-1 and depth-2 losses by hand from the definition."""
    model = _model(mtp_depth=2).train()
    x, y = _batch(T=12)
    _, _ = model(x, y)
    T = x.shape[1]
    cos, sin = model.rope_cos[:T], model.rope_sin[:T]
    h = model.tok_emb(x)
    for b in model.blocks:
        h = b(h, cos, sin)
    full = torch.cat([x, y[:, -1:]], dim=1)        # t_0..t_T
    for k, mod in enumerate(model.mtp.depths, start=1):
        n = T - k
        h = mod(h[:, :n], model.tok_emb(full[:, k:k + n]), cos[:n], sin[:n])
        logits = model.logits(mod.norm(h))
        ref = F.cross_entropy(logits.reshape(-1, VOCAB), full[:, k + 1:k + 1 + n].reshape(-1))
        assert torch.allclose(ref, model.last_mtp_losses[k - 1], atol=1e-6)


def test_ignore_index_and_short_sequences():
    model = _model(mtp_depth=3).train()
    x, y = _batch(T=8)
    y = y.clone()
    y[:, :4] = -100                                # e.g. SFT prompt tokens are masked
    _, loss = model(x, y)
    assert torch.isfinite(loss)
    x2, y2 = _batch(T=2)                           # shorter than depth 3: deeper heads are skipped
    _, loss2 = model(x2, y2)
    assert torch.isfinite(loss2) and len(model.last_mtp_losses) == 1


@pytest.mark.parametrize("kw", [{}, {"n_experts": 4, "moe_top_k": 2}, {"mla_kv_rank": 16, "mla_rope_dim": 4},
                                {"grad_checkpoint": True}])
def test_gradients_reach_trunk_and_mtp_modules(kw):
    model = _model(mtp_depth=1, **kw).train()
    x, y = _batch()
    _, loss = model(x, y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model.mtp.depths[0].proj.weight.grad.abs().sum() > 0


def test_mtp_trains_generates_and_reloads(tmp_path):
    tok = tiny_tok()
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                            context_length=32, mtp_depth=1))
    ds = TokenDataset(torch.tensor(tok.encode(TEXT)), 32)
    lines = []
    tc = TrainConfig(max_steps=60, batch_size=8, lr=3e-3, warmup_steps=5, eval_every=30, eval_iters=2,
                     out_dir=str(tmp_path))
    train(model, ds, tc, device="cpu", log=lines.append)
    vals = [float(line.split("val ")[1].split()[0]) for line in lines if "val " in line]
    assert vals[-1] < vals[0]
    assert "mtp " in lines[-1]
    out = generate(model, tok, "The", max_new_tokens=5)
    assert isinstance(out, str)
    re = load_checkpoint(tmp_path / "last.pt")
    assert re.mtp is not None and len(re.mtp) == 1
