import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_model, tiny_tok  # noqa: E402
from llm.data import TokenDataset  # noqa: E402
from llm.muon import Muon, muon_param_groups, newton_schulz  # noqa: E402
from llm.trainer import TrainConfig, train  # noqa: E402


@pytest.mark.parametrize("shape", [(16, 16), (8, 32), (48, 12)])
def test_newton_schulz_pushes_singular_values_towards_one(shape):
    torch.manual_seed(0)
    # badly conditioned input: singular values spread over 3 orders of magnitude
    u, _ = torch.linalg.qr(torch.randn(shape[0], shape[0]))
    v, _ = torch.linalg.qr(torch.randn(shape[1], shape[1]))
    k = min(shape)
    G = u[:, :k] @ torch.diag(torch.logspace(0, -3, k)) @ v[:, :k].T * 7.0
    O = newton_schulz(G, steps=10)
    s = torch.linalg.svdvals(O)
    assert O.shape == G.shape
    assert s.max() < 1.25 and s[: k // 2].min() > 0.6          # the dominant half is ≈ orthogonal
    # same directions as the exact polar factor U·Vᵀ
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    cos = torch.nn.functional.cosine_similarity(O.flatten(), (U @ Vh).flatten(), dim=0)
    assert cos > 0.9


def test_param_groups_split_hidden_matrices_from_embeddings_and_norms():
    m = tiny_model(64)
    groups = muon_param_groups(m, weight_decay=0.1)
    muon_ids = {id(p) for g in groups if g["use_muon"] for p in g["params"]}
    assert id(m.tok_emb.weight) not in muon_ids and id(m.lm_head.weight) not in muon_ids
    assert id(m.blocks[0].attn.wq.weight) in muon_ids and id(m.blocks[1].ffn.w_down.weight) in muon_ids
    assert all(p.dim() == 2 for g in groups if g["use_muon"] for p in g["params"])
    assert sum(len(g["params"]) for g in groups) == len(list(m.parameters()))   # nothing lost
    with pytest.raises(ValueError):
        Muon([{"params": [next(p for p in m.parameters() if p.dim() == 1)], "use_muon": True}])


def test_adamw_branch_matches_torch_adamw():
    torch.manual_seed(0)
    a = torch.randn(5, 3, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    ours = Muon([{"params": [a], "use_muon": False, "weight_decay": 0.1}], lr=1e-2, betas=(0.9, 0.95))
    ref = torch.optim.AdamW([b], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
    for _ in range(5):
        g = torch.randn(5, 3)
        a.grad, b.grad = g.clone(), g.clone()
        ours.step()
        ref.step()
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_muon_update_rms_matches_adamw_scale():
    torch.manual_seed(0)
    w = torch.zeros(64, 16, requires_grad=True)
    opt = Muon([{"params": [w], "use_muon": True}], lr=1.0, momentum=0.0)
    w.grad = torch.randn(64, 16)
    opt.step()
    # orthogonal [64,16] has RMS 1/√64; × 0.2·√64 → RMS ≈ 0.2 (a typical AdamW update RMS)
    assert 0.15 < w.pow(2).mean().sqrt().item() < 0.25


def test_training_with_muon_reduces_loss_and_default_is_adamw(tmp_path):
    t = tiny_tok()
    m = tiny_model(t.vocab_size)
    assert isinstance(m.configure_optimizer(1e-3), torch.optim.AdamW)
    assert isinstance(m.configure_optimizer(1e-3, optimizer="muon"), Muon)
    with pytest.raises(ValueError):
        m.configure_optimizer(1e-3, optimizer="sgd")
    ds = TokenDataset(torch.tensor(t.encode(TEXT * 3)), context_length=32)
    x, y = ds.batch("val", 8)
    before = m(x, y)[1].item()
    train(m, ds, TrainConfig(max_steps=60, batch_size=8, eval_every=1000, eval_iters=1, warmup_steps=5,
                             optimizer="muon", out_dir=str(tmp_path)), device="cpu", log=lambda *_: None)
    assert m(x, y)[1].item() < before * 0.7
