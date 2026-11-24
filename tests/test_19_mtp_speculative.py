import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_tok  # noqa: E402
from llm.config import ModelConfig  # noqa: E402
from llm.data import TokenDataset  # noqa: E402
from llm.generate import stream  # noqa: E402
from llm.model import GPT  # noqa: E402
from llm.speculative import (SpecStats, accept_or_resample, filtered_probs,  # noqa: E402
                             mtp_speculative_stream, speculative_generate)
from llm.trainer import TrainConfig, train  # noqa: E402

VOCAB = 64


def _model(**kw):
    torch.manual_seed(0)
    base = dict(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, context_length=32, mtp_depth=2)
    return GPT(ModelConfig(**{**base, **kw})).double().eval()


def _greedy(model, x, n):
    return list(stream(model, x, n, temperature=0, repetition_penalty=1.0))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A tiny MTP model trained on very repetitive text, so drafts are often accepted."""
    tok = tiny_tok()
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                            context_length=48, mtp_depth=2))
    ds = TokenDataset(torch.tensor(tok.encode(TEXT)), 48)
    tc = TrainConfig(max_steps=120, batch_size=8, lr=3e-3, warmup_steps=10, eval_every=1000, eval_iters=1,
                     out_dir=str(tmp_path_factory.mktemp("ckpt")))
    train(model, ds, tc, device="cpu", log=lambda s: None)
    return model.double().eval(), tok


def _reference_drafts(model, seq, D):
    """Greedy drafts recomputed from scratch, without any KV cache (the training-time MTP path)."""
    x = torch.tensor([seq])
    n = len(seq) - 1                          # positions 0..n-1 are in the main model; seq[n] is pending
    cos, sin = model.rope_cos[:n], model.rope_sin[:n]
    h = model.tok_emb(x[:, :n])
    for b in model.blocks:
        h = b(h, cos, sin)
    ext, drafts = list(seq), []
    for k, mod in enumerate(model.mtp.depths[:D], start=1):
        h = mod(h, model.tok_emb(torch.tensor([ext[k:n + k]])), cos, sin)
        d = int(model.logits(mod.norm(h[:, -1])).argmax())
        drafts.append(d)
        ext.append(d)
    return drafts


@pytest.mark.parametrize("kw", [{}, {"mla_kv_rank": 16, "mla_rope_dim": 4}, {"n_experts": 4, "moe_top_k": 2},
                                {"qk_norm": True, "mtp_depth": 3}])
def test_greedy_matches_plain_decoding_random_weights(kw):
    model = _model(**kw)
    x = torch.randint(0, VOCAB, (1, 5), generator=torch.Generator().manual_seed(3))
    ref = _greedy(model, x, 20)
    for d in range(1, len(model.mtp) + 1):
        assert list(mtp_speculative_stream(model, x, 20, n_draft=d)) == ref


def test_greedy_matches_and_accepts_on_trained_model(trained):
    model, tok = trained
    x = torch.tensor([tok.encode("The quick")])
    ref = _greedy(model, x, 30)
    for d in (1, 2):
        st = SpecStats()
        assert list(mtp_speculative_stream(model, x, 30, n_draft=d, stats=st)) == ref
        assert st.accepted > 0 and st.tokens_per_call > 1.0
        assert st.main_calls < 1 + 30                 # fewer main passes than plain decoding
        assert sum(st.accepted_at_depth) == st.accepted
        assert st.accepted_at_depth[0] >= st.accepted_at_depth[-1]


def test_cached_drafts_equal_uncached_recomputation(trained):
    """The per-depth KV caches (with truncation of draft-dependent positions) are exact."""
    model, tok = trained
    x = torch.tensor([tok.encode("The quick brown")])
    st = SpecStats()
    out = list(mtp_speculative_stream(model, x, 30, n_draft=2, stats=st))
    seq = x[0].tolist() + out
    assert st.drafts
    for L, drafts in st.drafts:
        assert drafts == _reference_drafts(model, seq[:L], len(drafts)), L


def test_runs_to_the_end_of_the_context_like_stream():
    model = _model(mtp_depth=3)
    x = torch.randint(0, VOCAB, (1, 20), generator=torch.Generator().manual_seed(1))
    out = list(mtp_speculative_stream(model, x, 100))
    assert len(out) == 32 - 20 + 1                    # stops once the context is full
    assert out == _greedy(model, x, len(out))


def test_accept_or_resample_preserves_target_distribution():
    """Speculative sampling's guarantee: the emitted token is distributed exactly as p."""
    torch.manual_seed(0)
    V, N = 6, 20000
    p = torch.softmax(torch.randn(V) * 1.5, -1)
    q = torch.softmax(torch.randn(V) * 1.5, -1)
    counts = torch.zeros(V)
    accepted = 0
    for _ in range(N):
        d = int(torch.multinomial(q, 1))
        ok, tok = accept_or_resample(p, q, d, greedy=False)
        assert (tok == d) if ok else True
        accepted += ok
        counts[tok] += 1
    assert (counts / N - p).abs().max() < 0.015
    # expected acceptance = sum_x min(p, q)
    assert abs(accepted / N - torch.minimum(p, q).sum().item()) < 0.015


def test_sampled_second_token_has_target_distribution():
    """End to end at temperature 1: token 2 (draft accepted or resampled) has the same marginal as plain sampling."""
    model = _model(vocab_size=8, mtp_depth=1)
    x = torch.tensor([[1, 2, 3]])
    # exact marginal of the 2nd generated token: sum_a p(a | x) p(b | x, a)
    p1 = torch.softmax(model(x)[0][0, -1].double(), -1)
    exact = sum(p1[a] * torch.softmax(model(torch.tensor([[1, 2, 3, a]]))[0][0, -1].double(), -1) for a in range(8))
    torch.manual_seed(0)
    N, counts = 3000, torch.zeros(8)
    for _ in range(N):
        out = list(mtp_speculative_stream(model, x, 2, temperature=1.0))
        counts[out[1]] += 1
    assert (counts / N - exact).abs().max() < 0.03


def test_filtered_probs_matches_sample_next_filters():
    logits = torch.randn(VOCAB, generator=torch.Generator().manual_seed(0))
    p = filtered_probs(logits, 0.7, top_k=10, top_p=0.9)
    assert torch.isclose(p.sum(), torch.tensor(1.0)) and (p > 0).sum() <= 10
    g = filtered_probs(logits, 0.0)
    assert g.sum() == 1 and g.argmax() == logits.argmax()


def test_stop_ids_max_tokens_and_errors():
    model = _model()
    x = torch.tensor([[1, 2, 3]])
    ref = _greedy(model, x, 10)
    assert list(mtp_speculative_stream(model, x, 4)) == ref[:4]
    stop = ref[2]
    assert list(mtp_speculative_stream(model, x, 10, stop_ids={stop})) == ref[:ref.index(stop)]
    with pytest.raises(ValueError):
        list(mtp_speculative_stream(_model(mtp_depth=0), x, 4))
    with pytest.raises(ValueError):
        list(mtp_speculative_stream(model, x, 4, n_draft=3))
    with pytest.raises(ValueError):
        list(mtp_speculative_stream(_model(sliding_window=8), x, 4))


def test_speculative_generate_text(trained):
    model, tok = trained
    assert isinstance(speculative_generate(model, tok, "The", max_new_tokens=8), str)
