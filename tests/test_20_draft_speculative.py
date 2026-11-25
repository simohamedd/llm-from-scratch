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
from llm.speculative import SpecStats, draft_speculative_stream, filtered_probs, speculative_generate  # noqa: E402
from llm.trainer import TrainConfig, train  # noqa: E402

VOCAB = 64


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    base = dict(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, context_length=32)
    return GPT(ModelConfig(**{**base, **kw})).double().eval()


def _greedy(model, x, n):
    return list(stream(model, x, n, temperature=0, repetition_penalty=1.0))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A target and a smaller draft trained on the same repetitive text, so they often agree."""
    tok = tiny_tok()
    ds = TokenDataset(torch.tensor(tok.encode(TEXT)), 48)
    out = []
    for d_model, n_layers in ((32, 2), (16, 1)):
        torch.manual_seed(0)
        m = GPT(ModelConfig(vocab_size=tok.vocab_size, d_model=d_model, n_layers=n_layers, n_heads=4, n_kv_heads=2,
                            context_length=48))
        tc = TrainConfig(max_steps=80, batch_size=8, lr=3e-3, warmup_steps=10, eval_every=1000, eval_iters=1,
                         out_dir=str(tmp_path_factory.mktemp("ckpt")))
        train(m, ds, tc, device="cpu", log=lambda s: None)
        out.append(m.double().eval())
    return out[0], out[1], tok


@pytest.mark.parametrize("dkw", [{}, {"d_model": 16, "n_layers": 1}, {"mla_kv_rank": 16, "mla_rope_dim": 4},
                                 {"n_experts": 4, "moe_top_k": 2}])
def test_greedy_matches_target_whatever_the_draft(dkw):
    target = _model(0)
    draft = _model(1, **dkw)                       # random, unrelated weights: mostly rejected drafts
    x = torch.randint(0, VOCAB, (1, 5), generator=torch.Generator().manual_seed(3))
    ref = _greedy(target, x, 20)
    for k in (1, 3, 7):
        assert list(draft_speculative_stream(target, draft, x, 20, n_draft=k)) == ref


def test_draft_equal_to_target_accepts_everything():
    target = _model(0)
    x = torch.randint(0, VOCAB, (1, 4), generator=torch.Generator().manual_seed(5))
    st = SpecStats()
    out = list(draft_speculative_stream(target, target, x, 21, n_draft=4, stats=st))
    assert out == _greedy(target, x, 21)
    assert st.acceptance_rate == 1.0
    assert st.main_calls == 1 + 4                  # prefill + 4 rounds of (4 drafts + bonus) = 1 + 20 tokens
    assert st.tokens_per_call == pytest.approx(21 / 4)


def test_trained_draft_is_accepted_and_saves_target_passes(trained):
    target, draft, tok = trained
    x = torch.tensor([tok.encode("The quick")])
    ref = _greedy(target, x, 30)
    st = SpecStats()
    assert list(draft_speculative_stream(target, draft, x, 30, n_draft=3, stats=st)) == ref
    assert st.accepted > 0 and st.tokens_per_call > 1.0
    assert st.main_calls < 1 + 30
    assert sum(st.accepted_at_depth) == st.accepted
    assert st.accepted_at_depth == sorted(st.accepted_at_depth, reverse=True)   # depth j needs 1..j-1 accepted


def test_draft_cache_equals_fresh_recompute(trained):
    """After every round the draft's cached guesses are exactly what a cache-free draft would produce."""
    target, draft, tok = trained
    x = torch.tensor([tok.encode("The quick brown")])
    st = SpecStats()
    list(draft_speculative_stream(target, draft, x, 25, n_draft=3, stats=st))
    for L, drafts in st.drafts:
        out = _greedy(target, x, L - x.shape[1])  # committed tokens at that round
        seq = x[0].tolist() + out
        ref = []
        for _ in drafts:
            logits, _ = draft(torch.tensor([seq + ref]))
            ref.append(int(logits[0, -1].argmax()))
        assert drafts == ref


def test_stops_at_end_of_context():
    target, draft = _model(0), _model(1, d_model=16, n_layers=1)
    x = torch.randint(0, VOCAB, (1, 28), generator=torch.Generator().manual_seed(1))
    out = list(draft_speculative_stream(target, draft, x, 100, n_draft=4))
    assert out == _greedy(target, x, len(out))
    assert 28 + len(out) == 33                     # last token is emitted when the 32-token window is full


def test_sampled_second_token_marginal_is_exact():
    """With sampling, P(second emitted token) must equal the target's, whatever the draft."""
    target, draft = _model(0, vocab_size=8), _model(1, vocab_size=8, d_model=16, n_layers=1)
    x = torch.tensor([[1, 2, 3]])
    logits, _ = target(x)
    p1 = filtered_probs(logits[0, -1], 1.5)
    nxt, _ = target(torch.cat([x.repeat(8, 1), torch.arange(8)[:, None]], dim=1))
    p2 = (p1[:, None] * filtered_probs(nxt[:, -1], 1.5)).sum(0)     # exact marginal of the 2nd token
    counts = torch.zeros(8, dtype=torch.double)
    torch.manual_seed(0)
    n = 1500
    for _ in range(n):
        out = list(draft_speculative_stream(target, draft, x, 2, temperature=1.5, n_draft=2))
        counts[out[1]] += 1      # out[1] is a draft-accepted or resampled token in the first round
    assert (counts / n - p2).abs().max() < 0.05


def test_errors_and_stop_ids_and_generate():
    target = _model(0)
    x = torch.tensor([[1, 2]])
    with pytest.raises(ValueError):
        list(draft_speculative_stream(target, _model(1, vocab_size=VOCAB + 1), x, 5))
    with pytest.raises(ValueError):
        list(draft_speculative_stream(target, _model(1, sliding_window=8), x, 5))
    with pytest.raises(ValueError):
        list(draft_speculative_stream(target, target, x, 5, n_draft=0))
    ref = _greedy(target, x, 10)
    stop = ref[5]
    out = list(draft_speculative_stream(target, _model(1), x, 10, n_draft=3, stop_ids={stop}))
    assert out == ref[:ref.index(stop)]
    tok = tiny_tok()
    t, d = _model(0, vocab_size=tok.vocab_size), _model(1, vocab_size=tok.vocab_size, d_model=16, n_layers=1)
    from llm.generate import generate
    assert speculative_generate(t, tok, "The", draft=d, max_new_tokens=8) == \
        generate(t, tok, "The", max_new_tokens=8, temperature=0, repetition_penalty=1.0)
