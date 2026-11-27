import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import tiny_tok  # noqa: E402
from llm.batching import batch_generate_ids, batch_stream, generate_batch, left_pad, padded_attention_mask  # noqa: E402
from llm.config import ModelConfig  # noqa: E402
from llm.generate import generate, stream  # noqa: E402
from llm.model import GPT  # noqa: E402

VOCAB = 64
PROMPTS = [[5, 9, 2], [7, 1, 3, 3, 8, 12, 40, 2, 6], [33], [4, 4, 20, 21, 22, 23]]

CONFIGS = [
    {},
    {"sliding_window": 4, "global_every": 2},
    {"sliding_window": 3},
    {"attn_softcap": 20.0, "final_softcap": 15.0},
    {"qk_norm": True},
    {"mla_kv_rank": 16, "mla_rope_dim": 4},
    {"mla_kv_rank": 16, "mla_rope_dim": 4, "mla_q_rank": 12, "attn_softcap": 20.0},
    {"n_experts": 4, "moe_top_k": 2, "moe_shared_experts": 1},
    {"rope_scaling": "yarn", "rope_factor": 2.0},
]


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    base = dict(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, context_length=32)
    return GPT(ModelConfig(**{**base, **kw})).double().eval()


def _single(model, p, n, **kw):
    kw = {"temperature": 0, "repetition_penalty": 1.0, **kw}
    return list(stream(model, torch.tensor([p]), n, **kw))


def test_left_pad():
    ids, valid = left_pad([[1, 2], [3, 4, 5, 6], [7]], pad_id=0)
    assert ids.tolist() == [[0, 0, 1, 2], [3, 4, 5, 6], [0, 0, 0, 7]]
    assert valid.tolist() == [[False, False, True, True], [True] * 4, [False, False, False, True]]
    with pytest.raises(ValueError):
        left_pad([[1], []], pad_id=0)


def test_default_path_has_no_padding_mask():
    model = _model()
    cache = model.new_cache()
    assert cache.key_valid is None and padded_attention_mask(3, 3, 0, cache) is None
    # every query must see at least itself, even a pad query with no real key before it
    cache.key_valid = torch.tensor([[False, False, True], [True, True, True]])
    m = padded_attention_mask(3, 3, 0, cache)
    assert m.shape == (2, 1, 3, 3) and m.any(-1).all()
    assert m[0, 0].tolist() == [[True, False, False], [False, True, False], [False, False, True]]


@pytest.mark.parametrize("kw", CONFIGS)
def test_padded_prefill_logits_match_unpadded(kw):
    """RoPE is relative, so shifting a prompt right by p pads must not change its logits."""
    model = _model(**kw)
    ids, valid = left_pad(PROMPTS, pad_id=0)
    cache = model.new_cache()
    cache.key_valid = valid
    logits, _ = model(ids, cache=cache, all_logits=True)
    assert torch.isfinite(logits).all()
    L = ids.shape[1]
    for b, p in enumerate(PROMPTS):
        ref, _ = model(torch.tensor([p]), all_logits=True)
        torch.testing.assert_close(logits[b, L - len(p):], ref[0], atol=1e-6, rtol=1e-5)   # RoPE tables are float32


@pytest.mark.parametrize("kw", CONFIGS)
def test_batched_greedy_equals_one_at_a_time(kw):
    model = _model(**kw)
    out = batch_generate_ids(model, PROMPTS, max_new_tokens=12, temperature=0, repetition_penalty=1.0)
    assert out == [_single(model, p, 12) for p in PROMPTS]


def test_merged_mla_weights_batched():
    model = _model(mla_kv_rank=16, mla_rope_dim=4)
    ref = [_single(model, p, 10) for p in PROMPTS]
    model.merge_mla_weights()
    assert batch_generate_ids(model, PROMPTS, max_new_tokens=10, temperature=0, repetition_penalty=1.0) == ref


def test_pad_id_does_not_matter_and_repetition_penalty_ignores_pads():
    model = _model()
    kw = dict(max_new_tokens=12, temperature=0, repetition_penalty=1.5)
    ref = [_single(model, p, 12, repetition_penalty=1.5) for p in PROMPTS]
    for pad in (0, 5, 63):   # 5 also occurs in a prompt: still only the real tokens are penalised
        assert batch_generate_ids(model, PROMPTS, pad_id=pad, **kw) == ref


def test_rows_stop_independently():
    model = _model()
    free = [_single(model, p, 12) for p in PROMPTS]
    stop = {free[1][3]}                         # some token row 1 produces at step 4
    ref = [_single(model, p, 12, stop_ids=stop) for p in PROMPTS]
    steps = list(batch_stream(model, PROMPTS, 12, temperature=0, repetition_penalty=1.0, stop_ids=stop))
    out = [[s[b] for s in steps if s[b] is not None] for b in range(len(PROMPTS))]
    assert out == ref
    # once a row is done it stays None
    for b in range(len(PROMPTS)):
        col = [s[b] is None for s in steps]
        assert col == sorted(col)
    # stops early when every row is done
    everything = {t for row in free for t in row} | set(range(VOCAB))
    assert list(batch_stream(model, PROMPTS, 12, temperature=0, stop_ids=everything)) == []


def test_stops_at_context_length():
    model = _model()
    out = batch_generate_ids(model, PROMPTS, max_new_tokens=100, temperature=0, repetition_penalty=1.0)
    L = max(len(p) for p in PROMPTS)
    assert all(len(o) == 32 - L + 1 for o in out)   # padded length never exceeds the context
    assert out[1] == _single(model, PROMPTS[1], 32 - L + 1)


def test_sampling_batch_runs_and_is_seeded():
    model = _model()
    torch.manual_seed(1)
    a = batch_generate_ids(model, PROMPTS, max_new_tokens=8, temperature=0.9, top_k=20, top_p=0.9)
    torch.manual_seed(1)
    b = batch_generate_ids(model, PROMPTS, max_new_tokens=8, temperature=0.9, top_k=20, top_p=0.9)
    assert a == b and all(len(o) == 8 and all(0 <= t < VOCAB for t in o) for o in a)


def test_generate_batch_text_matches_generate():
    tok = tiny_tok()
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                            context_length=48)).double().eval()
    prompts = ["The quick", "fox", "lazy dog. The quick brown"]
    kw = dict(max_new_tokens=10, temperature=0, repetition_penalty=1.0)
    assert generate_batch(model, tok, prompts, **kw) == [generate(model, tok, p, **kw) for p in prompts]
