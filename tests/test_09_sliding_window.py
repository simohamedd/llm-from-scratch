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
from llm.sliding_window import attention_mask, layer_window  # noqa: E402
from llm.stability import softcapped_attention  # noqa: E402
from llm.trainer import TrainConfig, train  # noqa: E402


def swa_model(vocab, **kw):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=vocab, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                           context_length=32, **kw))


def test_mask_is_banded_causal_and_offset_for_cache():
    m = attention_mask(6, 6, window=3)
    assert m[5].tolist() == [False, False, False, True, True, True]
    assert m[1].tolist() == [True, True, False, False, False, False]
    assert torch.equal(attention_mask(6, 6), torch.ones(6, 6, dtype=torch.bool).tril())
    # 2 queries on top of 4 cached keys: they are key positions 4 and 5
    assert torch.equal(attention_mask(2, 6, window=3), m[4:])


def test_layer_window_pattern():
    assert [layer_window(8, 0, i) for i in range(4)] == [8, 8, 8, 8]       # Mistral
    assert [layer_window(8, 2, i) for i in range(4)] == [8, 0, 8, 0]       # Gemma 2
    assert [layer_window(0, 2, i) for i in range(4)] == [0, 0, 0, 0]       # off


def test_window_matches_reference_and_big_window_is_full_attention():
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask(9, 9, 4))
    torch.testing.assert_close(softcapped_attention(q, k, v, cap=1e6, window=4), ref, atol=1e-5, rtol=1e-4)
    x = torch.randint(0, 64, (2, 20))
    full, _ = swa_model(64)(x, all_logits=True)
    big, _ = swa_model(64, sliding_window=32)(x, all_logits=True)     # window >= length: identical
    torch.testing.assert_close(big, full, atol=1e-5, rtol=1e-5)
    small, _ = swa_model(64, sliding_window=4)(x, all_logits=True)
    torch.testing.assert_close(small[:, :4], full[:, :4], atol=1e-5, rtol=1e-5)  # first 4 unaffected
    assert not torch.allclose(small[:, 10:], full[:, 10:], atol=1e-3)


def test_rolling_kv_cache_is_bounded_and_matches_full_forward():
    for kw in ({"sliding_window": 4}, {"sliding_window": 4, "global_every": 2},
               {"sliding_window": 4, "attn_softcap": 5.0}):
        m = swa_model(64, **kw).eval()
        x = torch.randint(0, 64, (1, 20))
        full, _ = m(x, all_logits=True)
        cache = m.new_cache()
        m(x[:, :7], cache=cache)                                        # prefill
        steps = torch.cat([m(x[:, i:i + 1], cache=cache)[0] for i in range(7, 20)], 1)
        torch.testing.assert_close(steps, full[:, 7:20], atol=1e-4, rtol=1e-4)
        assert cache.length == 20
        assert cache.k[0].shape[2] == 3                                 # window - 1 keys kept
        if kw.get("global_every") == 2:
            assert cache.k[1].shape[2] == 20                            # global layer keeps all
        # chunked prefill on top of a trimmed cache uses the offset banded mask
        cache = m.new_cache()
        m(x[:, :9], cache=cache)
        chunk, _ = m(x[:, 9:20], cache=cache, all_logits=True)
        torch.testing.assert_close(chunk, full[:, 9:20], atol=1e-4, rtol=1e-4)
        assert len(list(stream(m, x[:, :3], max_new_tokens=40))) == 40   # past context_length


def test_plain_kv_cache_unchanged():
    c = KVCache(2)
    k = torch.randn(1, 2, 5, 8)
    c.update(0, k, k)
    c.update(0, k[:, :, :1], k[:, :, :1])
    assert c.length == 6 and c.k[0].shape[2] == 6


def test_sliding_window_model_trains(tmp_path):
    t = tiny_tok()
    m = swa_model(t.vocab_size, sliding_window=8, global_every=2)
    ds = TokenDataset(torch.tensor(t.encode(TEXT * 3)), context_length=32)
    x, y = ds.batch("val", 8)
    m.eval()
    before = m(x, y)[1].item()
    train(m, ds, TrainConfig(max_steps=60, batch_size=8, eval_every=1000, eval_iters=1,
                             warmup_steps=5, out_dir=str(tmp_path)), device="cpu", log=lambda *_: None)
    m.eval()
    assert m(x, y)[1].item() < before * 0.7
