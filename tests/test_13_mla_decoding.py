import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_tok  # noqa: E402
from llm.config import ModelConfig  # noqa: E402
from llm.data import TokenDataset  # noqa: E402
from llm.generate import stream  # noqa: E402
from llm.lora import apply_lora  # noqa: E402
from llm.model import GPT  # noqa: E402
from llm.trainer import TrainConfig, train  # noqa: E402


def mla_model(vocab, **kw):
    torch.manual_seed(0)
    kw.setdefault("mla_kv_rank", 16)
    kw.setdefault("mla_rope_dim", 4)
    return GPT(ModelConfig(vocab_size=vocab, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                           context_length=32, **kw))


def test_query_compression_is_opt_in_and_changes_params():
    assert ModelConfig().mla_q_rank == 0
    full, low = mla_model(64), mla_model(64, mla_q_rank=8)
    a_full, a_low = full.blocks[0].attn, low.blocks[0].attn
    assert hasattr(a_full, "wq") and not hasattr(a_full, "w_dq")
    assert not hasattr(a_low, "wq") and a_low.w_dq.out_features == 8
    # 32*(4*(8+4)) = 1536 query weights  vs  32*8 + 8 (norm) + 8*48 = 648
    q_params = lambda a: sum(p.numel() for n, p in a.named_parameters() if n.startswith(("wq", "w_dq", "q_norm", "w_uq")))
    assert q_params(a_full) == 1536 and q_params(a_low) == 648
    # the KV cache is unaffected by query compression
    assert full.num_params() - low.num_params() == 2 * (1536 - 648)


def test_merged_absorbed_naive_paths_agree():
    for kw in ({}, {"mla_q_rank": 8}, {"attn_softcap": 5.0}, {"sliding_window": 5}):
        m = mla_model(64, **kw).eval()
        attn = m.blocks[0].attn
        x = torch.randn(2, 12, 32)
        cos, sin = m.rope_cos[:12], m.rope_sin[:12]
        naive = attn(x, cos, sin, absorb=False)
        absorbed = attn(x, cos, sin, absorb=True)
        attn.merge_weights()
        assert attn.merged and attn.w_ov.shape == (32, 4 * 16) and attn.w_qlat.shape[0] == 4 * 16
        merged = attn(x, cos, sin, absorb=True)
        torch.testing.assert_close(absorbed, naive, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(merged, naive, atol=1e-5, rtol=1e-4)
        assert "w_ov" not in attn.state_dict()                       # never saved in checkpoints


def test_merged_generation_matches_full_forward():
    for kw in ({}, {"mla_q_rank": 8}, {"sliding_window": 4, "global_every": 2}):
        m = mla_model(64, **kw).eval()
        x = torch.randint(0, 64, (1, 20))
        full, _ = m(x, all_logits=True)
        m.merge_mla_weights()
        assert all(b.attn.merged for b in m.blocks)
        cache = m.new_cache()
        m(x[:, :7], cache=cache)                                      # prefill
        steps = torch.cat([m(x[:, i:i + 1], cache=cache)[0] for i in range(7, 20)], 1)
        torch.testing.assert_close(steps, full[:, 7:20], atol=1e-4, rtol=1e-4)
        assert len(list(stream(m, x[:, :3], max_new_tokens=20))) == 20  # stream() calls eval(): stays merged
        assert all(b.attn.merged for b in m.blocks)


def test_train_mode_drops_merged_weights_and_lora_is_refused():
    m = mla_model(64).merge_mla_weights()
    m.train()
    assert not any(b.attn.merged for b in m.blocks)
    lora = apply_lora(mla_model(64), targets=("wq",))
    try:
        lora.merge_mla_weights()
    except TypeError as e:
        assert "LoRA" in str(e)
    else:
        raise AssertionError("merging on top of LoRA should be refused")


def test_query_compressed_model_trains(tmp_path):
    t = tiny_tok()
    m = mla_model(t.vocab_size, mla_q_rank=8)
    ds = TokenDataset(torch.tensor(t.encode(TEXT * 3)), context_length=32)
    x, y = ds.batch("val", 8)
    m.eval()
    before = m(x, y)[1].item()
    train(m, ds, TrainConfig(max_steps=60, batch_size=8, eval_every=1000, eval_iters=1,
                             warmup_steps=5, out_dir=str(tmp_path)), device="cpu", log=lambda *_: None)
    m.eval()
    assert m(x, y)[1].item() < before * 0.7
