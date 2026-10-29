import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_model, tiny_tok  # noqa: E402,F401
from llm.model import KVCache  # noqa: E402


def test_kv_cache_matches_full_forward():
    m = tiny_model(64).eval()
    x = torch.randint(0, 64, (1, 12))
    full, _ = m(x, all_logits=True)
    cache = KVCache(m.cfg.n_layers)
    m(x[:, :8], cache=cache)                                  # prefill
    step_logits = [m(x[:, i:i + 1], cache=cache)[0] for i in range(8, 12)]
    torch.testing.assert_close(torch.cat(step_logits, 1), full[:, 8:12], atol=1e-4, rtol=1e-4)
