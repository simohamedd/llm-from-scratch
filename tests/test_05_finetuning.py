import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_model, tiny_tok  # noqa: E402,F401
from llm.finetune import dpo, sft  # noqa: E402
from llm.lora import apply_lora, merge_lora  # noqa: E402


def test_lora_sft_dpo_and_merge():
    t = tiny_tok()
    m = tiny_model(t.vocab_size)
    apply_lora(m, r=4)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) < m.num_params() * 0.2
    ex = [{"prompt": "hi", "response": "hello there"}, {"prompt": "bye", "response": "see you"}]
    sft(m, t, ex, steps=5, batch_size=2, log=lambda *_: None)
    dpo(m, t, [{"prompt": "hi", "chosen": "hello", "rejected": "go away"}], steps=3, batch_size=1, log=lambda *_: None)
    x = torch.randint(0, t.vocab_size, (1, 6))
    before = m(x)[0]
    merge_lora(m)
    torch.testing.assert_close(m(x)[0], before, atol=1e-4, rtol=1e-4)
