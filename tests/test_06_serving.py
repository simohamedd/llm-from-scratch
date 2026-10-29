import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_model, tiny_tok  # noqa: E402,F401
from llm.quantize import model_size_mb, quantize_int8  # noqa: E402


def test_int8_quantization_is_close_and_smaller():
    m = tiny_model(64).eval()
    x = torch.randint(0, 64, (1, 10))
    ref = m(x)[0]
    size = model_size_mb(m)
    quantize_int8(m)
    assert model_size_mb(m) < size
    assert torch.nn.functional.cosine_similarity(m(x)[0].flatten(), ref.flatten(), dim=0) > 0.99


def test_openai_compatible_api():
    from fastapi.testclient import TestClient

    from llm.serve import build_app
    t = tiny_tok()
    client = TestClient(build_app(tiny_model(t.vocab_size), t))
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
    assert r.status_code == 200 and r.json()["object"] == "chat.completion"
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 5})
    assert r.text.strip().endswith("data: [DONE]")
