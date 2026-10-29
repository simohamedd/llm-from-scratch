# Chapter 6 — Using your model like a real LLM

**Code:** [`llm/serve.py`](../llm/serve.py), [`llm/quantize.py`](../llm/quantize.py) · **Branch:** `chapter/06-serving`

## OpenAI-compatible API
```bash
python -m llm.serve --ckpt checkpoints/sft.pt
```
Endpoints: `GET /v1/models`, `GET /health`, `POST /v1/chat/completions` (with `"stream": true` → Server-Sent Events).

Any OpenAI-compatible client works — including LangChain, LlamaIndex and most agent frameworks:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
r = client.chat.completions.create(model="tiny-llm", messages=[{"role": "user", "content": "Say hello."}])
print(r.choices[0].message.content)
```
```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello."}]}'
```

## int8 quantization
```python
from llm.quantize import quantize_int8, model_size_mb
quantize_int8(model)   # Linear weights -> int8 + per-row scale (~4x smaller)
```

## Production equivalents
| This repo | Production tool |
|---|---|
| `serve.py` | vLLM, SGLang, TGI, Ollama |
| `quantize.py` | llama.cpp GGUF, bitsandbytes, AWQ, GPTQ |
| KV cache | PagedAttention (vLLM), prefix caching |
