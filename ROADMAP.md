# Roadmap — daily improvement backlog

A scheduled job works through this list **one item per day**: it takes the first unchecked item,
builds it from scratch with tests and a doc, runs the full test suite, ticks the box, logs it in
[`CHANGELOG.md`](CHANGELOG.md) and pushes to `develop`. `main` is only updated by a human merge.

Rules for every item:
- from scratch in plain PyTorch, readable (~100-300 lines), no new heavy dependencies
- a test in `tests/`, a doc in `docs/advanced/` (or the matching chapter doc), README updated
- the whole suite must pass; if an item can't be finished in one day, split it and do the first part
- if `radar/state.json` / `docs/LLM_RADAR.md` shows a notable new technique, it may be inserted near the top

## Model architecture
- [x] Mixture of Experts FFN with load-balancing loss — `llm/moe.py`
- [x] QK-norm + logit soft-capping (OLMo 2 / Gemma 2 training stability) — `llm/stability.py`
- [x] Sliding-window attention + rolling KV cache (Mistral / Gemma) — `llm/sliding_window.py`
- [x] Multi-head Latent Attention — compressed KV cache (DeepSeek-V2/V3) — `llm/mla.py`
- [x] Shared expert + aux-loss-free load balancing for MoE (DeepSeek-V3) — `llm/moe.py`
- [x] YaRN / NTK RoPE scaling for longer context at inference — `llm/rope_scaling.py`
- [x] MLA query compression + precomputed absorbed weights for faster decoding (follow-up to `llm/mla.py`)

## Training
- [x] Muon optimizer option for matrix weights — `llm/muon.py`
- [x] Warmup-Stable-Decay (WSD) learning-rate schedule — `llm/schedules.py`
- [x] Gradient checkpointing flag to save memory — `llm/checkpointing.py`
- [x] Multi-GPU training with DistributedDataParallel (`torchrun`) — `llm/distributed.py`
- [x] Multi-token prediction auxiliary head (DeepSeek-V3) — `llm/mtp.py`
- [x] Speculative decoding using the MTP head as the draft (follow-up to `llm/mtp.py`) — `llm/speculative.py`

## Inference
- [x] Speculative decoding with a draft model — `llm/speculative.py` (`draft_speculative_stream`)
- [x] Batched generation with padding + left-padded attention masks — `llm/batching.py`
- [ ] int4 group-wise quantization (GGUF Q4-style)
- [ ] Prefix caching in the server for repeated system prompts

## Fine-tuning & alignment
- [ ] GRPO reasoning on a 2-digit addition task — see `feature/grpo-reasoning`
- [ ] QLoRA: LoRA on top of the int8/int4 base model
- [ ] Evaluation script: perplexity + small exact-match benchmark, results table in docs

## Apps & tooling
- [ ] RAG over local documents using the model's embeddings — see `feature/rag-agent`
- [ ] Tool-calling agent loop over the OpenAI-compatible API
- [ ] Export weights to `safetensors` + a Hugging Face–style `config.json`
- [ ] Google Colab notebook that runs the whole course on a free GPU
- [ ] Multilingual tokenizer experiment (Arabic / French / English compression comparison)
