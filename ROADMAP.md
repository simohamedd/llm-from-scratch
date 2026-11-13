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
- [ ] QK-norm + logit soft-capping (OLMo 2 / Gemma 2 training stability)
- [ ] Sliding-window attention option (Mistral / Gemma)
- [ ] Multi-head Latent Attention — compressed KV cache (DeepSeek-V2/V3)
- [ ] Shared expert + aux-loss-free load balancing for MoE (DeepSeek-V3)
- [ ] YaRN / NTK RoPE scaling for longer context at inference

## Training
- [ ] Muon optimizer option for matrix weights
- [ ] Warmup-Stable-Decay (WSD) learning-rate schedule
- [ ] Gradient checkpointing flag to save memory
- [ ] Multi-GPU training with DistributedDataParallel (`torchrun`)
- [ ] Multi-token prediction auxiliary head (DeepSeek-V3)

## Inference
- [ ] Speculative decoding with a draft model — see `feature/speculative-decoding`
- [ ] Batched generation with padding + left-padded attention masks
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
