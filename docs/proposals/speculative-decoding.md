# Proposal — Speculative decoding

**Chapter:** 04-inference · **Inspired by:** Leviathan et al. 2023, vLLM, llama.cpp

A small draft model proposes k tokens; the large model verifies all k in one forward pass and
accepts a prefix via rejection sampling. Identical output distribution, fewer large-model calls.

## Plan
- [x] `draft_speculative_stream(target, draft, ...)` + `speculative_generate(..., draft=)` in `llm/speculative.py`
- [x] draft = nano, target = micro (shared tokenizer)
- [x] test: greedy output identical to target-only greedy
- [x] benchmark tokens/sec and acceptance rate vs k

**Status:** implemented on `develop` (2026-11-25) — see [`docs/advanced/draft-model-speculative-decoding.md`](../advanced/draft-model-speculative-decoding.md).
