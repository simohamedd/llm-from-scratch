# Proposal — Speculative decoding

**Chapter:** 04-inference · **Inspired by:** Leviathan et al. 2023, vLLM, llama.cpp

A small draft model proposes k tokens; the large model verifies all k in one forward pass and
accepts a prefix via rejection sampling. Identical output distribution, fewer large-model calls.

## Plan
- [ ] `speculative_generate(target, draft, ...)` in `llm/generate.py`
- [ ] draft = nano, target = micro (shared tokenizer)
- [ ] test: greedy output identical to target-only greedy
- [ ] benchmark tokens/sec and acceptance rate vs k
