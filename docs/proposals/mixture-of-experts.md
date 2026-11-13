# Proposal — Mixture of Experts (MoE) FFN

**Chapter:** 02-transformer · **Inspired by:** Mixtral, DeepSeek-V3, Qwen3-MoE

Replace the dense SwiGLU with E small experts plus a router; each token only uses its top-k
experts, so total parameters grow while compute per token stays flat.

## Plan
- [x] starter `MoEFFN(d_model, hidden, n_experts=8, top_k=2)` in `llm/moe.py`
- [x] load-balancing auxiliary loss (Switch Transformer) or aux-loss-free bias (DeepSeek-V3)
- [x] `ModelConfig.n_experts`; `Block` uses MoE when > 0
- [x] tests: output shape, routing weights sum to 1, every expert receives tokens
- [x] experiment: dense vs MoE at equal *active* parameters

**Status:** ✅ done 2026-11-13 → [docs/advanced/mixture-of-experts.md](../advanced/mixture-of-experts.md)
