# Proposal — GRPO for reasoning

**Chapter:** 05-finetuning · **Inspired by:** DeepSeek-R1, TRL `GRPOTrainer`

Group Relative Policy Optimization: sample G answers per prompt, score each with a *verifiable*
reward (e.g. is the arithmetic correct?), and use `(reward − group mean) / group std` as the
advantage. No value network, no learned reward model.

## Plan
- [ ] toy task: 2-digit addition, reward = exact match
- [ ] `grpo()` in `llm/finetune.py`: group sampling, normalised advantages, clipped ratio, KL to reference
- [ ] plot accuracy vs steps; compare with SFT-only
