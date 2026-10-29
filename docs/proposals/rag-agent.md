# Proposal — RAG + tool-calling agent

**Chapter:** 07-apps

## Plan
- [ ] `apps/rag.py`: chunk documents, embed with the model's mean-pooled hidden states, cosine top-k, stuff into the prompt
- [ ] `apps/agent.py`: ReAct loop (think → tool → observe) over the local OpenAI-compatible API (`python -m llm.serve`)
- [ ] SFT examples teaching a `<|tool|>` JSON call format
- [ ] expose one tool as an MCP server and call it from the agent
- [ ] side-by-side comparison with a hosted model through the Anthropic / OpenAI SDKs
