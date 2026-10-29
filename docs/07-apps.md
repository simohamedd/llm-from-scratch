# Chapter 7 — Building apps on top of LLMs

**Branch:** `chapter/07-apps` · status: 🚧 open for contributions

Once a model speaks the OpenAI API (chapter 6), the whole application ecosystem opens up.

| Pattern | Idea | Build-from-scratch task |
|---|---|---|
| **RAG** | retrieve relevant documents, put them in the prompt | embed chunks with the model's hidden states, cosine search, stuff top-k into the prompt |
| **Tool use / function calling** | model emits a structured call, your code runs it | teach a `<|tool|>` token via SFT, parse JSON, return results |
| **Agents** | loop: think → call tool → observe → repeat | a 100-line ReAct loop over the local API |
| **MCP** | standard protocol to expose tools/data to any model | wrap a tool as an MCP server and call it |

Use the hosted SDKs (Anthropic, OpenAI) side by side with your own model to compare behaviour.
The LLM Radar tracks LangGraph, LlamaIndex, MCP and the SDKs so this chapter stays current.
