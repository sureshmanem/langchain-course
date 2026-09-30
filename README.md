# Agents Under the Hood

A hands-on look at how LLM agents work internally. Instead of relying on a pre-built agent framework, this project hand-rolls the core **tool-calling agent loop** (the ReAct pattern) with LangChain and a local Ollama model.

## Project structure

| File | Description |
| --- | --- |
| `1_agent_loop_langchain_tool_calling.py` | A shopping-assistant agent that uses `.bind_tools()` and a manual loop to call `get_product_price` and `apply_discount` tools until it reaches a final answer. |
| `1_agentLoopExplaination.md` | Step-by-step walkthrough of the agent loop script. |
| `main.py` | Minimal entry point / hello-world. |
| `pyproject.toml` / `uv.lock` | Project dependencies, managed with [uv](https://docs.astral.sh/uv/). |

## How the agent loop works

1. Send the conversation (system prompt + user question) to the LLM.
2. If the LLM responds with a tool call, run that Python function.
3. Append the tool result to the conversation as a `ToolMessage`.
4. Repeat until the LLM answers in plain text (or `MAX_ITERATIONS` is reached).

See [`1_agentLoopExplaination.md`](1_agentLoopExplaination.md) for a detailed explanation.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/) running locally, with a model that supports tool calling:

  ```bash
  ollama pull qwen3:1.7b
  ```

## Setup

Install dependencies:

```bash
uv sync
```

Optionally, create a `.env` file for LangSmith tracing (or an OpenAI key if you switch models):

```env
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=agents-under-the-hood
OPENAI_API_KEY=your-openai-key
```

## Running

```bash
uv run 1_agent_loop_langchain_tool_calling.py
```

Example output:

```text
Question: What is the price of a headphones after applying a gold discount?
============================================================

--- Iteration 1 ---
  [Tool Selected] get_product_price with args: {'product': 'headphones'}
    >> Executing get_product_price(product='headphones')
  [Tool Result] 149.95

--- Iteration 2 ---
  [Tool Selected] apply_discount with args: {'price': 149.95, 'discount_tier': 'gold'}
    >> Executing apply_discount(price=149.95, discount_tier='gold')
  [Tool Result] 115.46

--- Iteration 3 ---

Final Answer: ...
```

## Switching models

The model is created with LangChain's provider-agnostic `init_chat_model`. To use OpenAI instead of Ollama, change the model string in `run_agent`:

```python
llm = init_chat_model("openai:gpt-5", temperature=0)
```

and set `OPENAI_API_KEY` in your `.env`.
