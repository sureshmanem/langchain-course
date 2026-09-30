# Agents Under the Hood

A hands-on look at how LLM agents work internally. Instead of relying on a pre-built agent framework, this project hand-rolls the core **tool-calling agent loop** (the ReAct pattern) with LangChain and a local Ollama model.

## Project structure

| File | Description |
| --- | --- |
| `1_agent_loop_langchain_tool_calling.py` | A shopping-assistant agent that uses `.bind_tools()` and a manual loop to call `get_product_price` and `apply_discount` tools until it reaches a final answer. |
| `2_agent_loop_raw_function_calling.py` | The same agent without LangChain: calls the raw Ollama / OpenAI SDKs, with hand-written JSON tool schemas and manual LangSmith tracing. |
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

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

`OPENAI_API_KEY` is only needed for the OpenAI provider, and the `LANGSMITH_*` keys are optional (for tracing).

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

Choose the LLM with `--provider` (defaults to local Ollama):

```bash
uv run 1_agent_loop_langchain_tool_calling.py --provider ollama   # ollama:qwen3:1.7b
uv run 1_agent_loop_langchain_tool_calling.py --provider openai   # openai:gpt-5
```

The same `--provider` flag works for `2_agent_loop_raw_function_calling.py`.

You can also set the default in `.env` with `LLM_PROVIDER=openai`. The OpenAI provider requires `OPENAI_API_KEY`. To change the model names, edit the `MODELS` dict in each script.
