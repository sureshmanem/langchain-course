# Agent Loop Explained: `1_agent_loop_langchain_tool_calling.py`

This document walks through how the script implements a **tool-calling agent loop** using LangChain, LangGraph-style manual looping (without LangGraph itself), and a local Ollama model.

## What this script demonstrates

Instead of using a pre-built agent framework, this script hand-rolls the core mechanism every agent framework is built on: a loop that

1. sends the conversation to an LLM,
2. lets the LLM decide whether it needs a tool,
3. runs that tool in Python if requested,
4. feeds the tool's result back into the conversation,
5. repeats until the LLM answers in plain text instead of requesting a tool.

This is often called the **ReAct pattern** (Reason + Act), and it's what powers agent frameworks like LangGraph, OpenAI's function-calling agents, and Claude's tool use under the hood.

---

## 1. Imports and setup

```python
from dotenv import load_dotenv
load_dotenv()
```

Loads variables from `.env` (API keys, LangSmith config) into the process environment **before** any LangChain/LangSmith modules are imported. This matters because `langsmith` reads `LANGSMITH_API_KEY` / `LANGSMITH_TRACING` at import time — if you imported it first, tracing could silently be misconfigured.

```python
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
```

- `init_chat_model` — a provider-agnostic factory. Passing `"ollama:qwen3:1.7b"` tells LangChain to build an Ollama chat model client without importing `langchain_ollama` directly.
- `tool` — a decorator that turns any Python function into a LangChain `Tool` object, using its type hints and docstring to build a JSON schema the LLM can read.
- `HumanMessage` / `SystemMessage` / `ToolMessage` — typed chat messages that make up the conversation history sent to the model each turn.
- `traceable` — a LangSmith decorator that logs a structured trace (inputs, outputs, timing, nested LLM/tool calls) of the decorated function to the LangSmith dashboard.

```python
MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"
```

- `MAX_ITERATIONS` is a safety valve. Without it, a model that keeps calling tools (or gets stuck in a request/response cycle) would loop forever.
- `MODEL` must be a model Ollama has pulled locally **and** one that supports "tools" capability (visible via `ollama list` / `ollama show <model>`).

---

## 2. Defining tools

```python
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)
```

Key details:
- The **docstring** (`"""Look up the price..."""`) is not just documentation — it becomes the tool's `description` field in the JSON schema sent to the LLM. The model uses this text to decide *when* to call the tool.
- The **type hint** `product: str` becomes the tool's parameter schema, telling the model what type of argument to supply.
- `prices.get(product, 0)` defaults unknown products to `0` rather than raising an exception, so a hallucinated product name degrades gracefully instead of crashing the loop.
- The `print(...)` line exists purely for developer visibility — it's how you can tell from the console that the LLM's tool call actually executed real Python code, not just text.

```python
@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    ...
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)
```

Same pattern: two typed parameters (`price`, `discount_tier`), a docstring that also tells the model the valid tier values (`bronze`, `silver`, `gold`), and a graceful fallback (`0` discount) for unrecognized tiers.

---

## 3. Setting up the agent

```python
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
```

- `tools` is the list passed to the model so it knows what's available.
- `tools_dict` is a **separate** lookup table keyed by tool name (e.g. `"get_product_price"`), used later to find the actual Python function to execute once the model requests it by name. This is necessary because the model returns tool calls as `{name, args}` — a plain string and a dict — not a reference to the Python object.

```python
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
```

- `init_chat_model` builds the raw chat client.
- `temperature=0` makes output deterministic — important for tool-calling agents, since you want consistent decisions about which tool to call and with what arguments, not creative variation.
- `.bind_tools(tools)` is the critical call: it serializes each `@tool`-decorated function into a JSON schema (name, description, parameters) and attaches it to every request sent to the model. This is what enables the model to respond with a structured `tool_calls` field instead of only plain text.

  > Note: an earlier version of this script used `.bind(tools=tools)`, which does **not** properly format tool schemas for the provider — it just attaches a raw keyword argument. Using `.bind_tools(tools)` is required for the model to reliably emit `tool_calls`.

---

## 4. The system prompt

```python
    messages = [
        SystemMessage(content=(
            "You are a helpful shopping assistant. ..."
            "1. NEVER guess or assume any product price. ..."
            "2. Only call apply_discount AFTER you have received a price ..."
            "3. NEVER calculate discounts yourself using math. ..."
            "4. If the user does not specify a discount tier, ask them ..."
        )),
        HumanMessage(content=question),
    ]
```

The system prompt is doing real behavioral work, not just flavor text. It enforces:

1. **No guessing prices** — forces a real tool call instead of the LLM hallucinating a plausible-looking number.
2. **Correct ordering** — discount must be applied to the *real* looked-up price, not a value the model invents.
3. **No mental math** — the model must delegate arithmetic to `apply_discount` rather than computing the discount itself (LLMs are unreliable at precise arithmetic).
4. **Ask instead of assume** — if the tier isn't specified, the model should ask the user rather than silently picking one (though in this script's `__main__` call the tier is always specified, so this branch isn't exercised).

This is a common technique for smaller/local models (like `qwen3:1.7b`) which are more prone to skip tool use and answer directly than larger hosted models.

---

## 5. The core loop

```python
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
```

Each iteration is one full round-trip to the LLM. `messages` contains the entire conversation so far, so the model has full context of what's already happened (previous tool calls and their results).

```python
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
```

This is the loop's **exit condition**. `ai_message.tool_calls` is populated only when the model decides it needs a tool; if it's empty, the model has produced its final natural-language answer, and the function returns immediately.

```python
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
```

- The loop only processes `tool_calls[0]` — the **first** tool call in the list — even if the model requested multiple tools in one turn. This is a deliberate simplification: it forces exactly one tool execution per iteration, keeping the trace easy to follow, at the cost of not supporting true parallel tool calls.
- `tool_call_id` is important: it's a unique identifier the model attaches to its tool request. The corresponding `ToolMessage` (sent back to the model next iteration) must carry this same ID so the model can match the result to the request it made — this is part of the OpenAI/Ollama tool-calling message protocol.

```python
        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")
```

- Look up the real Python tool object by the name the model returned.
- `raise ValueError(...)` is a fail-fast guard: if the model hallucinates a tool name that was never registered, the script crashes loudly instead of silently ignoring it or guessing.
- `tool_to_use.invoke(tool_args)` actually executes the Python function (e.g. `get_product_price(product="laptop")`) with the arguments the model supplied. This is the "Act" step of ReAct — real code runs here, not the LLM.

```python
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
```

Two messages are appended to history before the next loop iteration:

1. `ai_message` — the assistant's own message that *requested* the tool call, so the model can see (on the next turn) that it already asked for this.
2. A `ToolMessage` — the *result* of running that tool, tagged with the matching `tool_call_id` so the model knows which request this result answers.

Without appending both, the model would lose track of what it asked for and why, and the conversation would become incoherent.

```python
    print("ERROR: Max iterations reached without a final answer")
    return None
```

If the loop runs `MAX_ITERATIONS` times without the model ever returning a tool-call-free message, the function gives up and returns `None`. This guards against infinite loops from a model that keeps requesting tools indefinitely.

---

## 6. Entry point

```python
if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a headphones after applying a gold discount?")
```

Runs the agent with a single hardcoded question. A typical execution trace looks like:

```
--- Iteration 1 ---
  [Tool Selected] get_product_price with args: {'product': 'headphones'}
    >> Executing get_product_price(product='headphones')
  [Tool Result] 149.95

--- Iteration 2 ---
  [Tool Selected] apply_discount with args: {'price': 149.95, 'discount_tier': 'gold'}
    >> Executing apply_discount(price=149.95, discount_tier='gold')
  [Tool Result] 115.46

--- Iteration 3 ---

Final Answer: The price of the headphones after applying the gold discount is $115.46.
```

Three round trips to the LLM total: one to request the price lookup, one to request the discount application (now that it has the real price), and one to produce the final answer once both tool results are in context.

---

## Key takeaways

| Concept | Where it shows up | Why it matters |
|---|---|---|
| Tool schema generation | `@tool` decorator + docstrings/type hints | Lets the LLM know what tools exist and how to call them |
| Deterministic output | `temperature=0` | Reliable tool selection instead of creative variance |
| Tool binding | `llm.bind_tools(tools)` | Actually enables structured `tool_calls` output (vs. plain `.bind()`) |
| Loop termination | `if not tool_calls: return` | The model itself signals "I'm done" by not requesting a tool |
| Message bookkeeping | `tool_call_id` matching | Required so the model can correlate results with its own requests |
| Safety limits | `MAX_ITERATIONS` | Prevents runaway loops from a misbehaving or confused model |
| Prompt engineering | `SystemMessage` strict rules | Compensates for smaller local models being less reliable about tool discipline |
