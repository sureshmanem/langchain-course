# Load environment variables (.env) before any LangSmith imports,
# since langsmith reads LANGSMITH_API_KEY/LANGSMITH_TRACING at import time.
from dotenv import load_dotenv

load_dotenv()

# Difference 1: Use the raw Ollama client instead of LangChain's init_chat_model
import ollama  # official Ollama python client; talks to the local Ollama server
from langsmith import traceable  # sends a trace of this function's execution to LangSmith

MAX_ITERATIONS = 10  # hard cap on the tool-call loop so a misbehaving model can't loop forever
MODEL = "qwen3:1.7b"  # local Ollama model (must support the "tools" capability)


# --- Tools (plain Python functions, traced for LangSmith) ---


@traceable(run_type="tool")  # without @tool, we trace each tool manually so it shows up in LangSmith
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    # Logged so we can see in the console when the model actually invokes this tool
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)  # unknown products default to 0 rather than raising


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(
        f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)  # unknown tier -> no discount
    return round(price * (1 - discount / 100), 2)


# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
tools_for_llm = [
    {
        "type": "function",  # OpenAI-style function-calling schema, which Ollama also accepts
        "function": {
            "name": "get_product_price",  # must match a key in tools_dict so we can dispatch the call
            "description": "Look up the price of a product in the catalog.",  # the model reads this to decide when to call it
            "parameters": {  # JSON Schema describing the arguments the model must produce
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],  # model must always supply this argument
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.


@traceable(name="Ollama Chat", run_type="llm")  # run_type="llm" makes LangSmith render it as a model call
def ollama_chat_traced(messages):
    # Single round-trip to the model; tool schemas are sent on every call (no bind_tools equivalent)
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# --- Agent Loop ---


@traceable(name="Ollama Agent Loop")  # parent trace; the LLM and tool traces nest under it
def run_agent(question: str):
    # Lookup table so we can call a python function by the name the model returns
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    # Difference 4: Messages are plain dicts with a "role" key instead of
    # SystemMessage/HumanMessage/ToolMessage objects. The history grows each
    # iteration as tool calls/results are appended, giving the model memory.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    # Core ReAct-style loop: ask the model -> if it wants a tool, run it and
    # feed the result back -> repeat until the model answers with no tool call.
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Difference 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message  # the assistant's reply: text content and/or tool_calls

        tool_calls = ai_message.tool_calls  # None/empty when the model answers in plain text

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments  # already parsed into a dict by the Ollama client

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            # Model hallucinated a tool name that isn't registered; fail loudly rather than guess
            raise ValueError(f"Tool '{tool_name}' not found")

        # Difference 7: Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)  # unpack the model's args as keyword arguments

        print(f"  [Tool Result] {observation}")

        # Append the assistant's tool-call message and the tool's result so
        # the next LLM call has full context of what was requested and returned.
        # Unlike ToolMessage, Ollama's tool message needs no tool_call_id.
        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    # Loop exhausted MAX_ITERATIONS without the model producing a final answer
    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello Ollama Agent (raw function calling)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
