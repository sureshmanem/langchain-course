# Load environment variables (.env) before any LangChain/LangSmith imports,
# since langsmith reads LANGSMITH_API_KEY/LANGSMITH_TRACING at import time.
import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import (
    init_chat_model,
)  # provider-agnostic LLM factory (e.g. "ollama:<model>")
from langchain.tools import (
    tool,
)  # decorator that turns a python function into a callable LLM tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)  # chat message types
from langsmith import (
    traceable,
)  # sends a trace of this function's execution to LangSmith

MAX_ITERATIONS = (
    10  # hard cap on the tool-call loop so a misbehaving model can't loop forever
)
# Selectable LLMs; the local Ollama model must support the "tools" capability
MODELS = {
    "ollama": "ollama:qwen3:1.7b",
    "openai": "openai:gpt-5",  # requires OPENAI_API_KEY
}
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")


# --- Tools (LangChain @tool decorator) ---


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog.
    Available products: laptop, headphones, keyboard."""
    # Logged so we can see in the console when the model actually invokes this tool
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)  # unknown products default to 0 rather than raising


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(
        f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)  # unknown tier -> no discount
    return round(price * (1 - discount / 100), 2)


# --- Agent Loop ---


@traceable(
    name="LangChain Agent Loop"
)  # publishes this run (and its LLM/tool calls) to LangSmith
def run_agent(question: str, provider: str = DEFAULT_PROVIDER):
    tools = [get_product_price, apply_discount]
    tools_dict = {
        t.name: t for t in tools
    }  # lookup table so we can invoke a tool by name later

    model = MODELS[provider]
    llm = init_chat_model(
        model, temperature=0
    )  # temperature=0 for deterministic tool choice
    llm_with_tools = llm.bind_tools(
        tools
    )  # attaches tool schemas so the model can emit tool_calls

    print(f"Model: {model}")
    print(f"Question: {question}")
    print("=" * 60)

    # Conversation history the model sees; grows each iteration as tool
    # calls/results are appended, giving the model memory of prior steps.
    messages = [
        SystemMessage(
            content=(
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
            )
        ),
        HumanMessage(content=question),
    ]

    # Core ReAct-style loop: ask the model -> if it wants a tool, run it and
    # feed the result back -> repeat until the model answers with no tool call.
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Single round-trip to the LLM; may return plain text or a tool_calls list
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get(
            "id"
        )  # links the ToolMessage result back to this specific call

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            # Model hallucinated a tool name that isn't registered; fail loudly rather than guess
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)  # actually run the python function

        print(f"  [Tool Result] {observation}")

        # Append the assistant's tool-call message and the tool's result so
        # the next LLM call has full context of what was requested and returned
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    # Loop exhausted MAX_ITERATIONS without the model producing a final answer
    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangChain tool-calling agent loop")
    parser.add_argument(
        "--provider",
        choices=MODELS.keys(),
        default=DEFAULT_PROVIDER,
        help="LLM to use (default: $LLM_PROVIDER or 'ollama')",
    )
    args = parser.parse_args()

    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent(
        "What is the price of a laptop after applying a silver discount?",
        provider=args.provider,
    )
