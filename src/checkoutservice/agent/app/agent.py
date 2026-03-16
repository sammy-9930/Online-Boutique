"""
agent.py
--------
LangGraph-based CheckoutService agent.
 
The LLM in reasoning_node decides:
  - Which downstream service to call next
  - How to wire the outputs of one tool into the inputs of the next
  - When all steps are complete and the final OrderResult can be returned
  - How to handle errors returned by any tool
 
PlaceOrder orchestration:
  1. get_user_cart
  2. For each cart item: get_product  →  convert_currency
  3. get_shipping_quote
  4. charge_card   (total = sum of converted item prices + shipping)
  5. ship_order
  6. send_order_confirmation
  7. empty_cart
  8. Return OrderResult JSON
 
Token usage and reasoning latency are tracked per request for experiments.
"""

import json
import logging
import os
import time
from typing import Annotated, Any, TypedDict

from langchain_ollama import ChatOllama
 
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool as lc_tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
 
from tools import (
    get_user_cart,
    get_product,
    convert_currency,
    get_shipping_quote,
    charge_card,
    ship_order,
    send_order_confirmation,
    empty_cart,
)
 
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AgentState — fresh per request
# ---------------------------------------------------------------------------
 
class AgentState(TypedDict):
    messages:     Annotated[list[BaseMessage], add_messages]
    token_usage:  dict[str, int]   # cumulative input/output token counts
    reasoning_ms: float            # cumulative LLM time in ms
    iterations:   int              # number of reasoning rounds

# ---------------------------------------------------------------------------
# LangChain tool wrappers
# ---------------------------------------------------------------------------
 
@lc_tool
def tool_get_user_cart(user_id: str) -> str:
    """
    Fetch all cart items for a user from CartService.
 
    Args:
        user_id: the user's ID string.
 
    Returns a JSON object: {"items": [{"product_id": str, "quantity": int}, ...]}
    """
    try:
        return json.dumps(get_user_cart(user_id))
    except Exception as e:
        return json.dumps({"error": str(e)})


@lc_tool
def tool_get_product(product_id: str) -> str:
    """
    Get product name and price from ProductCatalogService.
 
    Args:
        product_id: the product's ID string.
 
    Returns a JSON object:
        {"id": str, "name": str, "price_usd": {"currency_code": str, "units": int, "nanos": int}}
    """
    try:
        return json.dumps(get_product(product_id))
    except Exception as e:
        return json.dumps({"error": str(e)})
    

@lc_tool
def tool_convert_currency(from_currency_code: str, from_units: int, from_nanos: int,
                          to_currency: str) -> str:
    """
    Convert a Money amount to the user's preferred currency via CurrencyService.
 
    Args:
        from_currency_code: source currency, e.g. "USD"
        from_units:         integer part of the source amount
        from_nanos:         fractional part (billionths) of the source amount
        to_currency:        target currency code, e.g. "EUR"
 
    Returns a JSON object: {"currency_code": str, "units": int, "nanos": int}
    """
    try:
        return json.dumps(convert_currency(from_currency_code, from_units, from_nanos, to_currency))
    except Exception as e:
        return json.dumps({"error": str(e)})
    

@lc_tool
def tool_get_shipping_quote(address: dict, items: list) -> str:
    """
    Get a shipping cost estimate from ShippingService.
 
    Args:
        address: {"street_address": str, "city": str, "state": str, "country": str, "zip_code": int}
        items:   [{"product_id": str, "quantity": int}, ...]
 
    Returns a JSON object: {"currency_code": str, "units": int, "nanos": int}
    """
    try:
        return json.dumps(get_shipping_quote(address, items))
    except Exception as e:
        return json.dumps({"error": str(e)})
    

@lc_tool
def tool_charge_card(currency_code: str, units: int, nanos: int,
                     credit_card_number: str, credit_card_cvv: int,
                     credit_card_expiry_year: int, credit_card_expiry_month: int) -> str:
    """
    Charge the customer's credit card via PaymentService.
 
    Args:
        currency_code:            currency of the total charge, e.g. "USD"
        units:                    integer part of the total amount
        nanos:                    fractional part (billionths) of the total amount
        credit_card_number:       card number string
        credit_card_cvv:          CVV int
        credit_card_expiry_year:  4-digit year int
        credit_card_expiry_month: 1-12 int
 
    Returns a JSON object: {"transaction_id": str}
    """
    try:
        return json.dumps(charge_card(
            currency_code, units, nanos,
            credit_card_number, credit_card_cvv,
            credit_card_expiry_year, credit_card_expiry_month,
        ))
    except Exception as e:
        return json.dumps({"error": str(e)})
    

@lc_tool
def tool_ship_order(address: dict, items: list) -> str:
    """
    Dispatch the shipment via ShippingService.
 
    Args:
        address: {"street_address": str, "city": str, "state": str, "country": str, "zip_code": int}
        items:   [{"product_id": str, "quantity": int}, ...]
 
    Returns a JSON object: {"tracking_id": str}
    """
    try:
        return json.dumps(ship_order(address, items))
    except Exception as e:
        return json.dumps({"error": str(e)})
    

@lc_tool
def tool_send_order_confirmation(email: str, order: dict) -> str:
    """
    Send an order-confirmation email via EmailService.
 
    Args:
        email: customer e-mail address string
        order: {
            "order_id": str,
            "shipping_tracking_id": str,
            "shipping_cost": {"currency_code": str, "units": int, "nanos": int},
            "shipping_address": {"street_address": str, "city": str, "state": str,
                                 "country": str, "zip_code": int},
            "items": [
                {"item": {"product_id": str, "quantity": int},
                 "cost": {"currency_code": str, "units": int, "nanos": int}}, ...
            ]
        }
 
    Returns a JSON object: {"success": true}
    """
    try:
        return json.dumps(send_order_confirmation(email, order))
    except Exception as e:
        return json.dumps({"error": str(e)})
    

@lc_tool
def tool_empty_cart(user_id: str) -> str:
    """
    Clear the user's cart after successful checkout via CartService.
 
    Args:
        user_id: the user's ID string.
 
    Returns a JSON object: {"success": true}
    """
    try:
        return json.dumps(empty_cart(user_id))
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
LC_TOOLS = [
    tool_get_user_cart,
    tool_get_product,
    tool_convert_currency,
    tool_get_shipping_quote,
    tool_charge_card,
    tool_ship_order,
    tool_send_order_confirmation,
    tool_empty_cart,
]
TOOL_MAP = {t.name: t for t in LC_TOOLS}


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------
 
def _build_llm():
    model    = os.environ.get("OLLAMA_MODEL",    "llama3.1")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
 
    logger.info("Using Ollama model: %s @ %s", model, base_url)
    return ChatOllama(model=model, base_url=base_url).bind_tools(LC_TOOLS)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
 
SYSTEM_PROMPT = """You are the CheckoutService agent for an e-commerce store.
 
Your only job is to execute PlaceOrder by calling the provided tools in the
correct sequence. You must never fabricate values — always use tool results.
 
=== TOOL SEQUENCE ===
 
Step 1 — tool_get_user_cart(user_id)
  → gives you the list of cart items
 
Step 2 — For EACH cart item:
  a) tool_get_product(product_id)         → gives price_usd
  b) tool_convert_currency(from_currency_code, from_units, from_nanos, to_currency)
     Convert the product's price_usd to the user's preferred currency (user_currency).
     Collect all converted item costs; you will need them later.
 
Step 3 — tool_get_shipping_quote(address, items)
  → gives the shipping cost in USD; then convert it with tool_convert_currency too.
 
Step 4 — tool_charge_card(currency_code, units, nanos, credit_card_number,
                           credit_card_cvv, credit_card_expiry_year,
                           credit_card_expiry_month)
  Charge the TOTAL amount = sum of all converted item costs + converted shipping cost.
  All values must be in the user's preferred currency.
 
Step 5 — tool_ship_order(address, items)
  → gives you the tracking_id
 
Step 6 — tool_send_order_confirmation(email, order)
  Build the full order dict from all previous results and send it.
 
Step 7 — tool_empty_cart(user_id)
  → clears the cart
 
Step 8 — Return the final result as a single JSON object:
  {
    "order_id":              "<uuid>",
    "shipping_tracking_id":  "<from step 5>",
    "shipping_cost":         {"currency_code": ..., "units": ..., "nanos": ...},
    "shipping_address":      {the address dict},
    "items": [
      {
        "item": {"product_id": ..., "quantity": ...},
        "cost": {"currency_code": ..., "units": ..., "nanos": ...}
      }, ...
    ]
  }
 
=== RULES ===
1. Use the EXACT argument names shown in each tool description.
2. Never guess or fabricate values — always derive them from tool results.
3. Call tools one at a time; wait for the result before proceeding.
4. Once all steps are done, output ONLY the final JSON object, no extra text.
5. If any tool returns {"error": "..."}, stop and return {"error": "<message>"}.
"""

# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
 
def reasoning_node(state: AgentState) -> AgentState:
    """
    LLM reasoning node.
    The LLM sees the full message history and decides:
      - which tool to call next, OR
      - that it has enough data and produces the final OrderResult JSON.
    """
    llm = _build_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
 
    t0 = time.perf_counter()
    response: AIMessage = llm.invoke(messages)
    elapsed_ms = (time.perf_counter() - t0) * 1000
 
    token_usage = state.get("token_usage", {"input_tokens": 0, "output_tokens": 0}).copy()
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        token_usage["input_tokens"]  += response.usage_metadata.get("input_tokens",  0)
        token_usage["output_tokens"] += response.usage_metadata.get("output_tokens", 0)
 
    iterations   = state.get("iterations",   0)   + 1
    reasoning_ms = state.get("reasoning_ms", 0.0) + elapsed_ms
 
    logger.info(
        "reasoning_node: iteration=%d latency=%.1fms tool_calls=%d",
        iterations, elapsed_ms,
        len(response.tool_calls) if response.tool_calls else 0,
    )
 
    return {
        "messages":     [response],
        "token_usage":  token_usage,
        "reasoning_ms": reasoning_ms,
        "iterations":   iterations,
    }
 
 
def tool_call_node(state: AgentState) -> AgentState:
    """
    Tool execution node.
    Runs every tool call the LLM requested and returns ToolMessage results
    back into the message history so the LLM can reason further.
    """
    last: AIMessage = state["messages"][-1]
    tool_messages   = []
 
    for tc in last.tool_calls:
        name    = tc["name"]
        args    = tc["args"]
        call_id = tc["id"]
 
        logger.info("tool_call_node: calling %s args=%s", name, args)
 
        if name in TOOL_MAP:
            try:
                result = TOOL_MAP[name].invoke(args)
            except Exception as e:
                result = json.dumps({"error": str(e)})
        else:
            result = json.dumps({"error": f"unknown tool: {name}"})
 
        tool_messages.append(ToolMessage(content=result, tool_call_id=call_id))
 
    return {"messages": tool_messages}
 
 
def _should_continue(state: AgentState) -> str:
    """Edge: route to tool_call if the LLM wants tools, otherwise END."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_call"
    return END
 
 
# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
 
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("reasoning", reasoning_node)
    g.add_node("tool_call", tool_call_node)
    g.set_entry_point("reasoning")
    g.add_conditional_edges(
        "reasoning", _should_continue, {"tool_call": "tool_call", END: END}
    )
    g.add_edge("tool_call", "reasoning")
    return g.compile()
 
 
# ---------------------------------------------------------------------------
# Public API  (called by server.py)
# ---------------------------------------------------------------------------
 
def run_agent(request: str) -> dict[str, Any]:
    """
    Run the agent for a single PlaceOrder request string.
 
    Args:
        request: structured natural-language prompt describing the order,
                 e.g. produced by server.py from the incoming proto message.
 
    Returns:
        {
          "answer":       str,   # final LLM response (JSON string of OrderResult)
          "token_usage":  dict,  # {"input_tokens": N, "output_tokens": N}
          "reasoning_ms": float, # total LLM time in milliseconds
          "iterations":   int,   # number of reasoning rounds
        }
    """
    graph = build_graph()
 
    initial: AgentState = {
        "messages":     [HumanMessage(content=request)],
        "token_usage":  {"input_tokens": 0, "output_tokens": 0},
        "reasoning_ms": 0.0,
        "iterations":   0,
    }
 
    final = graph.invoke(initial)
 
    # Prefer the last AIMessage with content; fall back to last ToolMessage
    answer = ""
    for msg in reversed(final["messages"]):
        if isinstance(msg, ToolMessage) and msg.content:
            answer = msg.content
            break
    if not answer:
        for msg in reversed(final["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
                break
 
    return {
        "answer":       answer,
        "token_usage":  final.get("token_usage",  {}),
        "reasoning_ms": round(final.get("reasoning_ms", 0.0), 3),
        "iterations":   final.get("iterations",   0),
    }



