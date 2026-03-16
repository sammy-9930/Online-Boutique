"""
agent.py
--------
LangGraph-based PaymentService agent.
 
Graph structure (stateful per-request, not between requests):

The LLM in reasoning_node decides:
  - To call tool_charge_credit_card with the correct arguments
  - How to handle validation errors returned by the tool
  - When it has the transaction_id and can produce the final answer
 
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
 
from tools import charge_credit_card
 
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
# LangChain tool wrapper
# ---------------------------------------------------------------------------
 
@lc_tool
def tool_charge_credit_card(
    currency_code: str,
    units: int,
    nanos: int,
    credit_card_number: str,
    credit_card_cvv: int,
    credit_card_expiry_year: int,
    credit_card_expiry_month: int,
) -> str:
    """
    Charge a credit card and return a transaction ID.
 
    Validates the card number (Luhn algorithm), card type (Visa / MasterCard / AmEx),
    and expiry date before issuing a mock transaction.
 
    Args:
        currency_code:            currency of the charge, e.g. "USD"
        units:                    integer part of the charge amount
        nanos:                    fractional part (billionths) of the charge amount
        credit_card_number:       card number string
        credit_card_cvv:          3- or 4-digit CVV integer
        credit_card_expiry_year:  4-digit expiry year integer
        credit_card_expiry_month: expiry month integer (1-12)
 
    Returns a JSON object: {"transaction_id": str}
    On validation failure returns: {"error": str}
    """
    try:
        result = charge_credit_card(
            currency_code=currency_code,
            units=units,
            nanos=nanos,
            credit_card_number=credit_card_number,
            credit_card_cvv=credit_card_cvv,
            credit_card_expiry_year=credit_card_expiry_year,
            credit_card_expiry_month=credit_card_expiry_month,
        )
        return json.dumps(result)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {e}"})
 
 
LC_TOOLS = [tool_charge_credit_card]
TOOL_MAP  = {t.name: t for t in LC_TOOLS}

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
 
SYSTEM_PROMPT = """You are the PaymentService agent for an e-commerce store.
 
You handle exactly one operation using the provided tool:
 
  tool_charge_credit_card(
      currency_code, units, nanos,
      credit_card_number, credit_card_cvv,
      credit_card_expiry_year, credit_card_expiry_month
  )
 
This tool:
  - Validates the card number using the Luhn algorithm
  - Validates the card type (Visa, MasterCard, or AmEx only)
  - Validates the card is not expired
  - Returns {"transaction_id": "<uuid>"} on success
  - Returns {"error": "<reason>"} on validation failure
 
Rules:
1. Always use the EXACT argument names shown above — do not rename them.
2. Always call the tool — never fabricate a transaction_id.
3. Use exactly one tool call per request.
4. Once you have the tool result, return it immediately as a JSON object with no extra text.
5. If the tool returns {"error": "..."}, return that error JSON directly.
"""

# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
 
def reasoning_node(state: AgentState) -> AgentState:
    """
    LLM reasoning node.
    The LLM sees the full message history and decides:
      - to call tool_charge_credit_card with the correct arguments, OR
      - that it already has the result and produces the final answer.
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
    Run the agent for a single Charge request string.
 
    Args:
        request: structured prompt describing the charge, produced by server.py.
 
    Returns:
        {
          "answer":       str,   # final LLM response (JSON string with transaction_id)
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
 
