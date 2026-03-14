"""
agent.py
--------
LangGraph-based ProductCatalog agent.

Graph structure (stateful per-request, not between requests):

    START
      │
      ▼
  [reasoning_node]  ──── has tool_calls? ──▶ [tool_call_node]
      │                                             │
      │ no tool_calls                               │ (always loops back)
      ▼                                             ▼
     END                                    [reasoning_node]

The LLM in reasoning_node decides:
  - Which tool to call (list_products / get_product / search_products)
  - When it has enough information to produce the final answer
  - How to handle errors returned by tools

Token usage and reasoning latency are tracked per request for experiments.
"""

import json
import logging
import os
import time
from typing import Annotated, Any, TypedDict

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

from tools import list_products, get_product, search_products

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentState — the graph's state object, fresh per request
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    token_usage: dict[str, int]   # cumulative input/output token counts
    reasoning_ms: float           # cumulative LLM time (ms)
    iterations: int               # number of reasoning rounds


# ---------------------------------------------------------------------------
# LangChain tool wrappers
# (these bind the deterministic tool functions into the LLM's tool schema)
# ---------------------------------------------------------------------------

@lc_tool
def tool_list_products() -> str:
    """
    List all products available in the catalog.
    Returns a JSON object with a 'products' array.
    """
    result = list_products()
    return json.dumps(result)


@lc_tool
def tool_get_product(product_id: str) -> str:
    """
    Get a single product by its ID string (e.g. 'OLJCESPC7Z').
    Returns the product as JSON, or an error message if not found.
    """
    try:
        result = get_product(product_id)
        return json.dumps(result)
    except ValueError as e:
        return json.dumps({"error": str(e)})


@lc_tool
def tool_search_products(query: str) -> str:
    """
    Search for products using a keyword query.
    Matches against product name, description, and categories (case-insensitive).
    Returns a JSON object with a 'results' array.
    """
    try:
        result = search_products(query)
        return json.dumps(result)
    except ValueError as e:
        return json.dumps({"error": str(e)})


LC_TOOLS = [tool_list_products, tool_get_product, tool_search_products]
TOOL_MAP = {t.name: t for t in LC_TOOLS}


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm():
    """
    Return an Ollama tool-bound LLM.
    """
    from langchain_ollama import ChatOllama

    model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    logger.info("Using Ollama model: %s @ %s", model, base_url)
    return ChatOllama(model=model, base_url=base_url).bind_tools(LC_TOOLS)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the ProductCatalogService agent for an e-commerce store.

You answer product queries strictly using the provided tools. Never invent product details.

Available tools:
- tool_list_products: list all products in the catalog
- tool_get_product: fetch one product by its exact ID string
- tool_search_products: search products by keyword

Rules:
1. Always call a tool to get data — never guess or fabricate.
2. Use the minimum number of tool calls needed.
3. If a product or query returns an error, report it clearly.
4. Return your final answer as concise JSON or plain text depending on the caller's need.
"""


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def reasoning_node(state: AgentState) -> AgentState:
    """
    LLM reasoning node.
    The LLM sees the full message history and decides:
      - which tool to call next, OR
      - that it has enough data and produces the final answer.
    """
    llm = _build_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    t0 = time.perf_counter()
    response: AIMessage = llm.invoke(messages)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Accumulate token usage if the LLM provides it
    token_usage = state.get("token_usage", {"input_tokens": 0, "output_tokens": 0}).copy()
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        token_usage["input_tokens"] += response.usage_metadata.get("input_tokens", 0)
        token_usage["output_tokens"] += response.usage_metadata.get("output_tokens", 0)

    iterations = state.get("iterations", 0) + 1
    reasoning_ms = state.get("reasoning_ms", 0.0) + elapsed_ms

    logger.info(
        "reasoning_node: iteration=%d latency=%.1fms tool_calls=%d",
        iterations, elapsed_ms,
        len(response.tool_calls) if response.tool_calls else 0,
    )

    return {
        "messages": [response],
        "token_usage": token_usage,
        "reasoning_ms": reasoning_ms,
        "iterations": iterations,
    }


def tool_call_node(state: AgentState) -> AgentState:
    """
    Tool execution node.
    Runs every tool call the LLM requested and returns ToolMessage results
    back into the message history so the LLM can reason further.
    """
    last: AIMessage = state["messages"][-1]
    tool_messages = []

    for tc in last.tool_calls:
        name = tc["name"]
        args = tc["args"]
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
    g.add_conditional_edges("reasoning", _should_continue, {"tool_call": "tool_call", END: END})
    g.add_edge("tool_call", "reasoning")
    return g.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(request: str) -> dict[str, Any]:
    """
    Run the agent for a single request string.

    Args:
        request: A natural-language or structured query, e.g.:
                 "List all products"
                 "Get product with id OLJCESPC7Z"
                 "Search for accessories"

    Returns:
        {
          "answer":       str,   # final LLM response
          "token_usage":  dict,  # {"input_tokens": N, "output_tokens": N}
          "reasoning_ms": float, # total LLM time in milliseconds
          "iterations":   int,   # number of reasoning rounds
        }
    """
    graph = build_graph()

    initial: AgentState = {
        "messages": [HumanMessage(content=request)],
        "token_usage": {"input_tokens": 0, "output_tokens": 0},
        "reasoning_ms": 0.0,
        "iterations": 0,
    }

    final = graph.invoke(initial)

    # Extract final answer from the last AIMessage with content
    answer = ""
    for msg in reversed(final["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            answer = msg.content
            break

    return {
        "answer": answer,
        "token_usage": final.get("token_usage", {}),
        "reasoning_ms": round(final.get("reasoning_ms", 0.0), 3),
        "iterations": final.get("iterations", 0),
    }