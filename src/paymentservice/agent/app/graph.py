from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from app.agent import PaymentAgent
from app.llm.llama import get_llama_llm


class PaymentState(TypedDict):
    query: str
    currency_code: Optional[str]
    units: Optional[int]
    nanos: Optional[int]
    credit_card_number: Optional[str]
    credit_card_cvv: Optional[int]
    credit_card_expiration_year: Optional[int]
    credit_card_expiration_month: Optional[int]
    route: str
    result: dict


agent = PaymentAgent()
llm = get_llama_llm()


def classify_request(state: PaymentState):
    prompt = f"""
You are a router for a payment service.
Classify the user request into exactly one label:
- charge

Rules:
- If the user wants to make a payment or charge a credit card, return charge

User query: {state['query']}

Return only one label.
""".strip()

    response = llm.invoke(prompt)
    label = response.content.strip().lower()

    if "charge" in label:
        state["route"] = "charge"
    else:
        state["route"] = "charge"

    return state


def run_agent(state: PaymentState):
    state["result"] = agent.run(
        query=state["query"],
        currency_code=state.get("currency_code", "USD"),
        units=state.get("units", 0),
        nanos=state.get("nanos", 0),
        credit_card_number=state.get("credit_card_number", ""),
        credit_card_cvv=state.get("credit_card_cvv", 0),
        credit_card_expiration_year=state.get("credit_card_expiration_year", 0),
        credit_card_expiration_month=state.get("credit_card_expiration_month", 0)
    )
    return state


def build_graph():
    graph = StateGraph(PaymentState)
    graph.add_node("classify_request", classify_request)
    graph.add_node("run_agent", run_agent)

    graph.set_entry_point("classify_request")
    graph.add_edge("classify_request", "run_agent")
    graph.add_edge("run_agent", END)

    return graph.compile()