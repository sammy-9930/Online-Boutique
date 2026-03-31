from fastapi import APIRouter
from app.schemas import PaymentRequest
from app.graph import build_graph

router = APIRouter()
graph = build_graph()


@router.get("/health")
def health():
    return {"status": "ok", "service": "paymentagent"}


@router.post("/payment/charge")
def charge(request: PaymentRequest):
    result = graph.invoke({
        "query": request.query,
        "currency_code": request.currency_code,
        "units": request.units,
        "nanos": request.nanos,
        "credit_card_number": request.credit_card_number,
        "credit_card_cvv": request.credit_card_cvv,
        "credit_card_expiration_year": request.credit_card_expiration_year,
        "credit_card_expiration_month": request.credit_card_expiration_month,
        "result": {}
    })
    return result["result"]