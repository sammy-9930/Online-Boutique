from pydantic import BaseModel
from typing import Optional, Any


class PaymentRequest(BaseModel):
    query: str
    currency_code: Optional[str] = "USD"
    units: Optional[int] = 0
    nanos: Optional[int] = 0
    credit_card_number: str
    credit_card_cvv: int
    credit_card_expiration_year: int
    credit_card_expiration_month: int


class PaymentResponse(BaseModel):
    mode: str
    action: str
    data: Any