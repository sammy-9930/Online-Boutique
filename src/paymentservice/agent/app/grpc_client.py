import grpc
from fastapi import HTTPException
from app.config import PAYMENT_HOST, PAYMENT_PORT
from app.clients import demo_pb2
from app.clients import demo_pb2_grpc


class PaymentGrpcClient:
    def __init__(self):
        target = f"{PAYMENT_HOST}:{PAYMENT_PORT}"
        self.channel = grpc.insecure_channel(target)
        self.stub = demo_pb2_grpc.PaymentServiceStub(self.channel)

    def _charge_to_dict(self, response):
        return {
            "transaction_id": response.transaction_id
        }

    def charge(self, currency_code: str, units: int, nanos: int,
               credit_card_number: str, credit_card_cvv: int,
               credit_card_expiration_year: int, credit_card_expiration_month: int):
        try:
            request = demo_pb2.ChargeRequest(
                amount=demo_pb2.Money(
                    currency_code=currency_code,
                    units=units,
                    nanos=nanos
                ),
                credit_card=demo_pb2.CreditCardInfo(
                    credit_card_number=credit_card_number,
                    credit_card_cvv=credit_card_cvv,
                    credit_card_expiration_year=credit_card_expiration_year,
                    credit_card_expiration_month=credit_card_expiration_month
                )
            )
            response = self.stub.Charge(request, timeout=5)
            return self._charge_to_dict(response)

        except grpc.RpcError as e:
            msg = e.details() or "payment service error"

            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise HTTPException(status_code=503, detail="payment gRPC service unavailable")
            elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise HTTPException(status_code=504, detail="payment request timed out")
            elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                raise HTTPException(status_code=400, detail=msg)
            else:
                raise HTTPException(status_code=400, detail=msg)