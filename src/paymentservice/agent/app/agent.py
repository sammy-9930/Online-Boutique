from app.grpc_client import PaymentGrpcClient

client = PaymentGrpcClient()


class PaymentAgent:
    def run(self, query: str, currency_code: str = "USD", units: int = 0,
            nanos: int = 0, credit_card_number: str = "",
            credit_card_cvv: int = 0, credit_card_expiration_year: int = 0,
            credit_card_expiration_month: int = 0):

        return {
            "mode": "agent",
            "action": "charge",
            "data": client.charge(
                currency_code=currency_code,
                units=units,
                nanos=nanos,
                credit_card_number=credit_card_number,
                credit_card_cvv=credit_card_cvv,
                credit_card_expiration_year=credit_card_expiration_year,
                credit_card_expiration_month=credit_card_expiration_month
            )
        }