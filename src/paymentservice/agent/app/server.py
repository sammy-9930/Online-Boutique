"""
server.py
---------
gRPC server that exposes exactly the same interface as the original Node.js
paymentservice (port 50051):

  rpc Charge(ChargeRequest) returns (ChargeResponse)

This file is pure wire-protocol glue only:
  proto request  →  natural-language prompt  →  agent  →  parse response  →  proto response

No business logic lives here. All validation and charge logic is in tools.py.
All reasoning about which tool to call lives in agent.py.
"""

import json
import re
import logging
import os
import time
import concurrent.futures

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from genproto import demo_pb2, demo_pb2_grpc
import agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("payment_server")

PORT = int(os.getenv("PORT", "50051"))


# ---------------------------------------------------------------------------
# Proto <-> dict helpers
# ---------------------------------------------------------------------------

def _credit_card_to_dict(cc) -> dict:
    return {
        "credit_card_number":       cc.credit_card_number,
        "credit_card_cvv":          cc.credit_card_cvv,
        "credit_card_expiry_year":  cc.credit_card_expiration_year,
        "credit_card_expiry_month": cc.credit_card_expiration_month,
    }


def _money_to_dict(money) -> dict:
    return {
        "currency_code": money.currency_code,
        "units":         money.units,
        "nanos":         money.nanos,
    }


def _parse_answer(answer: str) -> dict:
    """
    Extract the first JSON object from the agent's final answer string.
    Handles markdown fences, leading prose, and bare JSON.
    """
    text = answer.strip()

    # Strip markdown fences
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first {...} block anywhere in the text
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON from agent answer: %r", answer)
    return {}


# ---------------------------------------------------------------------------
# PaymentService implementation
# ---------------------------------------------------------------------------

class PaymentServicer(demo_pb2_grpc.PaymentServiceServicer):

    def Charge(self, request, context):
        """
        rpc Charge(ChargeRequest) returns (ChargeResponse)
        """
        logger.info(
            "[Charge] amount=%s %d card=****%s",
            request.amount.currency_code,
            request.amount.units,
            str(request.credit_card.credit_card_number)[-4:],
        )
        try:
            amount      = _money_to_dict(request.amount)
            credit_card = _credit_card_to_dict(request.credit_card)

            prompt = (
                f"Charge a credit card with the following details:\n"
                f"  amount:               {json.dumps(amount)}\n"
                f"  credit_card_number:   {credit_card['credit_card_number']}\n"
                f"  credit_card_cvv:      {credit_card['credit_card_cvv']}\n"
                f"  credit_card_expiry_year:  {credit_card['credit_card_expiry_year']}\n"
                f"  credit_card_expiry_month: {credit_card['credit_card_expiry_month']}\n\n"
                f"Call tool_charge_credit_card with these exact values and return "
                f"the result as a JSON object with key: transaction_id."
            )

            result      = agent.run_agent(prompt)
            logger.info(
                "[Charge] agent done: iterations=%d reasoning_ms=%.1f tokens=%s",
                result["iterations"], result["reasoning_ms"], result["token_usage"],
            )

            result_dict = _parse_answer(result["answer"])

            # Surface validation errors as gRPC INVALID_ARGUMENT
            if "error" in result_dict:
                logger.warning("[Charge] validation error: %s", result_dict["error"])
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(result_dict["error"])
                return demo_pb2.ChargeResponse()

            transaction_id = result_dict.get("transaction_id", "")
            logger.info("[Charge] returning transaction_id=%s", transaction_id)
            return demo_pb2.ChargeResponse(transaction_id=transaction_id)

        except Exception as e:
            logger.exception("[Charge] error: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return demo_pb2.ChargeResponse()


# ---------------------------------------------------------------------------
# Health check (required by Online Boutique checkoutservice)
# ---------------------------------------------------------------------------

class HealthServicer(health_pb2_grpc.HealthServicer):

    def Check(self, request, context):
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING
        )

    def Watch(self, request, context):
        yield health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING
        )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def serve():
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    demo_pb2_grpc.add_PaymentServiceServicer_to_server(PaymentServicer(), server)
    health_pb2_grpc.add_HealthServicer_to_server(HealthServicer(), server)
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    logger.info("PaymentService agent listening on port %d", PORT)
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(grace=5)


if __name__ == "__main__":
    serve()