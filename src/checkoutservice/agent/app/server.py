"""
server.py
---------
gRPC server that exposes exactly the same interface as the original Go
checkoutservice (port 5050):

  rpc PlaceOrder(PlaceOrderRequest) returns (PlaceOrderResponse)

This file is pure wire-protocol glue only:
  proto request  →  natural-language prompt  →  agent  →  parse response  →  proto response

No business logic lives here. All orchestration reasoning is in agent.py / tools.py.

Setup before running:
  1. Copy protos/demo.proto from the Online Boutique repo
  2. Generate stubs:
       python -m grpc_tools.protoc -I./protos \\
           --python_out=./genproto \\
           --grpc_python_out=./genproto \\
           protos/demo.proto
  3. pip install -r requirements.txt
  4. Set env vars for downstream services (see below) then:
       python server.py
     or in background:
       nohup python server.py > checkout_agent.log 2>&1 &

Environment variables:
  PORT                        gRPC listen port            (default: 5050)
  CART_SERVICE_ADDR           host:port                   (default: localhost:7070)
  PRODUCT_CATALOG_SERVICE_ADDR                            (default: localhost:3550)
  CURRENCY_SERVICE_ADDR                                   (default: localhost:7000)
  SHIPPING_SERVICE_ADDR                                   (default: localhost:50051)
  PAYMENT_SERVICE_ADDR                                    (default: localhost:50051)
  EMAIL_SERVICE_ADDR                                      (default: localhost:8080)
  OLLAMA_MODEL                Ollama model name           (default: llama3.1)
  OLLAMA_BASE_URL             Ollama endpoint             (default: http://localhost:11434)
"""

import json
import re
import logging
import os
import time
import uuid
import concurrent.futures

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from genproto import demo_pb2, demo_pb2_grpc
import agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("checkout_server")

PORT = int(os.getenv("PORT", "5050"))


# ---------------------------------------------------------------------------
# Proto <-> dict helpers
# ---------------------------------------------------------------------------

def _address_to_dict(addr) -> dict:
    return {
        "street_address": addr.street_address,
        "city":           addr.city,
        "state":          addr.state,
        "country":        addr.country,
        "zip_code":       addr.zip_code,
    }


def _credit_card_to_dict(cc) -> dict:
    return {
        "credit_card_number":       cc.credit_card_number,
        "credit_card_cvv":          cc.credit_card_cvv,
        "credit_card_expiry_year":  cc.credit_card_expiry_year,
        "credit_card_expiry_month": cc.credit_card_expiry_month,
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

    # Find the first {...} block (possibly nested) anywhere in the text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON from agent answer: %r", answer)
    return {}


def _dict_to_order_result(d: dict) -> demo_pb2.OrderResult:
    """Convert the agent's final JSON dict into a proto OrderResult."""
    sc = d.get("shipping_cost", {})
    sa = d.get("shipping_address", {})

    proto_items = []
    for oi in d.get("items", []):
        it   = oi.get("item", {})
        cost = oi.get("cost", {})
        proto_items.append(
            demo_pb2.OrderItem(
                item=demo_pb2.CartItem(
                    product_id=it.get("product_id", ""),
                    quantity=it.get("quantity", 0),
                ),
                cost=demo_pb2.Money(
                    currency_code=cost.get("currency_code", "USD"),
                    units=int(cost.get("units", 0)),
                    nanos=int(cost.get("nanos", 0)),
                ),
            )
        )

    return demo_pb2.OrderResult(
        order_id=d.get("order_id", str(uuid.uuid4())),
        shipping_tracking_id=d.get("shipping_tracking_id", "UNKNOWN"),
        shipping_cost=demo_pb2.Money(
            currency_code=sc.get("currency_code", "USD"),
            units=int(sc.get("units", 0)),
            nanos=int(sc.get("nanos", 0)),
        ),
        shipping_address=demo_pb2.Address(
            street_address=sa.get("street_address", ""),
            city=sa.get("city", ""),
            state=sa.get("state", ""),
            country=sa.get("country", ""),
            zip_code=int(sa.get("zip_code", 0)),
        ),
        items=proto_items,
    )


# ---------------------------------------------------------------------------
# CheckoutService implementation
# ---------------------------------------------------------------------------

class CheckoutServicer(demo_pb2_grpc.CheckoutServiceServicer):

    def PlaceOrder(self, request, context):
        """
        rpc PlaceOrder(PlaceOrderRequest) returns (PlaceOrderResponse)
        """
        logger.info(
            "[PlaceOrder] user_id=%s user_currency=%s email=%s",
            request.user_id, request.user_currency, request.email,
        )
        try:
            address     = _address_to_dict(request.address)
            credit_card = _credit_card_to_dict(request.credit_card)

            # Build a structured prompt for the agent.
            # The agent knows the full PlaceOrder sequence from its system prompt.
            prompt = (
                f"Place an order for the following customer:\n"
                f"  user_id:       {request.user_id}\n"
                f"  user_currency: {request.user_currency}\n"
                f"  email:         {request.email}\n"
                f"  address:       {json.dumps(address)}\n"
                f"  credit_card:   {json.dumps(credit_card)}\n\n"
                f"Follow the full PlaceOrder sequence described in your instructions "
                f"and return the final OrderResult as a JSON object."
            )

            result = agent.run_agent(prompt)
            logger.info(
                "[PlaceOrder] agent done: iterations=%d reasoning_ms=%.1f tokens=%s",
                result["iterations"], result["reasoning_ms"], result["token_usage"],
            )

            order_dict   = _parse_answer(result["answer"])

            # Surface agent errors as gRPC errors
            if "error" in order_dict:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(order_dict["error"])
                return demo_pb2.PlaceOrderResponse()

            order_result = _dict_to_order_result(order_dict)
            logger.info(
                "[PlaceOrder] returning order_id=%s tracking_id=%s",
                order_result.order_id, order_result.shipping_tracking_id,
            )
            return demo_pb2.PlaceOrderResponse(order=order_result)

        except Exception as e:
            logger.exception("[PlaceOrder] error: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return demo_pb2.PlaceOrderResponse()


# ---------------------------------------------------------------------------
# Health check (required by Online Boutique frontend)
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
    demo_pb2_grpc.add_CheckoutServiceServicer_to_server(CheckoutServicer(), server)
    health_pb2_grpc.add_HealthServicer_to_server(HealthServicer(), server)
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    logger.info("CheckoutService agent listening on port %d", PORT)
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(grace=5)


if __name__ == "__main__":
    serve()