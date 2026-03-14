"""
server.py
---------
gRPC server that exposes exactly the same interface as the original Go
shippingservice (port 50051):

  rpc GetQuote(GetQuoteRequest)   returns (GetQuoteResponse)
  rpc ShipOrder(ShipOrderRequest) returns (ShipOrderResponse)

This file is pure wire-protocol glue only:
  proto request -> natural-language prompt -> agent -> parse response -> proto response

No business logic lives here. All reasoning is in agent.py / tools.py.

Setup before running:
  1. Copy protos/demo.proto from the Online Boutique repo
  2. Generate stubs:
       python -m grpc_tools.protoc -I./protos \
           --python_out=./genproto \
           --grpc_python_out=./genproto \
           protos/demo.proto
  3. pip install -r requirements.txt
  4. python server.py
     or in background: nohup python server.py > shipping_agent.log 2>&1 &
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
logger = logging.getLogger("shipping_server")

PORT = int(os.getenv("PORT", "50051"))


# ---------------------------------------------------------------------------
# Proto <-> dict helpers (trivial, kept inline)
# ---------------------------------------------------------------------------

def _address_to_dict(addr) -> dict:
    return {
        "street_address": addr.street_address,
        "city":           addr.city,
        "state":          addr.state,
        "country":        addr.country,
        "zip_code":       addr.zip_code,
    }


def _items_to_list(items) -> list[dict]:
    return [{"product_id": i.product_id, "quantity": i.quantity} for i in items]


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
# ShippingService implementation
# ---------------------------------------------------------------------------

class ShippingServicer(demo_pb2_grpc.ShippingServiceServicer):

    def GetQuote(self, request, context):
        """
        rpc GetQuote(GetQuoteRequest) returns (GetQuoteResponse)
        """
        logger.info("[GetQuote] received")
        try:
            address = _address_to_dict(request.address)
            items   = _items_to_list(request.items)

            prompt = (
                f"Call tool_calculate_shipping_quote with argument "
                f"items={json.dumps(items)}. "
                f"Return the result as a JSON object with keys: currency_code, units, nanos."
            )

            result_dict = _parse_answer(agent.run_agent(prompt)["answer"])

            response = demo_pb2.GetQuoteResponse(
                cost_usd=demo_pb2.Money(
                    currency_code=result_dict.get("currency_code", "USD"),
                    units=int(result_dict.get("units", 0)),
                    nanos=int(result_dict.get("nanos", 0)),
                )
            )
            logger.info("[GetQuote] returning units=%d nanos=%d",
                        response.cost_usd.units, response.cost_usd.nanos)
            return response

        except Exception as e:
            logger.exception("[GetQuote] error: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return demo_pb2.GetQuoteResponse()

    def ShipOrder(self, request, context):
        """
        rpc ShipOrder(ShipOrderRequest) returns (ShipOrderResponse)
        """
        logger.info("[ShipOrder] received")
        try:
            address = _address_to_dict(request.address)
            items   = _items_to_list(request.items)

            prompt = (
                f"Call tool_generate_tracking_id with argument "
                f"address={json.dumps(address)}. "
                f"Return the result as a JSON object with key: tracking_id."
            )

            result_dict = _parse_answer(agent.run_agent(prompt)["answer"])

            response = demo_pb2.ShipOrderResponse(
                tracking_id=result_dict.get("tracking_id", "UNKNOWN")
            )
            logger.info("[ShipOrder] returning tracking_id=%s", response.tracking_id)
            return response

        except Exception as e:
            logger.exception("[ShipOrder] error: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return demo_pb2.ShipOrderResponse()


# ---------------------------------------------------------------------------
# Health check (required by Online Boutique frontend / checkoutservice)
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
    demo_pb2_grpc.add_ShippingServiceServicer_to_server(ShippingServicer(), server)
    health_pb2_grpc.add_HealthServicer_to_server(HealthServicer(), server)
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    logger.info("ShippingService agent listening on port %d", PORT)
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(grace=5)


if __name__ == "__main__":
    serve()