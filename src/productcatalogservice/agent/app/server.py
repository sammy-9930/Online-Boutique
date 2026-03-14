"""
server.py
---------
gRPC server for the ProductCatalog agent.

gRPC handlers return and receive protobuf objects
internal tools work with python dictionaries 
we convert python dict-> protobuf messages before sending the response 

Exposes the EXACT same interface as the original Go microservice:

  rpc ListProducts(Empty)                   returns (ListProductsResponse)
  rpc GetProduct(GetProductRequest)         returns (Product)
  rpc SearchProducts(SearchProductsRequest) returns (SearchProductsResponse)

"""

import argparse
import logging
import os
import signal
import sys
import grpc
from concurrent import futures


from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from genproto import demo_pb2, demo_pb2_grpc
from tools import list_products, get_product, search_products, set_reload_flag

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("productcatalogservice-agent")


# ---------------------------------------------------------------------------
# Proto ↔ dict helpers
# (kept here — they have exactly one caller: the gRPC servicer below)
# ---------------------------------------------------------------------------

def _money_to_proto(price: dict) -> demo_pb2.Money:
    return demo_pb2.Money(
        currency_code=price.get("currencyCode", "USD"),
        units=int(price.get("units", 0)),
        nanos=int(price.get("nanos", 0)),
    )


def _dict_to_product(p: dict) -> demo_pb2.Product:
    return demo_pb2.Product(
        id=p["id"],
        name=p["name"],
        description=p["description"],
        picture=p.get("picture", ""),
        price_usd=_money_to_proto(p.get("priceUsd", {})),
        categories=p.get("categories", []),
    )


# ---------------------------------------------------------------------------
# gRPC Servicer
# ---------------------------------------------------------------------------

class ProductCatalogServicer(demo_pb2_grpc.ProductCatalogServiceServicer):
    """
    Implements the 3 ProductCatalogService RPCs.

    Each handler calls the corresponding tool function directly.
    The agent's LangGraph reasoning layer (agent.py / run_agent) sits
    on top and can be called from these handlers if you want LLM-mediated
    responses — but for a fair latency comparison with the original service,
    the default path calls tools directly without LLM overhead.
    """

    def ListProducts(self, request, context):
        """rpc ListProducts(Empty) returns (ListProductsResponse)"""
        logger.info("ListProducts called")
        try:
            result = list_products()
            products = [_dict_to_product(p) for p in result["products"]]
            return demo_pb2.ListProductsResponse(products=products)
        except Exception as e:
            logger.error("ListProducts error: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return demo_pb2.ListProductsResponse()

    def GetProduct(self, request, context):
        """rpc GetProduct(GetProductRequest) returns (Product)"""
        logger.info("GetProduct called: id=%s", request.id)
        try:
            p = get_product(request.id)
            return _dict_to_product(p)
        except ValueError as e:
            logger.warning("GetProduct not found: %s", e)
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return demo_pb2.Product()
        except Exception as e:
            logger.error("GetProduct error: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return demo_pb2.Product()

    def SearchProducts(self, request, context):
        """rpc SearchProducts(SearchProductsRequest) returns (SearchProductsResponse)"""
        logger.info("SearchProducts called: query=%r", request.query)
        try:
            result = search_products(request.query)
            results = [_dict_to_product(p) for p in result["results"]]
            return demo_pb2.SearchProductsResponse(results=results)
        except ValueError as e:
            logger.warning("SearchProducts invalid query: %s", e)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return demo_pb2.SearchProductsResponse()
        except Exception as e:
            logger.error("SearchProducts error: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return demo_pb2.SearchProductsResponse()


# ---------------------------------------------------------------------------
# Signal handlers (mirrors Go's SIGUSR1 / SIGUSR2 behaviour)
# ---------------------------------------------------------------------------

def _handle_usr1(signum, frame):
    """
    SIGUSR1: enable per-request catalog reload (the intentional latency bug).
    Mirrors the Go service behaviour documented in the repo README.
    """
    logger.warning("SIGUSR1 received: enabling per-request catalog reload (latency bug active)")
    set_reload_flag(True)


def _handle_usr2(signum, frame):
    """
    SIGUSR2: disable per-request catalog reload, restore normal caching.
    """
    logger.info("SIGUSR2 received: disabling per-request catalog reload (latency bug removed)")
    set_reload_flag(False)


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def serve(port: int) -> None:
    # Register signal handlers
    signal.signal(signal.SIGUSR1, _handle_usr1)
    signal.signal(signal.SIGUSR2, _handle_usr2)

    # Build gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )

    # Register ProductCatalogService
    demo_pb2_grpc.add_ProductCatalogServiceServicer_to_server(
        ProductCatalogServicer(), server
    )

    # Register health check service
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set(
        "hipstershop.ProductCatalogService",
        health_pb2.HealthCheckResponse.SERVING,
    )
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)

    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("ProductCatalog agent gRPC server listening on port %d", port)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(grace=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GRPC_PORT", 3550)),
        help="gRPC listen port (default: 3550)",
    )
    args = parser.parse_args()
    serve(args.port)