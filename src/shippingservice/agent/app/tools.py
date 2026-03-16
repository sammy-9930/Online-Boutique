"""
tools.py
--------
Deterministic tool functions for the ShippingService agent.
Exactly 2 tools — one per gRPC method in the original service:
 
  calculate_shipping_quote  →  GetQuote(GetQuoteRequest) returns (GetQuoteResponse)
  generate_tracking_id      →  ShipOrder(ShipOrderRequest) returns (ShipOrderResponse)
"""
 
import uuid
import random 
import time
import logging
from typing import Any
 
logger = logging.getLogger(__name__)

seeded = False 

def calculate_shipping_quote(items: list[dict]) -> dict[str, Any]:
    """
    Python equivalent of the Go shippingservice quote logic.
    Always returns $8.99, regardless of item count.
    """
    cost = 8.99
    units = int(cost)
    nanos = int(round((cost - units) * 1_000_000_000))

    logger.info(
        "calculate_shipping_quote: cost=%.2f units=%d nanos=%d",
        cost, units, nanos,
    )

    return {
        "currency_code": "USD",
        "units": units,
        "nanos": nanos,
    }


def _get_random_letter_code():
    # Go: 65 + rand.Intn(25)
    return chr(65 + random.randint(0, 24))

def _get_random_number(digits: int) -> str:
    s = ""
    for _ in range(digits):
        s += str(random.randint(0, 9))
    return s


def generate_tracking_id(address: dict) -> dict[str, Any]:
    """
    Python port of the Go CreateTrackingId() logic.
    """

    global seeded

    if not seeded:
        random.seed(time.time_ns())
        seeded = True

    base_address = f"{address.get('street_address','')}, {address.get('city','')}, {address.get('state','')}"

    tracking_id = "%s%s-%d%s-%d%s" % (
        _get_random_letter_code(),
        _get_random_letter_code(),
        len(base_address),
        _get_random_number(3),
        len(base_address) // 2,
        _get_random_number(7),
    )

    logger.info("generate_tracking_id: tracking_id=%s", tracking_id)

    return {"tracking_id": tracking_id}
    