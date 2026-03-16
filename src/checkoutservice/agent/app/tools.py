"""
tools.py
--------
Deterministic tool functions for the CheckoutService agent.
One tool per downstream gRPC call made by the original Go checkoutservice:
 
  get_user_cart              →  CartService.GetCart
  get_product                →  ProductCatalogService.GetProduct
  convert_currency           →  CurrencyService.Convert
  get_shipping_quote         →  ShippingService.GetQuote
  charge_card                →  PaymentService.Charge
  ship_order                 →  ShippingService.ShipOrder
  send_order_confirmation    →  EmailService.SendOrderConfirmation
  empty_cart                 →  CartService.EmptyCart
 
All functions accept / return plain Python dicts (JSON-serialisable).
The gRPC ↔ dict conversion lives in server.py.
No business-logic decisions are made here — the LLM in agent.py decides
which tool to call and in which order.
"""

import logging
import os
import uuid

import grpc

from genproto import demo_pb2
 
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# gRPC channel helpers  
# ---------------------------------------------------------------------------

def _channel(env_var: str, default: str):
    addr = os.environ.get(env_var, default)
    return grpc.insecure_channel(addr)

def _cart_stub():
    from genproto import demo_pb2_grpc
    return demo_pb2_grpc.CartServiceStub(_channel("CART_SERVICE_ADDR", "localhost:7070"))
 
def _catalog_stub():
    from genproto import demo_pb2_grpc
    return demo_pb2_grpc.ProductCatalogServiceStub(_channel("PRODUCT_CATALOG_SERVICE_ADDR", "localhost:3550"))

def _catalog_stub():
    from genproto import demo_pb2_grpc
    return demo_pb2_grpc.ProductCatalogServiceStub(_channel("PRODUCT_CATALOG_SERVICE_ADDR", "localhost:3550"))

def _currency_stub():
    from genproto import demo_pb2_grpc
    return demo_pb2_grpc.CurrencyServiceStub(_channel("CURRENCY_SERVICE_ADDR", "localhost:7000"))
 
 
def _shipping_stub():
    from genproto import demo_pb2_grpc
    return demo_pb2_grpc.ShippingServiceStub(_channel("SHIPPING_SERVICE_ADDR", "localhost:50051"))
 
 
def _payment_stub():
    from genproto import demo_pb2_grpc
    return demo_pb2_grpc.PaymentServiceStub(_channel("PAYMENT_SERVICE_ADDR", "localhost:50051"))
 
 
def _email_stub():
    from genproto import demo_pb2_grpc
    return demo_pb2_grpc.EmailServiceStub(_channel("EMAIL_SERVICE_ADDR", "localhost:8080"))

# ---------------------------------------------------------------------------
# Tool 1 — get_user_cart
# ---------------------------------------------------------------------------

def get_user_cart(user_id: str) -> dict:
    """
    Fetch all cart items for a user.
 
    Mirrors:  CartService.GetCart(GetCartRequest{UserId: userID})
 
    Returns:
        {"items": [{"product_id": str, "quantity": int}, ...]}
    """
    from genproto import demo_pb2
 
    logger.info("get_user_cart: user_id=%s", user_id)
    resp = _cart_stub().GetCart(demo_pb2.GetCartRequest(user_id=user_id))
    items = [{"product_id": i.product_id, "quantity": i.quantity} for i in resp.items]
    logger.info("get_user_cart: %d items returned", len(items))
    return {"items": items}

# ---------------------------------------------------------------------------
# Tool 2 — get_product
# ---------------------------------------------------------------------------

def get_product(product_id: str) -> dict:
    """
    Retrieve product details (name, price) by product ID.
 
    Mirrors:  ProductCatalogService.GetProduct(GetProductRequest{Id: id})
 
    Returns:
        {
          "id":          str,
          "name":        str,
          "price_usd":   {"currency_code": str, "units": int, "nanos": int}
        }
    """
 
    logger.info("get_product: product_id=%s", product_id)
    resp = _catalog_stub().GetProduct(demo_pb2.GetProductRequest(id=product_id))
    result = {
        "id":   resp.id,
        "name": resp.name,
        "price_usd": {
            "currency_code": resp.price_usd.currency_code,
            "units":         resp.price_usd.units,
            "nanos":         resp.price_usd.nanos,
        },
    }
    logger.info("get_product: name=%s price=%s", result["name"], result["price_usd"])
    return result

# ---------------------------------------------------------------------------
# Tool 3 — convert_currency
# ---------------------------------------------------------------------------
 
def convert_currency(from_currency_code: str, from_units: int, from_nanos: int,
                     to_currency: str) -> dict:
    """
    Convert a Money amount from one currency to another.
 
    Mirrors:  CurrencyService.Convert(CurrencyConversionRequest{From: money, ToCode: toCurrency})
 
    Args:
        from_currency_code: e.g. "USD"
        from_units:         integer part of the amount
        from_nanos:         fractional part (billionths)
        to_currency:        target currency code, e.g. "EUR"
 
    Returns:
        {"currency_code": str, "units": int, "nanos": int}
    """
 
    logger.info(
        "convert_currency: %s %d.%09d -> %s",
        from_currency_code, from_units, from_nanos, to_currency,
    )
    resp = _currency_stub().Convert(
        demo_pb2.CurrencyConversionRequest(
            from_=demo_pb2.Money(
                currency_code=from_currency_code,
                units=from_units,
                nanos=from_nanos,
            ),
            to_code=to_currency,
        )
    )
    result = {
        "currency_code": resp.currency_code,
        "units":         resp.units,
        "nanos":         resp.nanos,
    }
    logger.info("convert_currency: result=%s", result)
    return result

# ---------------------------------------------------------------------------
# Tool 4 — get_shipping_quote
# ---------------------------------------------------------------------------
 
def get_shipping_quote(address: dict, items: list[dict]) -> dict:
    """
    Get a shipping cost estimate from ShippingService.
 
    Mirrors:  ShippingService.GetQuote(GetQuoteRequest{Address: addr, Items: items})
 
    Args:
        address: {"street_address": str, "city": str, "state": str,
                  "country": str, "zip_code": int}
        items:   [{"product_id": str, "quantity": int}, ...]
 
    Returns:
        {"currency_code": str, "units": int, "nanos": int}
    """
 
    logger.info("get_shipping_quote: city=%s items=%d", address.get("city"), len(items))
    proto_items = [
        demo_pb2.CartItem(product_id=i["product_id"], quantity=i["quantity"])
        for i in items
    ]
    proto_addr = demo_pb2.Address(
        street_address=address.get("street_address", ""),
        city=address.get("city", ""),
        state=address.get("state", ""),
        country=address.get("country", ""),
        zip_code=int(address.get("zip_code", 0)),
    )
    resp = _shipping_stub().GetQuote(
        demo_pb2.GetQuoteRequest(address=proto_addr, items=proto_items)
    )
    result = {
        "currency_code": resp.cost_usd.currency_code,
        "units":         resp.cost_usd.units,
        "nanos":         resp.cost_usd.nanos,
    }
    logger.info("get_shipping_quote: result=%s", result)
    return result


# ---------------------------------------------------------------------------
# Tool 5 — charge_card
# ---------------------------------------------------------------------------
 
def charge_card(currency_code: str, units: int, nanos: int,
                credit_card_number: str, credit_card_cvv: int,
                credit_card_expiry_year: int, credit_card_expiry_month: int) -> dict:
    """
    Charge the customer's credit card via PaymentService.
 
    Mirrors:  PaymentService.Charge(ChargeRequest{Amount: money, CreditCard: card})
 
    Returns:
        {"transaction_id": str}
    """
 
    logger.info("charge_card: amount=%s %d.%09d", currency_code, units, nanos)
    resp = _payment_stub().Charge(
        demo_pb2.ChargeRequest(
            amount=demo_pb2.Money(
                currency_code=currency_code,
                units=units,
                nanos=nanos,
            ),
            credit_card=demo_pb2.CreditCardInfo(
                credit_card_number=credit_card_number,
                credit_card_cvv=credit_card_cvv,
                credit_card_expiry_year=credit_card_expiry_year,
                credit_card_expiry_month=credit_card_expiry_month,
            ),
        )
    )
    result = {"transaction_id": resp.transaction_id}
    logger.info("charge_card: transaction_id=%s", result["transaction_id"])
    return result


# ---------------------------------------------------------------------------
# Tool 6 — ship_order
# ---------------------------------------------------------------------------
 
def ship_order(address: dict, items: list[dict]) -> dict:
    """
    Dispatch the shipment via ShippingService.
 
    Mirrors:  ShippingService.ShipOrder(ShipOrderRequest{Address: addr, Items: items})
 
    Args:
        address: {"street_address": str, "city": str, "state": str,
                  "country": str, "zip_code": int}
        items:   [{"product_id": str, "quantity": int}, ...]
 
    Returns:
        {"tracking_id": str}
    """
 
    logger.info("ship_order: city=%s items=%d", address.get("city"), len(items))
    proto_items = [
        demo_pb2.CartItem(product_id=i["product_id"], quantity=i["quantity"])
        for i in items
    ]
    proto_addr = demo_pb2.Address(
        street_address=address.get("street_address", ""),
        city=address.get("city", ""),
        state=address.get("state", ""),
        country=address.get("country", ""),
        zip_code=int(address.get("zip_code", 0)),
    )
    resp = _shipping_stub().ShipOrder(
        demo_pb2.ShipOrderRequest(address=proto_addr, items=proto_items)
    )
    result = {"tracking_id": resp.tracking_id}
    logger.info("ship_order: tracking_id=%s", result["tracking_id"])
    return result


# ---------------------------------------------------------------------------
# Tool 7 — send_order_confirmation
# ---------------------------------------------------------------------------
 
def send_order_confirmation(email: str, order: dict) -> dict:
    """
    Send an order-confirmation email via EmailService.
 
    Mirrors:  EmailService.SendOrderConfirmation(SendOrderConfirmationRequest{Email, Order})
 
    Args:
        email: customer e-mail address
        order: {
            "order_id":          str,
            "shipping_tracking_id": str,
            "shipping_cost":     {"currency_code": str, "units": int, "nanos": int},
            "shipping_address":  {"street_address": str, "city": str, "state": str,
                                  "country": str, "zip_code": int},
            "items": [
                {
                  "item": {"product_id": str, "quantity": int},
                  "cost": {"currency_code": str, "units": int, "nanos": int}
                }, ...
            ]
          }
 
    Returns:
        {"success": true}
    """
 
    logger.info("send_order_confirmation: email=%s order_id=%s", email, order.get("order_id"))
 
    sc = order.get("shipping_cost", {})
    sa = order.get("shipping_address", {})
 
    proto_order_items = []
    for oi in order.get("items", []):
        it   = oi.get("item", {})
        cost = oi.get("cost", {})
        proto_order_items.append(
            demo_pb2.OrderItem(
                item=demo_pb2.CartItem(
                    product_id=it.get("product_id", ""),
                    quantity=it.get("quantity", 0),
                ),
                cost=demo_pb2.Money(
                    currency_code=cost.get("currency_code", "USD"),
                    units=cost.get("units", 0),
                    nanos=cost.get("nanos", 0),
                ),
            )
        )
 
    order_result = demo_pb2.OrderResult(
        order_id=order.get("order_id", ""),
        shipping_tracking_id=order.get("shipping_tracking_id", ""),
        shipping_cost=demo_pb2.Money(
            currency_code=sc.get("currency_code", "USD"),
            units=sc.get("units", 0),
            nanos=sc.get("nanos", 0),
        ),
        shipping_address=demo_pb2.Address(
            street_address=sa.get("street_address", ""),
            city=sa.get("city", ""),
            state=sa.get("state", ""),
            country=sa.get("country", ""),
            zip_code=int(sa.get("zip_code", 0)),
        ),
        items=proto_order_items,
    )
 
    _email_stub().SendOrderConfirmation(
        demo_pb2.SendOrderConfirmationRequest(email=email, order=order_result)
    )
    logger.info("send_order_confirmation: sent successfully")
    return {"success": True}
 
 
# ---------------------------------------------------------------------------
# Tool 8 — empty_cart
# ---------------------------------------------------------------------------
 
def empty_cart(user_id: str) -> dict:
    """
    Clear a user's cart after successful checkout.
 
    Mirrors:  CartService.EmptyCart(EmptyCartRequest{UserId: userID})
 
    Returns:
        {"success": true}
    """
    logger.info("empty_cart: user_id=%s", user_id)
    _cart_stub().EmptyCart(demo_pb2.EmptyCartRequest(user_id=user_id))
    logger.info("empty_cart: done")
    return {"success": True}




