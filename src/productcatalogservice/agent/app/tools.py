"""
tools.py
--------
Deterministic tool functions for the ProductCatalog agent.
Exactly 3 tools — one per gRPC method in the original service:

  list_products    →  ListProducts(Empty) returns (ListProductsResponse)
  get_product      →  GetProduct(GetProductRequest) returns (Product)
  search_products  →  SearchProducts(SearchProductsRequest) returns (SearchProductsResponse)

Catalog loading mirrors the Go parseCatalog() logic:
  - Loaded once at startup and cached (_reload_catalog = False)
  - When SIGUSR1 is received by server.py, _reload_catalog is set True,
    causing the catalog to be re-parsed on every request (the intentional bug)
  - SIGUSR2 restores normal caching behaviour
  - EXTRA_LATENCY env var injects a sleep on every call (mirrors Go behaviour)
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CATALOG_FILE = Path(__file__).parent / "products.json"

# Mirrors the Go server's catalogMutex-protected catalog and reloadCatalog flag
_catalog: list[dict] | None = None
_reload_catalog: bool = False   # set True by SIGUSR1, False by SIGUSR2


def set_reload_flag(value: bool) -> None:
    """Called by server.py signal handlers to toggle the reload-on-every-request bug."""
    global _reload_catalog, _catalog
    _reload_catalog = value
    if value:
        _catalog = None  # force immediate re-parse on next request


def _parse_catalog() -> list[dict]:
    """
    Parse products.json and return list of product dicts.
    Mirrors Go's parseCatalog()
    """
    with open(CATALOG_FILE, "r") as f:
        data = json.load(f)
    return data.get("products", [])


def _get_catalog() -> list[dict]:
    """
    Return the catalog, decides whether to use cache or reload from disk.
    Also handles artificial latency
    """
    global _catalog

    if _reload_catalog or _catalog is None:
        logger.debug("Parsing catalog from disk (reload_flag=%s)", _reload_catalog)
        _catalog = _parse_catalog()

    # Mirror EXTRA_LATENCY env var behaviour
    extra = os.environ.get("EXTRA_LATENCY", "")
    if extra:
        try:
            seconds = float(extra.rstrip("s"))
            time.sleep(seconds)
        except ValueError:
            pass

    return _catalog


# ---------------------------------------------------------------------------
# Tool 1: ListProducts
# ---------------------------------------------------------------------------

def list_products() -> dict[str, Any]:
    """
    Return all products in the catalog.

    Mirrors: rpc ListProducts(Empty) returns (ListProductsResponse)
    Response shape: { "products": [ <Product>, ... ] }
    """
    products = _get_catalog()
    logger.info("list_products: returning %d products", len(products))
    return {"products": products}


# ---------------------------------------------------------------------------
# Tool 2: GetProduct
# ---------------------------------------------------------------------------

def get_product(product_id: str) -> dict[str, Any]:
    """
    Return a single product by ID.

    Mirrors: rpc GetProduct(GetProductRequest) returns (Product)
    GetProductRequest: { id: string }

    Returns the Product dict on success, or raises ValueError (→ NOT_FOUND)
    if no product matches the given ID.
    """
    if not product_id or not product_id.strip():
        raise ValueError("product_id must not be empty")

    products = _get_catalog()
    for p in products:
        if p["id"] == product_id.strip():
            logger.info("get_product: found id=%s", product_id)
            return p

    raise ValueError(f"no product with ID '{product_id}' found")


# ---------------------------------------------------------------------------
# Tool 3: SearchProducts
# ---------------------------------------------------------------------------

def search_products(query: str) -> dict[str, Any]:
    """
    Search products by keyword against name, description, and categories.

    Mirrors: rpc SearchProducts(SearchProductsRequest) returns (SearchProductsResponse)
    SearchProductsRequest: { query: string }
    Response shape: { "results": [ <Product>, ... ] }

    Matching is case-insensitive substring, same as the Go implementation.
    """
    if not query or not query.strip():
        raise ValueError("search query must not be empty")

    q = query.lower().strip()
    products = _get_catalog()

    results = [
        p for p in products
        if q in p.get("name", "").lower()
        or q in p.get("description", "").lower()
        or any(q in cat.lower() for cat in p.get("categories", []))
    ]

    logger.info("search_products: query=%r matched %d products", query, len(results))
    return {"results": results}