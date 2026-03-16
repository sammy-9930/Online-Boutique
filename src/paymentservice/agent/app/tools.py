"""
tools.py
--------
Deterministic tool functions for the PaymentService agent.
 
  charge_credit_card  →  PaymentService.Charge(ChargeRequest) returns (ChargeResponse)
 
The original charge.js logic (ported faithfully to Python):
  1. Validate the credit card number with the Luhn algorithm
  2. Validate the card is not expired
  3. Validate the card type is supported (Visa, MasterCard, AmEx)
  4. Generate and return a UUID transaction ID (mock — no real payment processor)
"""

import logging
import re
import uuid
from datetime import datetime
from typing import Any
 
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Luhn algorithm
# ---------------------------------------------------------------------------
 
def _luhn_valid(card_number: str) -> bool:
    """
    Return True if card_number passes the Luhn checksum.
    To check if the entered credit card number is valid 
    """
    digits = [int(d) for d in str(card_number) if d.isdigit()]
    if not digits:
        return False
 
    digits.reverse()
    total = 0
    for i, digit in enumerate(digits):
        if i % 2 == 1:          # double every second digit from the right
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
 
    return total % 10 == 0

# ---------------------------------------------------------------------------
# Card-type detection  
# ---------------------------------------------------------------------------
 
CARD_PATTERNS = {
    "Visa":       re.compile(r"^4[0-9]{12}(?:[0-9]{3})?$"),
    "MasterCard": re.compile(r"^5[1-5][0-9]{14}$"),
    "AmEx":       re.compile(r"^3[47][0-9]{13}$"),
}
 
SUPPORTED_CARD_TYPES = set(CARD_PATTERNS.keys())
 
 
def _detect_card_type(card_number: str) -> str:
    """Return the card brand name, or 'Unknown' if not recognised."""
    number = re.sub(r"\s|-", "", card_number)
    for brand, pattern in CARD_PATTERNS.items():
        if pattern.match(number):
            return brand
    return "Unknown"

# ---------------------------------------------------------------------------
# Expiry validation
# ---------------------------------------------------------------------------
 
def _card_expired(expiry_year: int, expiry_month: int) -> bool:
    """Return True if the card has expired."""
    now = datetime.utcnow()
    # Card is valid through the last day of the expiry month
    if expiry_year < now.year:
        return True
    if expiry_year == now.year and expiry_month < now.month:
        return True
    return False

# ---------------------------------------------------------------------------
# charge_credit_card
# ---------------------------------------------------------------------------
 
def charge_credit_card(
    currency_code: str,
    units: int,
    nanos: int,
    credit_card_number: str,
    credit_card_cvv: int,
    credit_card_expiry_year: int,
    credit_card_expiry_month: int,
) -> dict[str, Any]:
    """
    Mock credit-card charge
 
    Steps:
      1. Luhn-validate the card number
      2. Check the card type is supported (Visa / MasterCard / AmEx)
      3. Check the card has not expired
      4. Return a UUID transaction_id 
 
    Args:
        currency_code:            e.g. "USD"
        units:                    integer part of the charge amount
        nanos:                    fractional part (billionths)
        credit_card_number:       card number string (digits, optional spaces/dashes)
        credit_card_cvv:          3- or 4-digit CVV
        credit_card_expiry_year:  4-digit year, e.g. 2030
        credit_card_expiry_month: 1-12
 
    Returns:
        {"transaction_id": str}  on success
        raises ValueError        on validation failure
    """
    logger.info(
        "charge_credit_card: amount=%s %d.%09d card=****%s",
        currency_code, units, nanos,
        str(credit_card_number)[-4:],
    )
 
    # Strip formatting
    card_number_clean = re.sub(r"[\s\-]", "", str(credit_card_number))
 
    # --- Step 1: Luhn check ---
    if not _luhn_valid(card_number_clean):
        raise ValueError(
            f"Credit card info is invalid: card number {card_number_clean} "
            "fails Luhn check."
        )
 
    # --- Step 2: Supported card type ---
    card_type = _detect_card_type(card_number_clean)
    if card_type not in SUPPORTED_CARD_TYPES:
        raise ValueError(
            f"Credit card info is invalid: card type '{card_type}' is not supported. "
            f"Supported types: {', '.join(sorted(SUPPORTED_CARD_TYPES))}."
        )
 
    # --- Step 3: Expiry ---
    if _card_expired(int(credit_card_expiry_year), int(credit_card_expiry_month)):
        raise ValueError(
            f"Credit card info is invalid: card expired "
            f"{credit_card_expiry_month:02d}/{credit_card_expiry_year}."
        )
 
    # --- Step 4: Generate transaction ID (mock) ---
    transaction_id = str(uuid.uuid4())
 
    logger.info(
        "charge_credit_card: success card_type=%s transaction_id=%s",
        card_type, transaction_id,
    )
 
    return {"transaction_id": transaction_id}
 