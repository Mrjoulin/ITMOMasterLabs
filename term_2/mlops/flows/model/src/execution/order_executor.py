import json
from dataclasses import dataclass
from typing import Optional

import structlog
from prefect import task
from prefect.cache_policies import NO_CACHE
from py_clob_client_v2 import MarketOrderArgsV2, OrderType, PartialCreateOrderOptions
from py_clob_client_v2.exceptions import PolyApiException, PolyException
from py_clob_client_v2.order_utils import SideString

from .clob_client import get_clob_client

logger = structlog.get_logger()

# Polymarket CLOB amounts (USDC and outcome-token shares) are fixed-point, 6 decimals.
AMOUNT_DECIMALS = 1_000_000


@dataclass(frozen=True)
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    status: Optional[str] = None
    filled_price: Optional[float] = None
    filled_size: Optional[float] = None
    tx_hashes: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Optional[str] = None


def _order_id_from(response: dict) -> str | None:
    return response.get("orderID") or response.get("order_id")


def _as_float(source: dict, key: str) -> Optional[float]:
    try:
        return float(source[key])
    except (KeyError, TypeError, ValueError):
        return None


def _resolve_fill(order_details: dict) -> tuple[bool, Optional[float], Optional[float]]:
    """Turn a get_order() response into (success, filled_price, filled_size).

    A resolved (matched, partially matched, or killed) FAK order comes back as a
    trade-shaped record with human-decimal `price`/`size_matched` fields, not the raw
    fixed-point amounts used elsewhere in the API - those are only tried as a fallback
    for other response shapes (e.g. an open/resting order representation).
    """
    filled_size = _as_float(order_details, "size_matched")
    filled_price = _as_float(order_details, "price")

    if filled_size is None or filled_price is None:
        maker_amount = _as_float(order_details, "makerAmount")
        taker_amount = _as_float(order_details, "takerAmount")
        filled_taker_amount = _as_float(order_details, "filledTakerAmount")
        if maker_amount is None or not taker_amount or filled_taker_amount is None:
            return False, None, None
        filled_size = filled_taker_amount / AMOUNT_DECIMALS
        filled_price = (maker_amount / AMOUNT_DECIMALS) / (taker_amount / AMOUNT_DECIMALS)

    if not filled_size or filled_size <= 0:
        return False, None, None

    return True, filled_price, filled_size


@task(name="Place market order", retries=0, log_prints=True, cache_policy=NO_CACHE)
def place_market_order(token_id: str, amount_usd: float) -> OrderResult:
    """Buy up to amount_usd worth of token_id as a Fill-And-Kill market order.

    FAK fills whatever is immediately available (0 to amount_usd worth) and kills the
    remainder - never leaves a resting order, so callers never need to poll/reconcile
    one, but unlike FOK the fill can be partial. Never raises - failures are returned
    as a failed OrderResult so the caller can always persist an outcome.

    post_order()'s own response only carries fill info on a best-effort basis, so the
    authoritative state is re-fetched via get_order() right after posting.
    """
    try:
        client = get_clob_client()
        options = PartialCreateOrderOptions(
            tick_size=client.get_tick_size(token_id), neg_risk=client.get_neg_risk(token_id)
        )
        order_args = MarketOrderArgsV2(
            token_id=token_id, amount=amount_usd, side=SideString.BUY, order_type=OrderType.FAK
        )
        post_response = client.create_and_post_market_order(
            order_args=order_args, options=options, order_type=OrderType.FAK
        )
        logger.info("Order posted", token_id=token_id, amount_usd=amount_usd, response=post_response)

        order_id = _order_id_from(post_response)
        if order_id is None:
            return OrderResult(
                success=False, status=post_response.get("status", "REJECTED"),
                error_message=str(post_response.get("errorMsg") or post_response.get("error") or "no order id"),
                raw_response=json.dumps(post_response)[:512],
            )

        try:
            order_details = client.get_order(order_id)
        except Exception as e:
            logger.error("Unable to confirm order resolution", order_id=order_id, error=str(e))
            return OrderResult(
                success=False, order_id=order_id, status="UNCONFIRMED",
                error_message=f"Order posted but resolution check failed: {e}"[:512],
                raw_response=json.dumps(post_response)[:512],
            )

        success, filled_price, filled_size = _resolve_fill(order_details)
        tx_hashes = post_response.get("transactionsHashes")
        return OrderResult(
            success=success,
            order_id=order_id,
            status=order_details.get("status"),
            filled_price=filled_price,
            filled_size=filled_size,
            tx_hashes=",".join(tx_hashes) if isinstance(tx_hashes, list) else tx_hashes,
            error_message=None if success else "order not filled",
            raw_response=json.dumps(order_details)[:512],
        )
    except PolyApiException as e:
        logger.error(
            "Order rejected by CLOB", token_id=token_id, amount_usd=amount_usd,
            status_code=e.status_code, error=e.error_msg
        )
        return OrderResult(success=False, status="REJECTED", error_message=str(e.error_msg)[:512])
    except PolyException as e:
        logger.error("Client-side error placing order", token_id=token_id, amount_usd=amount_usd, error=str(e))
        return OrderResult(success=False, status="ERROR", error_message=str(e)[:512])
    except Exception as e:
        logger.error("Unexpected error placing order", token_id=token_id, amount_usd=amount_usd, error=str(e))
        return OrderResult(success=False, status="ERROR", error_message=str(e)[:512])
