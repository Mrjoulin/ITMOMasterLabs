"""Collect Polymarket CLOB top-of-book for both outcomes of the current 5-minute event.

Rewritten 2026-07-30. The previous version had two defects that made its output
unusable for any price or direction research (see
research/2026-07-30-profitability-analysis.md):

  * it subscribed to only the YES token and derived up_/down_ from the `price_change`
    `side` field, which is the BOOK side, not the outcome - so outcome identity was
    silently discarded;
  * it stored a 50-message rolling MEAN of prices, so nothing it recorded was a quote
    you could have traded.

This version subscribes to BOTH tokens, keys everything on `asset_id`, reconstructs
each token's ladder locally, and persists raw executable L1 quotes.
"""

from typing import Any, Dict, Optional, Tuple
import os
import json
from collections import deque

import websockets
import structlog
import asyncio
from prefect import flow, task

from utils import get_cur_ts, ts_to_dt, queue_name
from utils.postgres import TablesInfo, create_postgres_tables, get_event_info, save_order_book_batch
from utils.redis_db import get_redis_client, update_value_redis
from clients.order_book_state import BookRegistry, L1Quote


# New key + new payload shape. Deliberately NOT the old "order:current" so a stale value
# in the old (broken) format can never be silently consumed as if it were valid.
REDIS_L1_KEY = os.getenv("REDIS_ORDER_L1_KEY", "order_l1")
POLYMARKET_ORDER_BOOK_WSS = os.getenv(
    "POLYMARKET_ORDER_BOOK_WSS", "wss://ws-subscriptions-frontend-clob.polymarket.com/ws/market"
)

WINDOW_DURATION = 300
OFFSET_WAIT = 5
POSTGRES_FLUSH_TIMEOUT = 30
POLYMARKET_RECONNECT_TIMEOUT = 0.1
# Persist a row when the top of book actually changes, rate-limited; plus a heartbeat so
# quiet periods still leave a trace. Event-driven raw sampling, never averaging.
# Measured 2026-07-30: the feed runs ~270 book updates/s, and at 100ms/1s this produced
# 5090 rows/event (~293 MB/day). At 250ms/5s on a single leg it is ~1145 rows/event
# (~22 MB/day), which is ample for calibration and repricing analysis given a 1-cent
# tick and a 1-tick spread.
MIN_STORE_INTERVAL_MS = 250
HEARTBEAT_MS = 5000
# Redis drives live trading decisions, so keep it fresher than the archive.
REDIS_PUBLISH_MIN_MS = 100

OUTCOME_UP = "UP"
OUTCOME_DOWN = "DOWN"
# Prices are stored as integer units of PRICE_TICK to keep rows small and avoid
# float-equality traps. 0.4700 -> 4700.
PRICE_TICK = 0.0001

logger = structlog.get_logger(__name__)


@task(name="Get event outcome tokens", retries=3, retry_delay_seconds=1)
def get_outcome_tokens(event_start_ts: int) -> Dict[str, str]:
    """Map CLOB token id -> outcome label for this event.

    events_info stores yes_token_id/no_token_id already normalised so that the Yes/Up
    outcome is first (see clients/polymarket_client.py get_event_by_slug).
    """
    event_info = get_event_info(event_start_ts=event_start_ts)
    if event_info is None:
        raise ValueError(f"No event info for event_start_ts={event_start_ts}")

    yes_token, no_token = event_info.get("yes_token_id"), event_info.get("no_token_id")
    if not yes_token or not no_token:
        raise ValueError(
            f"Both outcome tokens are required, got yes={yes_token} no={no_token}. "
            f"Collecting one side only is what broke the previous dataset."
        )

    return {str(yes_token): OUTCOME_UP, str(no_token): OUTCOME_DOWN}


def build_subscription(asset_to_outcome: Dict[str, str]) -> str:
    return json.dumps({"type": "markets", "assets_ids": list(asset_to_outcome.keys())})


def to_price_ticks(price: Optional[float]) -> Optional[int]:
    return None if price is None else int(round(price / PRICE_TICK))


def l1_to_row(quote: L1Quote, event_start_ts: int) -> Dict[str, Any]:
    """Row for the UP leg only. DOWN is exactly derivable (see order_book_quotes view)."""
    return {
        "event_start_ts": event_start_ts,
        "collect_ts": quote.collect_ts,
        "up_bid_t": to_price_ticks(quote.best_bid),
        "up_ask_t": to_price_ticks(quote.best_ask),
        "up_bid_size": quote.bid_size,
        "up_ask_size": quote.ask_size,
        "up_bid_depth": quote.bid_depth_1c,
        "up_ask_depth": quote.ask_depth_1c,
        "src_lag_ms": (quote.collect_ts - quote.src_ts) if quote.src_ts else None,
    }


def _mirror(price: Optional[float]) -> Optional[float]:
    return None if price is None else round(1.0 - price, 6)


def _leg(
    asset_id: str, bid: Optional[float], ask: Optional[float],
    bid_size: Optional[float], ask_size: Optional[float]
) -> Dict[str, Any]:
    two_sided = bid is not None and ask is not None
    return {
        "asset_id": asset_id,
        "best_bid": bid,
        "best_ask": ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "mid": round((bid + ask) / 2, 6) if two_sided else None,
        "spread": round(ask - bid, 6) if two_sided else None,
    }


def redis_payload(
    up: L1Quote, outcome_assets: Dict[str, str], event_start_ts: int, collect_ts: int
) -> str:
    """Payload for the live decision path. Raw executable quotes, keyed by outcome.

    Both legs come from the single canonical UP quote: the two token books are exact
    mirrors, so deriving DOWN costs nothing and guarantees the payload is internally
    consistent (UP.best_ask + DOWN.best_bid == 1 by construction). It also carries
    `asset_id` per leg so consumers never need a database round-trip to learn the token.
    """
    return json.dumps({
        "event_start_ts": event_start_ts,
        "collect_ts": collect_ts,
        "src_ts": up.src_ts,
        "schema": "l1.v1",
        OUTCOME_UP: _leg(
            outcome_assets[OUTCOME_UP], up.best_bid, up.best_ask, up.bid_size, up.ask_size
        ),
        OUTCOME_DOWN: _leg(
            outcome_assets[OUTCOME_DOWN], _mirror(up.best_ask), _mirror(up.best_bid),
            up.ask_size, up.bid_size
        ),
    })


def should_store(
    quote: L1Quote, last_key: Optional[Tuple], last_ms: int, collect_ts: int
) -> bool:
    """Store on a real top-of-book change (rate-limited), or on heartbeat."""
    changed = last_key != quote.quote_key()
    rate_ok = collect_ts - last_ms >= MIN_STORE_INTERVAL_MS
    stale = collect_ts - last_ms >= HEARTBEAT_MS
    return (changed and rate_ok) or stale


@task(name="Listen to market updates")
async def listen(sub_message: str, listen_end: int):
    """Yield raw websocket messages until listen_end, reconnecting on failure."""
    while get_cur_ts() < listen_end:
        try:
            async with websockets.connect(POLYMARKET_ORDER_BOOK_WSS) as ws:
                await ws.send(sub_message)
                logger.info("connected_to_ws", url=POLYMARKET_ORDER_BOOK_WSS)

                while get_cur_ts() < listen_end:
                    msg = await ws.recv(decode=False)
                    try:
                        data = json.loads(msg)
                    except Exception as e:
                        logger.error("unable_to_parse_json", error=str(e))
                        continue
                    # the feed sends either a single object or a batch
                    for item in data if isinstance(data, list) else [data]:
                        if isinstance(item, dict):
                            yield item
        except Exception as e:
            logger.error("connection_exception", error=str(e))
            await asyncio.sleep(POLYMARKET_RECONNECT_TIMEOUT)


@flow(name="collect_order_book", log_prints=True)
async def collect_order_book():
    start_ts = get_cur_ts(precision="second")
    window_start = int(start_ts - (start_ts % WINDOW_DURATION))
    window_end = window_start + WINDOW_DURATION
    logger.info(
        "starting_order_book_collection",
        window_start=ts_to_dt(window_start), window_end=ts_to_dt(window_end)
    )

    create_postgres_tables(tables=[TablesInfo.ORDER_BOOK_TABLE])

    asset_to_outcome = get_outcome_tokens(event_start_ts=window_start)
    logger.info(
        "subscribing_to_both_outcomes",
        tokens={f"...{a[-8:]}": o for a, o in asset_to_outcome.items()}
    )

    registry = BookRegistry(asset_to_outcome)
    outcome_assets = {outcome: asset for asset, outcome in asset_to_outcome.items()}
    redis_client = get_redis_client()
    current_key = queue_name(REDIS_L1_KEY, "current")

    pending: deque = deque()
    last_key: Optional[Tuple] = None
    last_ms = 0
    last_flush = get_cur_ts()
    last_published_ms = 0
    msg_count = 0

    deadline = window_end + OFFSET_WAIT
    sub_message = build_subscription(asset_to_outcome)

    async for msg in listen(sub_message=sub_message, listen_end=deadline):
        collect_ts = get_cur_ts(precision="millisecond")
        msg_count += 1
        try:
            if not registry.apply_message(msg):
                continue

            # Ladder state is now current. Deriving a quote costs a pass over both
            # ladders, so only do it when something is actually due - at ~270 msg/s with
            # a 250ms store floor and 100ms publish floor that is ~10 derivations/s
            # instead of 270. HEARTBEAT_MS >= MIN_STORE_INTERVAL_MS, so gating the store
            # check on the interval cannot miss a heartbeat.
            due_store = collect_ts - last_ms >= MIN_STORE_INTERVAL_MS
            due_publish = collect_ts - last_published_ms >= REDIS_PUBLISH_MIN_MS
            if not (due_store or due_publish):
                continue

            # Store the UP leg only; DOWN is exactly derivable. canonical_quote merges
            # both ladders, so this is correct even when only the DOWN token ticked.
            quote = registry.canonical_quote(OUTCOME_UP, collect_ts)
            if quote is None:
                continue

            if due_store and should_store(quote, last_key, last_ms, collect_ts):
                pending.append(l1_to_row(quote, window_start))
                last_key = quote.quote_key()
                last_ms = collect_ts

            if due_publish:
                await update_value_redis(
                    redis_client=redis_client, key=current_key,
                    value=redis_payload(quote, outcome_assets, window_start, collect_ts)
                )
                last_published_ms = collect_ts

            if get_cur_ts() - last_flush >= POSTGRES_FLUSH_TIMEOUT and pending:
                await asyncio.to_thread(save_order_book_batch, records=pending)
                last_flush = get_cur_ts()
        except Exception as e:
            logger.error("unable_to_process_message", error=str(e), event_type=msg.get("event_type"))

    if pending:
        await asyncio.to_thread(save_order_book_batch, records=pending)

    logger.info(
        "order_book_collection_finished",
        window_start=ts_to_dt(window_start), messages=msg_count
    )


if __name__ == '__main__':
    asyncio.run(collect_order_book())
