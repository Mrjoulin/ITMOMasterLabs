import os
import json
import time
import asyncio
from typing import Optional

import structlog
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from redis.asyncio import Redis

from src.db.redis_db import get_redis_client
from src.db.postgres import get_engine, create_bets_table, save_bet, Bet
from src.execution.order_executor import place_market_order
from src.inference import inference
from src.utils.utils import load_config, get_cur_ts, ts_to_dt, run_in_thread


REDIS_AGG_QUEUE_KEY = os.getenv("REDIS_AGG_QUEUE_KEY", "agg")
# L1 book published by src/polymarket_orderbook.py. New key on purpose: the old
# "order:current" payload had outcome identity discarded and prices averaged, so it must
# never be read as if it were valid. See research/2026-07-30-profitability-analysis.md
REDIS_ORDER_L1_KEY = os.getenv("REDIS_ORDER_L1_KEY", "order_l1")
EXPECTED_BOOK_SCHEMA = "l1.v1"
MAX_QUOTE_AGE_MS = 5000

TRADING_ENABLED = os.getenv("TRADING_ENABLED", "false").lower() in ("true", "yes", "1")
BET_AMOUNT_USD = float(os.getenv("BET_AMOUNT_USD", "1.0"))

OFFSET_WAIT = 5
REDIS_BLPOP_TIMEOUT = 1
MIN_WAIT_BEFORE_PREDICT = 2
MAX_EXTRA_WAIT_TIMEOUT = 30
MIN_INTERVAL_ROWS_PERC = 0.8

logger = structlog.get_logger()


def get_event_window(config):
    event_duration = config["data"]["event_duration"]
    start_ts = get_cur_ts(precision="second")
    window_start = int(start_ts - (start_ts % event_duration))
    window_end = window_start + event_duration
    return window_start, window_end


async def listen_redis_agg(config, redis_client, time_series: list):
    event_start, event_end = get_event_window(config)
    logger.info(
        "Starting collecting aggregated data for event",
        window_start=ts_to_dt(event_start), window_end=ts_to_dt(event_end)
    )

    # Connect to Redis client
    agg_queue_name = f"{REDIS_AGG_QUEUE_KEY}:{event_start}"
    logger.info("Redis client created")

    # Main loop
    deadline = event_end + OFFSET_WAIT
    while get_cur_ts() < deadline:
        try:
            msg = await redis_client.blpop(agg_queue_name, timeout=REDIS_BLPOP_TIMEOUT)
            if msg is None or msg[1] is None:
                continue
            msg = msg[1]
            agg_data = json.loads(msg)
            if "fix_ts" in agg_data:
                time_series.append(agg_data)
        except Exception as e:
            logger.error("Error while processing message", error=e)


@task(name="Get order book", retries=3, cache_policy=NO_CACHE)
async def get_current_order_book(redis_client: Redis):
    cur_order_key = f'{REDIS_ORDER_L1_KEY}:current'
    result_str = await redis_client.get(cur_order_key)
    if result_str is None:
        raise RuntimeError(
            f"No L1 order book at '{cur_order_key}'. Is the rewritten collect_order_book "
            f"flow deployed and running?"
        )
    return json.loads(result_str)


def validate_book(book: dict, event_start: int) -> None:
    """Refuse to trade on a book that is the wrong schema, wrong event, or stale.

    Betting on a mispriced/stale book is exactly how the previous dataset produced
    fictional entry prices, so this fails loudly rather than degrading quietly.
    """
    schema = book.get("schema")
    if schema != EXPECTED_BOOK_SCHEMA:
        raise RuntimeError(f"Unexpected book schema {schema!r}, expected {EXPECTED_BOOK_SCHEMA!r}")
    if book.get("event_start_ts") != event_start:
        raise RuntimeError(
            f"Book is for event {book.get('event_start_ts')}, expected {event_start}"
        )
    for outcome in ("UP", "DOWN"):
        if outcome not in book:
            raise RuntimeError(f"Book missing {outcome} leg: {sorted(book)}")

    age = get_cur_ts(precision="millisecond") - book.get("collect_ts", 0)
    if age > MAX_QUOTE_AGE_MS:
        raise RuntimeError(f"Book is stale by {age} ms (limit {MAX_QUOTE_AGE_MS} ms)")


def last_dist_to_strike(time_series: list) -> Optional[float]:
    """Distance from the event's strike price at decision time, if available.

    None until the aggregator that emits `plm_dist_to_strike` is deployed.
    """
    for row in reversed(time_series):
        value = row.get("plm_dist_to_strike")
        if value is not None:
            return float(value)
    return None


@flow(name="model-bets", log_prints=True)
def make_model_bets():
    logger.info("Start making bets pipeline")
    config = load_config()

    logger.info("Run listen redis events")
    # Run listen in separate thread
    time_series = []
    redis_client = get_redis_client()
    run_in_thread(listen_redis_agg(config, redis_client, time_series=time_series))

    logger.info("Get postgres engine and create bets table")
    engine = get_engine()
    create_bets_table(engine=engine)
    logger.info("Bets table created")

    event_start, _ = get_event_window(config)
    window_seconds = config["data"]["window_seconds"]
    start_predict_ts = event_start + window_seconds
    wait_timeout = max(start_predict_ts - get_cur_ts(), MIN_WAIT_BEFORE_PREDICT)

    logger.info(f"Wait for {wait_timeout:.2f} sec to receive enough data")
    time.sleep(wait_timeout)  # Sleep until predict time
    wait_end_ts = get_cur_ts()

    min_cnt = int(window_seconds * MIN_INTERVAL_ROWS_PERC)
    if len(time_series) < min_cnt:
        logger.info("Wait ended but still not enough data, wait more", cur_len=len(time_series), requiered_len=min_cnt)
        while len(time_series) < min_cnt and get_cur_ts() < wait_end_ts + MAX_EXTRA_WAIT_TIMEOUT:
            time.sleep(1)

        if get_cur_ts() >= wait_end_ts + MAX_EXTRA_WAIT_TIMEOUT:
            logger.error(
                "Unable to collect enough data to the end of event", cur_len=len(time_series), requiered_len=min_cnt
            )
            raise RuntimeError("Not enough data")
        wait_end_ts = get_cur_ts()

    logger.info(
        "Wait ended, start features calc", wait_end_ts=int(wait_end_ts), expected_wait_end=int(start_predict_ts)
    )

    predicted_side = inference(config, time_series=time_series)
    if predicted_side == 0:
        logger.info("Model suggest not to bet", predicted_side=predicted_side)
        save_bet(
            engine=engine,
            bet=Bet(event_start_ts=event_start, bet_side="NO", bet_side_int=predicted_side)
        )
        return

    outcome = "UP" if predicted_side > 0 else "DOWN"
    logger.info(f"Model suggest to bet - {outcome}", predicted_side=predicted_side)
    cur_order_book = asyncio.run(get_current_order_book(redis_client))
    validate_book(cur_order_book, event_start)

    collect_ob_ts = get_cur_ts(precision="millisecond")
    leg = cur_order_book[outcome]
    # best_ask is what BUYING this outcome actually costs; best_bid is what you could
    # sell it back for. These are raw executable quotes, not averages.
    best_bid, best_ask = leg["best_bid"], leg["best_ask"]
    volume = leg.get("ask_size")
    quote_age_ms = collect_ob_ts - cur_order_book["collect_ts"]
    dist_to_strike = last_dist_to_strike(time_series)
    logger.info(
        "Get current order book", outcome=outcome, best_bid=best_bid, best_ask=best_ask,
        ask_size=volume, spread=leg.get("spread"), quote_age_ms=quote_age_ms,
        dist_to_strike=dist_to_strike
    )
    if best_ask is None:
        logger.error("No ask to buy against, skipping bet", outcome=outcome)
        save_bet(
            engine=engine,
            bet=Bet(
                event_start_ts=event_start, bet_side=outcome, bet_side_int=predicted_side,
                order_status="NO_QUOTE", is_live=False, dist_to_strike=dist_to_strike,
            )
        )
        return

    # The book payload already carries the CLOB token per outcome, so there is no reason
    # to go back to Postgres for it - same source of truth, one less round-trip.
    token_id = leg.get("asset_id") if TRADING_ENABLED else None
    if TRADING_ENABLED and not token_id:
        logger.error("Trading enabled but book carries no asset_id, falling back to paper bet")

    if token_id is not None:
        logger.info("Placing live order", token_id=token_id, outcome=outcome, amount_usd=BET_AMOUNT_USD)
        order_result = place_market_order(token_id=token_id, amount_usd=BET_AMOUNT_USD)
        logger.info("Order result", order_result=order_result)

        # FAK can partially fill, so the actual dollars at risk may be less than
        # BET_AMOUNT_USD - track what was really filled, not what was requested.
        filled_amount_usd = (
            order_result.filled_price * order_result.filled_size
            if order_result.filled_price and order_result.filled_size else 0.0
        )

        save_bet(
            engine=engine,
            bet=Bet(
                event_start_ts=event_start,
                bet_side=outcome,
                bet_side_int=predicted_side,
                bet_price=order_result.filled_price,
                bet_return=(
                    1 / order_result.filled_price
                    if order_result.filled_price and order_result.filled_price > 0 else None
                ),
                bet_best_ask=best_ask,
                bet_best_bid=best_bid,
                bet_volume=volume,
                bet_amount=filled_amount_usd,
                is_live=True,
                token_id=token_id,
                order_id=order_result.order_id,
                order_status=order_result.status,
                filled_price=order_result.filled_price,
                filled_size=order_result.filled_size,
                tx_hashes=order_result.tx_hashes,
                order_error=order_result.error_message,
                quoted_ask=best_ask,
                quoted_bid=best_bid,
                quote_age_ms=quote_age_ms,
                dist_to_strike=dist_to_strike,
            )
        )
        logger.info("Live bet saved", success=order_result.success)
        return

    # Paper mode: trading disabled, or no token id available for this event yet.
    # Price the paper bet at the real best_ask - what buying would actually have cost.
    # The previous version used a smoothed, direction-ambiguous field here, which is why
    # paper P&L was fictional (avg 'entry' 0.507 vs real fills 0.747).
    save_bet(
        engine=engine,
        bet=Bet(
            event_start_ts=event_start,
            bet_side=outcome,
            bet_side_int=predicted_side,
            bet_price=best_ask,
            bet_return=1 / best_ask if best_ask > 0 else None,
            bet_best_ask=best_ask,
            bet_best_bid=best_bid,
            bet_volume=volume,
            bet_amount=BET_AMOUNT_USD,
            is_live=False,
            order_status="PAPER",
            quoted_ask=best_ask,
            quoted_bid=best_bid,
            quote_age_ms=quote_age_ms,
            dist_to_strike=dist_to_strike,
        )
    )
    logger.info("Paper bet saved", entry_price=best_ask)


if __name__ == "__main__":
    make_model_bets()
