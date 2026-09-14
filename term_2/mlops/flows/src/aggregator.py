import os
import json
from collections import deque
from typing import Dict, Any, Optional

import asyncio
import structlog
from redis.asyncio import Redis
from prefect import flow, docker

from utils import get_cur_ts, ts_to_dt, run_in_thread, queue_name
from utils.redis_db import get_redis_client, publish_redis
from utils.postgres import TablesInfo, create_postgres_tables, save_features_batch
from utils.postgres import save_raw_prices_batch, save_event_start_price


# Configuration
REDIS_DATA_QUEUE_KEY = os.getenv("REDIS_DATA_QUEUE_KEY", "data")
REDIS_AGG_QUEUE_KEY = os.getenv("REDIS_AGG_QUEUE_KEY", "agg")
SAVE_RAW_DATA: bool = os.getenv("SAVE_RAW_DATA", "false").lower() in ('true', 'yes', '1')

# Constants
WINDOW_DURATION = 300      # 5 minutes
OFFSET_WAIT = 5            # 5 seconds more
REDIS_BLPOP_TIMEOUT = 0.5
POSTGRES_FLUSH_TIMEOUT = {
    "features": 60,
    "raw": 50
}
SOURCES = ["PLM", "BIN", "CNB"]
MIN_READY_SOURCES = 2

logger = structlog.get_logger(__name__)


async def _aggregate(
    redis_client: Redis, window_start: int,
    current_prices: Dict[str, Dict[str, Any]],
    aggregated_data: deque, raw_data: deque, fix_ts: Optional[int] = None,
    event_start_prices: Optional[Dict[str, float]] = None
):
    # Store PLM data
    if fix_ts is None:
        fix_ts = get_cur_ts(precision="millisecond")
    _features = {"fix_ts": fix_ts}
    event_start_prices = event_start_prices or {}

    for _src in current_prices["ready"]:
        _cur_info = current_prices[_src]
        _src_low = _src.lower()
        _st_price, _cur_price = _cur_info["vals"][0], _cur_info["vals"][-1]
        _cur_min, _cur_max = _cur_info["min"], _cur_info["max"]

        # NOTE: `st_price`/`st_diff` are scoped to THIS ~1-second micro-batch, because
        # the buffer is reset after every aggregation. They are momentum, NOT distance
        # from the event's strike price. The `event_st_price`/`dist_to_strike` fields
        # below are the event-relative quantities - use those for anything directional.
        _features[f"{_src_low}_st_price"] = _st_price
        _features[f"{_src_low}_price"] = _cur_price
        _features[f"{_src_low}_offset"] = fix_ts - _cur_info["ts"][-1]
        _features[f"{_src_low}_mean"] = _cur_info["sum"] / len(_cur_info["vals"])
        _features[f"{_src_low}_min"] = _cur_min
        _features[f"{_src_low}_max"] = _cur_max
        _features[f"{_src_low}_spread"] = _cur_max - _cur_min
        _features[f"{_src_low}_st_diff"] = _cur_price - _st_price
        _features[f"{_src_low}_min_diff"] = _cur_price - _cur_min
        _features[f"{_src_low}_max_diff"] = _cur_price - _cur_max

        # True event-relative state: the single most important variable for these
        # markets, and the one the previous schema never recorded.
        _event_st = event_start_prices.get(_src)
        _features[f"{_src_low}_event_st_price"] = _event_st
        _features[f"{_src_low}_dist_to_strike"] = (
            _cur_price - _event_st if _event_st is not None else None
        )
        _features[f"{_src_low}_elapsed_s"] = fix_ts / 1000.0 - window_start

    aggregated_data.append(_features)

    await publish_redis(
        redis_client,
        queue=queue_name(REDIS_AGG_QUEUE_KEY, window_start),
        msg=json.dumps(_features)
    )

    # Save raw data if needed
    if not SAVE_RAW_DATA:
        return

    for _src in current_prices["ready"]:
        _cur_info = current_prices[_src]
        for _price, _src_ts, _col_ts in zip(_cur_info["vals"], _cur_info["ts"], _cur_info["col_ts"]):
            # (time, collect_ts, event_start_ts, agg_ts, source_ts, source, price)
            raw_data.append((
                ts_to_dt(_col_ts, to_str=False), _col_ts, window_start, fix_ts, _src_ts, _src, _price
            ))


@flow(name="aggregator_streaming", log_prints=True)
async def aggregator_streaming_flow():
    logger.info(f"Configuration")
    logger.info(f"  REDIS_DATA_QUEUE_KEY: {REDIS_DATA_QUEUE_KEY}")
    logger.info(f"  REDIS_AGG_QUEUE_KEY: {REDIS_AGG_QUEUE_KEY}")
    logger.info(f"  SAVE_RAW_DATA: {SAVE_RAW_DATA}")

    # Determine the 5‑minute window (based on flow start time)
    start_ts = get_cur_ts(precision="second")
    window_start = int(start_ts - (start_ts % WINDOW_DURATION))
    window_end = window_start + WINDOW_DURATION
    logger.info(
        "Starting streaming aggregation for event",
        start_ts=ts_to_dt(start_ts), window_start=ts_to_dt(window_start), window_end=ts_to_dt(window_end)
    )

    # Connect to Redis client
    data_queue_name = queue_name(REDIS_DATA_QUEUE_KEY, window_start)
    redis_client = get_redis_client()
    logger.info("Redis client created")

    # Create all needed tables in postgres by table creation script
    create_postgres_tables(
        tables=[TablesInfo.EVENTS_INFO_TABLE, TablesInfo.FEATURES_TABLE, TablesInfo.RAW_PRICES_TABLE]
    )

    # Raw and aggregated rows
    raw_data = deque()
    aggregated_data = deque()

    current_prices = {}

    def _reset_current_prices():
        current_prices["ready"] = set()
        for _src in SOURCES:
            current_prices[_src] = {
                "vals": [],    # Prices
                "ts": [],      # Source timestamps
                "col_ts": [],  # Collect timestamps
                "sum": 0,
                "min": None,
                "max": None,
            }
    _reset_current_prices()

    window_start_price = None
    # First price seen in this event per source - the strike reference for
    # dist_to_strike. Survives the per-second reset of `current_prices`.
    event_start_prices: Dict[str, float] = {}
    last_flush = {"features": get_cur_ts(), "raw": get_cur_ts()}
    last_message_ts = None

    # Main loop
    deadline = window_end + OFFSET_WAIT
    while get_cur_ts() < deadline:
        try:
            msg = await redis_client.blpop(data_queue_name, timeout=REDIS_BLPOP_TIMEOUT)
            if msg is None or msg[1] is None:
                continue
            msg = msg[1]

            tick = json.loads(msg)
            data_ts_ms = tick.get('ts')
            src = tick.get('src')
            value = tick.get('v')

            # Base checks
            if None in (data_ts_ms, src, value):
                logger.warning("got_nulls_in_message", msg=msg)
                continue
            if data_ts_ms // 1000 < window_start or data_ts_ms // 1000 >= window_end:
                logger.warning("data_ts_not_in_interval", data_ts=ts_to_dt(data_ts_ms), msg=msg)
                continue
            if src not in SOURCES:
                logger.warning("src_not_in_list", src=src, msg=msg)
                continue

            last_message_ts = get_cur_ts(precision="millisecond")

            if src not in event_start_prices:
                event_start_prices[src] = value

            src_prices = current_prices[src]

            src_prices["vals"].append(value)
            src_prices["ts"].append(data_ts_ms)
            src_prices["col_ts"].append(last_message_ts)

            src_prices["sum"] += value
            src_prices["min"] = min(src_prices["min"] or value, value)
            src_prices["max"] = max(src_prices["max"] or value, value)
            current_prices["ready"].add(src)

            if src == "PLM":  # Publish to redis every Polymarket price update
                if len(current_prices["ready"]) >= MIN_READY_SOURCES:
                    run_in_thread(
                        _aggregate(
                            redis_client, window_start, current_prices.copy(),
                            aggregated_data, raw_data,
                            event_start_prices=dict(event_start_prices)
                        )
                    )
                    _reset_current_prices()

                # If first in event PLM price save it as starting price (Postgres)
                if window_start_price is None:
                    window_start_price = value
                    await asyncio.to_thread(
                        save_event_start_price,
                        window_start_ts=window_start, start_price=window_start_price,
                        prev_event_start_ts=window_start - WINDOW_DURATION
                    )

            # Flush to Postgres
            if get_cur_ts() - last_flush["features"] >= POSTGRES_FLUSH_TIMEOUT["features"]:
                await asyncio.to_thread(
                    save_features_batch,
                    records=aggregated_data, window_start_ts=window_start
                )
                last_flush["features"] = get_cur_ts()

            if SAVE_RAW_DATA and get_cur_ts() - last_flush["raw"] >= POSTGRES_FLUSH_TIMEOUT["raw"]:
                await asyncio.to_thread(save_raw_prices_batch, records=raw_data)
                last_flush["raw"] = get_cur_ts()
        except Exception as e:
            logger.warning("Got error while processing", error=str(e))
            if str(e) == 'Event loop is closed':
                print(e)
            continue

    if len(current_prices["ready"]) >= 1:
        await _aggregate(
            redis_client, window_start, current_prices, aggregated_data, raw_data,
            fix_ts=last_message_ts, event_start_prices=dict(event_start_prices)
        )

    if aggregated_data:
        save_features_batch(records=aggregated_data, window_start_ts=window_start)
    if SAVE_RAW_DATA and raw_data:
        save_raw_prices_batch(records=raw_data)

    logger.info(f"Streaming aggregation finished for window {window_start}")


if __name__ == "__main__":
    aggregator_streaming_flow.deploy(
        name="aggregator-streaming",
        work_pool_name="local-pool",
        cron="*/5 * * * *",
        image=docker.DockerImage(
            name="my-prefect-flows",
            tag="aggregator",
            dockerfile="Dockerfile"
        ),
        push=False
    )
