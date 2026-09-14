import os
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import Union, List, Dict, Any

import structlog
import psycopg2
from psycopg2.extras import execute_values
from prefect import task

POSTGRES_DSN = os.getenv(
    "PREFECT_API_DATABASE_CONNECTION_URL",
    "postgresql+asyncpg://polymarket:polymarket_pass@postgres:5432/polymarket"
)
POSTGRES_DSN_PSYCOPG2 = POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
CREATE_TABLES_SQL_DIR = "/app/src/sql/"
CREATE_TABLES_SQL_PREFIX = "create_"

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class TablesInfo:
    FEATURES_TABLE = "polymarket.price_agg"
    RAW_PRICES_TABLE = "polymarket.raw_prices"
    EVENTS_INFO_TABLE = "polymarket.events_info"
    ORDER_BOOK_TABLE = "polymarket.order_book"
    # Pre-2026-07-30 data: outcome identity discarded + prices smoothed. Not written to
    # and not usable for research; retained only as an archive.
    ORDER_BOOK_OLD_TABLE = "polymarket.order_book_old"
    BETS_TABLE = "polymarket.bets"


# UP leg only - the DOWN leg is exactly derivable (see the order_book_quotes view).
ORDER_BOOK_COLUMNS = [
    "event_start_ts",
    "up_bid_t", "up_ask_t",
    "up_bid_size", "up_ask_size",
    "up_bid_depth", "up_ask_depth",
    "src_lag_ms",
]


@task(name="Get Postgres Client", retries=3)
def get_postgres_connection():
    conn = psycopg2.connect(POSTGRES_DSN_PSYCOPG2)
    with conn.cursor() as cur:
        cur.execute('SELECT 1')
    return conn


@task(name="Create Postgres tables", retries=3)
def create_postgres_tables(tables: Union[str, List[str]]):
    conn = get_postgres_connection()

    tables = tables if isinstance(tables, (list, tuple, set)) else [tables]

    for table in tables:
        table_name = table.split(".")[-1]
        create_script_path = os.path.join(CREATE_TABLES_SQL_DIR, f"{CREATE_TABLES_SQL_PREFIX}{table_name}.sql")

        try:
            with open(create_script_path, "r") as f:
                create_query = f.read()
        except FileNotFoundError:
            logger.error(f"Table creation script for table {table} not found be path {create_script_path}!")
            continue

        logger.info(f"[Postgres] Execute creation script for table: {table}", script=create_script_path)

        with conn.cursor() as cursor:
            cursor.execute(create_query)
            conn.commit()

    conn.close()
    logger.info(f"[Postgres] Tables creation scripts executed")


@task(name="Save Postgres features", retries=3)
def save_features_batch(records: deque, window_start_ts: int):
    """Sync insert of one minute of aggregated data into PostgreSQL."""
    num_records = len(records)
    if num_records == 0:
        logger.warning("[Postgres] Get 0 records to save, skip")
        return

    conn = get_postgres_connection()

    columns = list(records[0].keys())
    insert_query = f"""
        INSERT INTO {TablesInfo.FEATURES_TABLE} 
        (time, event_start_ts, {','.join(columns)})
        VALUES %s
    """
    values = []

    for _ in range(num_records):
        rec = records.pop()
        values.append((
            datetime.fromtimestamp(rec['fix_ts'] / 1000), window_start_ts,
            *(rec.get(col) for col in columns)
        ))

    with conn.cursor() as cursor:
        execute_values(cursor, insert_query, values)
        conn.commit()
    conn.close()
    logger.info(f"[Postgres] Saved {num_records} records batch of features")


@task(name="Save Postgres raw prices", retries=3)
def save_raw_prices_batch(records: deque):
    """Sync insert of one minute of aggregated data into PostgreSQL."""
    num_records = len(records)
    if num_records == 0:
        logger.warning("[Postgres] Get 0 records to save, skip")
        return

    conn = get_postgres_connection()

    insert_query = f"""
        INSERT INTO {TablesInfo.RAW_PRICES_TABLE} 
        (time, collect_ts, event_start_ts, agg_ts, source_ts, source, price)
        VALUES %s
    """
    values = [records.pop() for _ in range(len(records))]

    with conn.cursor() as cursor:
        execute_values(cursor, insert_query, values)
        conn.commit()
    conn.close()
    logger.info(f"[Postgres] Saved {num_records} records batch of raw data")


def save_event_close_price(conn, window_start_ts: int, close_price: float):
    success = False
    win_side = None
    with conn.cursor() as cursor:
        cursor.execute(
            f'SELECT event_start_ts, start_price, close_price '
            f'FROM {TablesInfo.EVENTS_INFO_TABLE} '
            f'WHERE event_start_ts = %s',
            (window_start_ts,)
        )
        res = cursor.fetchone()
        if res is None:
            logger.error(f"[Postgres] Unable to find event to save close price! Event start ts: {window_start_ts}")
        elif res[2] is None:
            cursor.execute(
                f"UPDATE {TablesInfo.EVENTS_INFO_TABLE} SET close_price = %s WHERE event_start_ts = %s",
                (close_price, window_start_ts)
            )
            logger.info(f"[Postgres] Updated event close price for event with ts: {window_start_ts}")
            success = True
        else:
            logger.warning(f"[Postgres] Close price for ts {window_start_ts} already exists, skip")
        conn.commit()

        if res is not None and res[1] is not None:
            win_side = 1 if (res[2] or close_price) >= res[1] else -1

    if win_side is None:
        logger.warning(f"[Postgres] Can't determine win side for event {window_start_ts} - no start price")
        return success

    # Update bet result if was any bets on event
    with conn.cursor() as cursor:
        cursor.execute(
            f'SELECT event_start_ts, bet_side_int, was_correct '
            f'FROM {TablesInfo.BETS_TABLE} '
            f'WHERE event_start_ts = %s',
            (window_start_ts,)
        )
        res = cursor.fetchone()
        if res is None:
            logger.info(f"[Postgres] No bets found for interval {window_start_ts}, skip updating was_correct")
        elif res[1] is None or res[1] == 0:
            logger.info(f"[Postgres] Found bet for interval {window_start_ts} and it's decided not to bet, "
                        f"skip updating was_correct")
        elif res[2] is not None:
            was_correct = win_side == res[1]
            if was_correct == res[2]:
                logger.info(f"[Postgres] Found bet for interval {window_start_ts} and it's have same was_correct")
            else:
                logger.warning(
                    f"[Postgres] Found bet for interval {window_start_ts} and "
                    f"it's have different was_correct = {res[2]} (expected {was_correct})"
                )
        else:
            # Update was_correct
            was_correct = win_side == res[1]
            cursor.execute(
                f"UPDATE {TablesInfo.BETS_TABLE} SET was_correct = %s WHERE event_start_ts = %s",
                (was_correct, window_start_ts)
            )
            logger.info(f"[Postgres] Updated was_correct to {was_correct} for bet on event with ts: {window_start_ts}")
        conn.commit()

    return success


@task(name="Save Postgres start price", retries=3, log_prints=True)
def save_event_start_price(window_start_ts: int, start_price: float, prev_event_start_ts: int = None):
    conn = get_postgres_connection()
    with conn.cursor() as cursor:
        cursor.execute(
            f'SELECT event_start_ts, start_price FROM {TablesInfo.EVENTS_INFO_TABLE} WHERE event_start_ts = %s',
            (window_start_ts,)
        )
        res = cursor.fetchone()
        if res is None:
            cursor.execute(
                f"INSERT INTO {TablesInfo.EVENTS_INFO_TABLE} (event_start_ts, start_price) VALUES (%s, %s)",
                (window_start_ts, start_price)
            )
            logger.info(f"[Postgres] Save new event")
        elif res[1] is None:
            cursor.execute(
                f"UPDATE {TablesInfo.EVENTS_INFO_TABLE} SET start_price = %s WHERE event_start_ts = %s",
                (start_price, window_start_ts)
            )
            logger.info(f"[Postgres] Updated existing event start price for event with ts: {window_start_ts}")
        else:
            logger.warning(f"[Postgres] Start price for ts {window_start_ts} already saved")
        conn.commit()

    # Update previous event close price
    if prev_event_start_ts is not None:
        success = save_event_close_price(conn, prev_event_start_ts, start_price)
        if not success:
            logger.warning(f"Previous event close price not updated! Previous event start ts: {prev_event_start_ts}")

    conn.close()
    logger.info(f"[Postgres] Saved start price for event with ts: {window_start_ts}")


@task(name="Save Postgres event info", retries=3)
def save_event_info(event_info: Dict[str, Any]) -> int:
    assert event_info['start_ts'] is not None, f"All 'start_ts' fields should be not None!"

    start_ts = (event_info["start_ts"],)
    args = (
        event_info["event_id"], event_info["event_slug"], event_info["market_id"],
        event_info["yes_token_id"], event_info["no_token_id"]
    )

    conn = get_postgres_connection()
    with conn.cursor() as cursor:
        cursor.execute(
            f'SELECT * FROM {TablesInfo.EVENTS_INFO_TABLE} WHERE event_start_ts = %s',
            start_ts
        )
        res = cursor.fetchone()
        if res is None:
            cursor.execute(
                f"""
                INSERT INTO {TablesInfo.EVENTS_INFO_TABLE} 
                (event_start_ts, event_id, event_slug, market_id, yes_token_id, no_token_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                """, start_ts + args
            )
            logger.info(f"[Postgres] Save new event info")
            new_event = 1
        else:
            cursor.execute(
                f"""
                UPDATE {TablesInfo.EVENTS_INFO_TABLE} 
                SET (event_id, event_slug, market_id, yes_token_id, no_token_id) = (%s, %s, %s, %s, %s)
                WHERE event_start_ts = %s
                """, args + start_ts
            )
            logger.warning(f"[Postgres] Update event info")
            new_event = 0
        conn.commit()
    conn.close()
    logger.info(f"[Postgres] Saved event info with ts: {start_ts[0]}")

    return new_event


@task(name="Get Postgres event info", retries=3)
def get_event_info(event_start_ts: int) -> Dict[str, Any]:
    assert event_start_ts is not None, f"event_start_ts should be not None!"

    conn = get_postgres_connection()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(
            f'SELECT * FROM {TablesInfo.EVENTS_INFO_TABLE} WHERE event_start_ts = %s', (event_start_ts,)
        )
        res = cursor.fetchone()
    conn.close()
    logger.info(f"[Postgres] Fetched event info by ts: {res}")
    return res


@task(name="Save Postgres Order book", retries=3)
def save_order_book_batch(records: deque):
    """Insert raw top-of-book rows.

    Uses an explicit column list (not records[0].keys()) so a row with a missing field
    can never silently shift values into the wrong columns, and drains `records` from the
    left so a Prefect retry cannot double-insert the rows already written.
    """
    num_records = len(records)
    if num_records == 0:
        logger.warning("[Postgres] Get 0 order book records to save, skip")
        return

    conn = get_postgres_connection()
    insert_query = f"""
        INSERT INTO {TablesInfo.ORDER_BOOK_TABLE}
        (time, {','.join(ORDER_BOOK_COLUMNS)})
        VALUES %s
    """
    values = []
    for _ in range(num_records):
        rec = records.popleft()
        values.append((
            datetime.fromtimestamp(rec["collect_ts"] / 1000),
            *(rec.get(col) for col in ORDER_BOOK_COLUMNS)
        ))

    with conn.cursor() as cursor:
        execute_values(cursor, insert_query, values)
        conn.commit()
    conn.close()
    logger.info(f"[Postgres] Saved {num_records} order book rows")



