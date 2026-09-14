import os

import structlog
import pandas as pd
from prefect import task
from prefect.cache_policies import NO_CACHE
from sqlalchemy import create_engine, text, Engine
from sqlalchemy.orm import Session

from .sql_maker import SQLMaker
from .models import Bet, Base

# Columns added after the table was first created - Base.metadata.create_all() never
# alters an existing table, so new nullable columns are added here idempotently.
BET_ORDER_COLUMNS = {
    "is_live": "BOOLEAN DEFAULT FALSE",
    "token_id": "VARCHAR(128)",
    "order_id": "VARCHAR(128)",
    "order_status": "VARCHAR(16)",
    "filled_price": "DOUBLE PRECISION",
    "filled_size": "DOUBLE PRECISION",
    "tx_hashes": "VARCHAR(512)",
    "order_error": "VARCHAR(512)",
    "quoted_ask": "DOUBLE PRECISION",
    "quoted_bid": "DOUBLE PRECISION",
    "quote_age_ms": "DOUBLE PRECISION",
    "dist_to_strike": "DOUBLE PRECISION",
}

POSTGRES_DSN = os.getenv("PREFECT_API_DATABASE_CONNECTION_URL",)
POSTGRES_DSN_PSYCOPG2 = POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")

logger = structlog.get_logger()


def get_engine() -> Engine:
    return create_engine(POSTGRES_DSN_PSYCOPG2)


def _exec_query(engine: Engine, query: str):
    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error("Unable to load events", sql=query, error=e)
        return None


def load_events(engine: Engine, n_events: int = None):
    query = SQLMaker.get_events_query(n_events=n_events)

    df = _exec_query(engine, query)
    if df is not None:
        logger.info(
            "Loaded events", events_cnt=len(df),
            first_ts=df.iloc[len(df) - 1]['event_start_ts'],
            last_ts=df.iloc[0]['event_start_ts']
        )
    return df


@task(name="Create bets table", retries=3, log_prints=True, cache_policy=NO_CACHE)
def create_bets_table(engine: Engine):
    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        for column, col_type in BET_ORDER_COLUMNS.items():
            conn.execute(text(f"ALTER TABLE bets ADD COLUMN IF NOT EXISTS {column} {col_type}"))


@task(name="Load sampled price", retries=3, log_prints=True, cache_policy=NO_CACHE)
def load_sampled_price_time_series(engine: Engine, window_seconds: int, event_duration_sec: int, n_events: int = None):
    query = SQLMaker.get_sampled_prices_query(
        window_seconds=window_seconds,
        event_duration_sec=event_duration_sec,
        n_events=n_events
    )
    df = _exec_query(engine, query)
    if df is not None:
        logger.info(
            "Loaded samples price time series",
            prices_cnt=len(df), n_interval=df["interval_idx"].nunique(),
            first_ts=df.iloc[len(df) - 1]['time'],
            last_ts=df.iloc[0]['time']
        )
    return df


@task(name="Load event time series", retries=3, log_prints=True, cache_policy=NO_CACHE)
def load_time_series(engine: Engine, event_start_ts, window_seconds):
    query = SQLMaker.get_event_prices_query(
        event_start_ts=event_start_ts,
        window_seconds=window_seconds
    )
    df = _exec_query(engine, query)
    if df is not None:
        logger.info(
            "Loaded event prices",
            prices_cnt=len(df), event_ts=event_start_ts,
            first_ts=df.iloc[len(df) - 1]['time'],
            last_ts=df.iloc[0]['time']
        )
    return df


@task(name="Save bet", retries=3, log_prints=True, cache_policy=NO_CACHE)
def save_bet(engine: Engine, bet: Bet):
    sess = Session(engine)
    sess.add(bet)
    sess.commit()
    sess.close()

