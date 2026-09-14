import structlog
import pandas as pd
from sqlalchemy import create_engine

from .sql_maker import SQLMaker

PRICE_COLS = [
    "plm_price", "plm_spread", "plm_st_diff", "plm_mean", "plm_min", "plm_max",
    "bin_price", "bin_spread", "bin_st_diff", "bin_mean", "bin_min", "bin_max",
    "cnb_price", "cnb_spread", "cnb_st_diff", "cnb_mean", "cnb_min", "cnb_max",
]

logger = structlog.get_logger()


def get_engine(config):
    pg = config["postgres"]
    conn_str = f"postgresql://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['dbname']}"
    return create_engine(conn_str)


def _exec_query(engine, query: str):
    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error("Unable to load events", sql=query, error=e)
        return None


def load_events(engine, n_events: int = None):
    query = SQLMaker.get_events_query(n_events=n_events)

    df = _exec_query(engine, query)
    if df is not None:
        logger.info(
            "Loaded events", events_cnt=len(df),
            first_ts=df.iloc[len(df) - 1]['event_start_ts'],
            last_ts=df.iloc[0]['event_start_ts']
        )
    return df


def load_sampled_price_time_series(engine, window_seconds: int, event_duration_sec: int, n_events: int = None):
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


def load_time_series(engine, event_start_ts, window_seconds):
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
