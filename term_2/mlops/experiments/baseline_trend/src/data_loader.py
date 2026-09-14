import os
import yaml
import pandas as pd
from sqlalchemy import create_engine

# All price columns from price_agg table
PRICE_COLS = [
    "plm_price", "plm_spread", "plm_st_diff", "plm_mean", "plm_min", "plm_max",
    "bin_price", "bin_spread", "bin_st_diff", "bin_mean", "bin_min", "bin_max"
]


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    pg = config["postgres"]
    pg["user"] = os.path.expandvars(pg["user"])
    pg["password"] = os.path.expandvars(pg["password"])
    pg["dbname"] = os.path.expandvars(pg["dbname"])
    return config


def get_engine(config):
    pg = config["postgres"]
    conn_str = f"postgresql://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['dbname']}"
    return create_engine(conn_str)


def load_events(engine):
    query = """
        SELECT event_start_ts, start_price, close_price
        FROM polymarket.events_info
        WHERE start_price IS NOT NULL AND close_price IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    df["target"] = (df["close_price"] > df["start_price"]).astype(int)
    df = df[df["start_price"] != df["close_price"]]
    return df[["event_start_ts", "start_price", "target"]].reset_index(drop=True)


def load_time_series(engine, event_start_ts, window_seconds):
    event_start = pd.to_datetime(event_start_ts, unit="s")
    cols_str = ", ".join(PRICE_COLS)
    query = f"""
        SELECT time, {cols_str}
        FROM polymarket.price_agg
        WHERE event_start_ts = {event_start_ts}
          AND time >= TIMESTAMPTZ '{event_start.isoformat()}'
          AND time <= TIMESTAMPTZ '{event_start.isoformat()}' + INTERVAL '{window_seconds} seconds'
        ORDER BY time
    """
    return pd.read_sql(query, engine).dropna(subset=["plm_price"])
