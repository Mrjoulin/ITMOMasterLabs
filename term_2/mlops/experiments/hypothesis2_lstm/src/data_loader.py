import os
import yaml
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

PRICE_COLS = [
    "plm_price", "plm_spread", "plm_st_diff", "plm_mean", "plm_min", "plm_max",
    "bin_price", "bin_spread", "bin_st_diff", "bin_mean", "bin_min", "bin_max",
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
    return df[["event_start_ts", "target"]].reset_index(drop=True)


def load_sequence(engine, event_start_ts, window_seconds, window_size, pad_mode="last"):
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
    ts = pd.read_sql(query, engine).dropna(subset=["plm_price"])
    if ts.empty:
        return None
    # Extract numpy array (n_timesteps, n_features)
    values = ts[PRICE_COLS].fillna(0.0).values
    if len(values) >= window_size:
        return values[-window_size:]  # truncate
    else:
        # pad
        if pad_mode == "last":
            pad_val = values[-1]
        else:
            pad_val = values[0]
        padded = np.tile(pad_val, (window_size, 1))
        padded[:len(values)] = values
        return padded


def load_dataset(engine, config):
    events = load_events(engine)
    window_seconds = config["data"]["window_seconds"]
    window_size = config["data"]["window_size"]
    pad_mode = config["data"].get("pad_mode", "last")

    X, y = [], []
    for _, row in events.iterrows():
        seq = load_sequence(engine, row["event_start_ts"], window_seconds, window_size, pad_mode)
        if seq is not None:
            X.append(seq)
            y.append(row["target"])
    X = np.array(X)  # (n_samples, window_size, n_features)
    y = np.array(y)
    return X, y
