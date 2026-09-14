import pandas as pd
import pandas_ta as ta
import numpy as np

# Price columns to compute indicators for
PRICE_COLS = ["plm_price", "bin_price", "cnb_price"]


def compute_indicators_for_series(prices, prefix):
    """Compute technical indicators for a single price series."""
    if len(prices) < 20:
        return {}
    close = pd.Series(prices)
    features = {}

    # RSI
    rsi = ta.rsi(close, length=14)
    features[f"{prefix}_rsi"] = rsi.iloc[-1] if not rsi.empty else 50.0

    # MACD
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        features[f"{prefix}_macd"] = macd_df.iloc[-1, 0]
        features[f"{prefix}_macd_signal"] = macd_df.iloc[-1, 1]
        features[f"{prefix}_macd_hist"] = macd_df.iloc[-1, 2]

    # Bollinger Bands
    bb = ta.bbands(close, length=20)
    if not bb.empty:
        last = bb.iloc[-1]
        features[f"{prefix}_bb_upper"] = last.iloc[0]
        features[f"{prefix}_bb_middle"] = last.iloc[1]
        features[f"{prefix}_bb_lower"] = last.iloc[2]
        last_price = prices[-1]
        bb_range = features[f"{prefix}_bb_upper"] - features[f"{prefix}_bb_lower"]
        features[f"{prefix}_bb_pos"] = (last_price - features[f"{prefix}_bb_lower"]) / bb_range if bb_range != 0 else 0.5

    # Moving averages
    sma5 = ta.sma(close, length=5)
    sma20 = ta.sma(close, length=20)
    features[f"{prefix}_sma5"] = sma5.iloc[-1] if not sma5.empty else prices[-1]
    features[f"{prefix}_sma20"] = sma20.iloc[-1] if not sma20.empty else prices[-1]
    features[f"{prefix}_price_vs_sma5"] = (prices[-1] - features[f"{prefix}_sma5"]) / prices[-1] * 100 if prices[-1] != 0 else 0
    features[f"{prefix}_price_vs_sma20"] = (prices[-1] - features[f"{prefix}_sma20"]) / prices[-1] * 100 if prices[-1] != 0 else 0

    return features


def get_slope(time_x, prices):
    na_mask = prices.isna()
    y = prices.values[~na_mask]
    cur_x = time_x[~na_mask]
    if len(y) < 2:
        return None
    slope, _ = np.polyfit(cur_x, y, 1)
    return slope


def extract_features(ts_df, start_price):
    """Extract features from time-series DataFrame with all price_agg columns."""
    if len(ts_df) < 20:
        return None

    x = (ts_df["time"] - ts_df["time"].iloc[0]).dt.total_seconds().values
    features = {
        "start_price": start_price
    }
    for col in PRICE_COLS:
        if col in ts_df.columns:
            slope = get_slope(x, ts_df[col])

            prices = ts_df[col].dropna().values
            if len(prices) < 20:
                continue
            # Basic stats
            features[f"{col}_slope"] = slope
            features[f"{col}_mean"] = np.mean(prices)
            features[f"{col}_std"] = np.std(prices)
            features[f"{col}_last"] = prices[-1]
            features[f"{col}_min"] = np.min(prices)
            features[f"{col}_max"] = np.max(prices)
            # Technical indicators
            prefix = col.replace("_price", "")
            indicators = compute_indicators_for_series(prices, prefix)
            features.update(indicators)
    return features if features else None
