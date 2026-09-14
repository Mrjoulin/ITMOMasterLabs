import numpy as np
import pandas as pd

from .data_loader import PRICE_COLS


def extract_features(ts_df, start_price):
    if len(ts_df) < 2:
        return None
    features = {}
    x = (ts_df["time"] - ts_df["time"].iloc[0]).dt.total_seconds().values
    # For each price column, compute slope, volatility, mean
    price_cols = [c for c in PRICE_COLS if c.endswith(("mean", "price")) and c in ts_df.columns]
    for col in price_cols:
        na_mask = ts_df[col].isna()
        y = ts_df.loc[~na_mask, col].values
        cur_x = x[~na_mask]
        if len(y) < 2:
            continue
        slope, _ = np.polyfit(cur_x, y, 1)
        features[f"{col}_slope"] = slope
        features[f"{col}_volatility"] = np.std(y)
        features[f"{col}_mean"] = np.mean(y)
        features[f"{col}_last_vs_start"] = (y[-1] - start_price) / start_price * 100 if start_price != 0 else 0

    return features


def create_features(df):
    windows = [10, 60, 300]

    # Ensure sorted by time
    df = df.sort_values('fix_ts').reset_index(drop=True)

    # 3. Price & return features (using bin_price as primary, but keep all)
    for src in ['plm_price', 'bin_price', 'cnb_price']:
        # Lags and differences
        df[f'{src}_lag1'] = df[src].shift(1)
        df[f'{src}_lag2'] = df[src].shift(2)
        df[f'{src}_diff1'] = df[src] - df[f'{src}_lag1']
        df[f'{src}_log_ret'] = np.log(df[src] / df[f'{src}_lag1'])

        # Rolling mean & std of price
        for w in windows:
            df[f'{src}_ma_{w}'] = df[src].rolling(w, min_periods=1).mean()
            df[f'{src}_std_{w}'] = df[src].rolling(w, min_periods=1).std()

        # Rolling mean & std of log returns (volatility)
        for w in windows:
            df[f'{src}_vol_{w}'] = df[f'{src}_log_ret'].rolling(w, min_periods=1).std()

        # Price rate of change (momentum)
        for w in windows:
            df[f'{src}_roc_{w}'] = (df[src] - df[src].shift(w)) / df[src].shift(w)

    for src in ['plm', 'bin', 'cnb']:
        spread_col = f'{src}_spread'
        price_col = f'{src}_price'

        df[f'{spread_col}_norm'] = df[spread_col] / df[price_col]  # normalised spread
        df[f'{spread_col}_diff1'] = df[spread_col].diff()

        for w in windows:
            df[f'{spread_col}_ma_{w}'] = df[spread_col].rolling(w, min_periods=1).mean()
            df[f'{spread_col}_std_{w}'] = df[spread_col].rolling(w, min_periods=1).std()

    # ------------------------------------------------------------------
    # 5. Cross‑exchange features
    # ------------------------------------------------------------------
    # Price differences (arbitrage signals)
    df['bin_plm_diff'] = df['bin_price'] - df['plm_price']
    df['bin_cnb_diff'] = df['bin_price'] - df['cnb_price']
    df['plm_cnb_diff'] = df['plm_price'] - df['cnb_price']

    # Price ratios
    df['bin_plm_ratio'] = df['bin_price'] / df['plm_price']
    df['bin_cnb_ratio'] = df['bin_price'] / df['cnb_price']

    # Spread differences between exchanges
    df['bin_plm_spread_diff'] = df['bin_spread'] - df['plm_spread']
    df['bin_cnb_spread_diff'] = df['bin_spread'] - df['cnb_spread']

    # Rolling correlation of log returns between exchanges (window = 60s)
    corr_window = 60
    df['bin_plm_corr'] = df['bin_price_log_ret'].rolling(corr_window, min_periods=10).corr(df['plm_price_log_ret'])
    df['bin_cnb_corr'] = df['bin_price_log_ret'].rolling(corr_window, min_periods=10).corr(df['cnb_price_log_ret'])

    # Lead‑lag: difference between current price and another exchange's lagged price
    df['bin_lead_plm_lag1'] = df['bin_price'] - df['plm_price'].shift(1)
    df['plm_lead_bin_lag1'] = df['plm_price'] - df['bin_price'].shift(1)

    # 6. Time features (from fix_ts)
    if 'fix_ts' in df.columns and pd.api.types.is_datetime64_any_dtype(df['fix_ts']):
        dt = df['fix_ts']
        df['hour'] = dt.dt.hour
        df['minute'] = dt.dt.minute
        df['second'] = dt.dt.second
        df['dayofweek'] = dt.dt.dayofweek

        # Cyclical encoding (sin/cos)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    # ------------------------------------------------------------------
    # 7. Additional statistics on existing aggregated columns
    # ------------------------------------------------------------------
    for agg in ['mean', 'min', 'max', 'st_diff']:
        for src in ['plm', 'bin', 'cnb']:
            col = f'{src}_{agg}'
            for w in windows:
                df[f'{col}_ma_{w}'] = df[col].rolling(w, min_periods=1).mean()
                df[f'{col}_std_{w}'] = df[col].rolling(w, min_periods=1).std()

    # ------------------------------------------------------------------
    # 8. (Optional) Target creation – uncomment if needed
    # ------------------------------------------------------------------
    # # Predict bin_price change over next N seconds (avoid lookahead)
    # N = 60
    # df['target_bin_future_return'] = df['bin_price'].shift(-N) / df['bin_price'] - 1
    # # For classification (up/down)
    # df['target_bin_up'] = (df['bin_price'].shift(-1) > df['bin_price']).astype(int)

    return df

# Example usage:
# df_features = create_features(original_df, windows=[10, 30, 60, 300])
