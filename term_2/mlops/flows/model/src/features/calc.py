import warnings
import numpy as np
import pandas as pd
from prefect import task
from prefect.cache_policies import NO_CACHE


warnings.filterwarnings('ignore')
TARGET_DIFF_THRESHOLD = 40

AGG_COLUMNS = [
    "interval_idx", "fix_ts",
    "plm_price", "plm_offset", "plm_st_price", "plm_mean", "plm_min", "plm_max",
    "plm_spread", "plm_st_diff", "plm_min_diff", "plm_max_diff",
    "bin_price", "bin_offset", "bin_st_price", "bin_mean", "bin_min", "bin_max",
    "bin_spread", "bin_st_diff", "bin_min_diff", "bin_max_diff"
]


@task(name="Aggreagate features", cache_policy=NO_CACHE)
def aggregate_features(df, target_threshold: float = None):
    """
    Aggregate ≈1 second micro‑interval data from two sources (plm_ and bin_)
    to 2‑minute feature vectors.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        interval_idx, fix_ts,
        plm_price, plm_offset, plm_st_price, plm_mean, plm_min, plm_max,
        plm_spread, plm_st_diff, plm_min_diff, plm_max_diff,
        bin_price, bin_offset, bin_st_price, bin_mean, bin_min, bin_max,
        bin_spread, bin_st_diff, bin_min_diff, bin_max_diff,
        target_price - if present

    target_threshold: float
        Target diff threshold (if abs(close price - start price) < target_threshold then target 0)

    Returns
    -------
    features : pd.DataFrame
        Index = interval_idx, one row per window.
        Feature names are prefixed with plm_, bin_ or collab_.
    """
    target_threshold = target_threshold or TARGET_DIFF_THRESHOLD

    if 'interval_idx' not in df.columns:
        df['interval_idx'] = 0
        drop_interval_idx = True
    else:
        drop_interval_idx = False

    # Ensure all columns present
    if set(AGG_COLUMNS) - set(df.columns):
        raise RuntimeError(f"Columns missing! Missing: {set(AGG_COLUMNS) - set(df.columns)}")

    with_target = "target_price" in df.columns
    # Ensure ordering
    df = df.sort_values(['interval_idx', 'fix_ts'], ascending=[False, True])

    # Micro‑interval duration (ms → seconds) for both sources
    df['plm_duration_s'] = df['plm_offset'] / 1000.0
    df['bin_duration_s'] = df['bin_offset'] / 1000.0

    # -----------------------------------------------------------------
    # Helper technical indicator functions (unchanged, generic)
    # -----------------------------------------------------------------
    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean().iloc[-1]

    def sma(series, window):
        if len(series) < window:
            return np.nan
        return series.rolling(window).mean().iloc[-1]

    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def macd(series, fast=12, slow=26, signal=9):
        if len(series) < slow:
            return np.nan, np.nan, np.nan
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - macd_signal
        return macd_line.iloc[-1], macd_signal.iloc[-1], hist.iloc[-1]

    def stochastic(high, low, close, k_period=5, d_period=3):
        if len(close) < k_period:
            return np.nan, np.nan
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        pct_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        pct_d = pct_k.rolling(d_period).mean()
        return pct_k.iloc[-1], pct_d.iloc[-1]

    def bollinger(series, window=5):
        if len(series) < window:
            return np.nan, np.nan, np.nan, np.nan
        mid = series.rolling(window).mean()
        std = series.rolling(window).std(ddof=0)
        upper = mid + 2*std
        lower = mid - 2*std
        pct_b = (series - lower) / (upper - lower) * 100
        return mid.iloc[-1], upper.iloc[-1], lower.iloc[-1], pct_b.iloc[-1]

    def atr(high, low, close, period=5):
        if len(close) < period:
            return np.nan
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1]

    # -----------------------------------------------------------------
    # Feature extraction for ONE source (prefix = 'plm' or 'bin')
    # -----------------------------------------------------------------
    def extract_one_source(group, prefix):
        feats = {}
        price = group[f'{prefix}_price']
        st_price = group[f'{prefix}_st_price']
        mean_px = group[f'{prefix}_mean']
        min_px = group[f'{prefix}_min']
        max_px = group[f'{prefix}_max']
        spread = group[f'{prefix}_spread']
        st_diff = group[f'{prefix}_st_diff']
        min_diff = group[f'{prefix}_min_diff']
        max_diff = group[f'{prefix}_max_diff']
        dur_s = group[f'{prefix}_duration_s']
        seconds = (group['fix_ts'] - group['fix_ts'].iloc[0]) / 1000.0
        n = len(group)

        # ----- OHLCV proxies -----
        open_price = st_price.iloc[0]
        high = max_px.max()
        low = min_px.min()
        close = price.iloc[-1]
        feats[f'{prefix}_open'] = open_price
        feats[f'{prefix}_high'] = high
        feats[f'{prefix}_low'] = low
        feats[f'{prefix}_close'] = close
        feats[f'{prefix}_count'] = n
        feats[f'{prefix}_total_duration_s'] = dur_s.sum()
        feats[f'{prefix}_avg_duration_s'] = dur_s.mean()
        feats[f'{prefix}_duration_range_s'] = dur_s.max() - dur_s.min()

        # VWAP (time‑weighted)
        total_dur = dur_s.sum()
        feats[f'{prefix}_vwap_mean'] = (
            np.average(mean_px, weights=dur_s) if total_dur > 0 else np.nan
        )
        feats[f'{prefix}_vwap_price'] = (
            np.average(price, weights=dur_s) if total_dur > 0 else np.nan
        )

        feats[f'{prefix}_range'] = high - low
        feats[f'{prefix}_close_open_diff'] = close - open_price
        feats[f'{prefix}_close_open_ratio'] = close / open_price if open_price != 0 else np.nan
        feats[f'{prefix}_high_close_ratio'] = high / close if close != 0 else np.nan
        feats[f'{prefix}_low_close_ratio'] = low / close if close != 0 else np.nan
        if high != low:
            feats[f'{prefix}_rel_position'] = (close - low) / (high - low)
        else:
            feats[f'{prefix}_rel_position'] = 0.5
        feats[f'{prefix}_hlc3'] = (high + low + close) / 3.0

        # ----- Aggregate micro‑interval statistics -----
        col_map = [
            (price, 'price'),
            (st_price, 'st_price'),
            (mean_px, 'mean'),
            (spread, 'spread'),
            (st_diff, 'st_diff'),
            (min_diff, 'min_diff'),
            (max_diff, 'max_diff')
        ]
        for col, name in col_map:
            feats[f'{prefix}_{name}_mean'] = col.mean()
            feats[f'{prefix}_{name}_std'] = col.std()
            feats[f'{prefix}_{name}_min'] = col.min()
            feats[f'{prefix}_{name}_max'] = col.max()
            feats[f'{prefix}_{name}_skew'] = col.skew()
            feats[f'{prefix}_{name}_kurt'] = col.kurtosis()

        # Total micro‑movement
        feats[f'{prefix}_sum_st_diff'] = st_diff.sum()
        feats[f'{prefix}_sum_abs_st_diff'] = st_diff.abs().sum()
        feats[f'{prefix}_sum_pos_st_diff'] = st_diff[st_diff > 0].sum()
        feats[f'{prefix}_sum_neg_st_diff'] = st_diff[st_diff < 0].sum()
        feats[f'{prefix}_sum_spread'] = spread.sum()

        # ----- Price path -----
        feats[f'{prefix}_first_price'] = price.iloc[0]
        feats[f'{prefix}_last_price'] = close
        feats[f'{prefix}_price_change'] = close - price.iloc[0]
        feats[f'{prefix}_price_change_pct'] = (
            (close / price.iloc[0] - 1) * 100 if price.iloc[0] != 0 else np.nan
        )

        feats[f'{prefix}_n_new_high'] = (max_px == high).sum()
        feats[f'{prefix}_n_new_low'] = (min_px == low).sum()
        feats[f'{prefix}_fraction_new_high'] = feats[f'{prefix}_n_new_high'] / n
        feats[f'{prefix}_fraction_new_low'] = feats[f'{prefix}_n_new_low'] / n

        feats[f'{prefix}_up_moves'] = (st_diff > 0).sum()
        feats[f'{prefix}_down_moves'] = (st_diff < 0).sum()
        feats[f'{prefix}_flat_moves'] = (st_diff == 0).sum()
        feats[f'{prefix}_up_down_ratio'] = (
            feats[f'{prefix}_up_moves'] / feats[f'{prefix}_down_moves']
            if feats[f'{prefix}_down_moves'] > 0 else np.nan
        )

        # ----- Time regularity -----
        time_deltas = group['fix_ts'].diff().dropna() / 1000.0
        feats[f'{prefix}_mean_time_delta_s'] = time_deltas.mean() if len(time_deltas) > 0 else np.nan
        feats[f'{prefix}_std_time_delta_s'] = time_deltas.std() if len(time_deltas) > 0 else np.nan

        # ----- Trend (linear regression on price vs time) -----
        if n >= 2 and (~price.isna()).sum() >= 2:
            t = seconds.values[~price.isna()]
            p = price.values[~price.isna()]
            A = np.vstack([t, np.ones(len(t))]).T
            slope, intercept = np.linalg.lstsq(A, p, rcond=None)[0]
            feats[f'{prefix}_trend_slope'] = slope
            residuals = p - (slope * t + intercept)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((p - np.mean(p))**2)
            feats[f'{prefix}_trend_r2'] = 1 - ss_res/ss_tot if ss_tot != 0 else 1.0
            feats[f'{prefix}_trend_intercept'] = intercept
        else:
            feats[f'{prefix}_trend_slope'] = np.nan
            feats[f'{prefix}_trend_r2'] = np.nan
            feats[f'{prefix}_trend_intercept'] = np.nan

        # ----- Autocorrelation of micro‑returns -----
        if n >= 3:
            acf = pd.Series(st_diff).autocorr(lag=1)
            feats[f'{prefix}_st_diff_autocorr_lag1'] = acf
        else:
            feats[f'{prefix}_st_diff_autocorr_lag1'] = np.nan

        # ----- Technical indicators on the micro‑series -----
        for w in [5, 10, 20]:
            if n >= w:
                feats[f'{prefix}_SMA_{w}'] = sma(price, w)
                feats[f'{prefix}_EMA_{w}'] = ema(price, w)
            else:
                feats[f'{prefix}_SMA_{w}'] = np.nan
                feats[f'{prefix}_EMA_{w}'] = np.nan

        feats[f'{prefix}_RSI_14'] = rsi(price, 14) if n >= 15 else np.nan

        macd_line, macd_signal, macd_hist = macd(price)
        feats[f'{prefix}_MACD_line'] = macd_line
        feats[f'{prefix}_MACD_signal'] = macd_signal
        feats[f'{prefix}_MACD_histogram'] = macd_hist

        stoch_k, stoch_d = stochastic(max_px, min_px, price, k_period=5, d_period=3)
        feats[f'{prefix}_Stoch_%K_5'] = stoch_k
        feats[f'{prefix}_Stoch_%D_5'] = stoch_d

        bb_mid, bb_up, bb_low, bb_pctb = bollinger(price, window=5)
        feats[f'{prefix}_BB_middle_5'] = bb_mid
        feats[f'{prefix}_BB_upper_5'] = bb_up
        feats[f'{prefix}_BB_lower_5'] = bb_low
        feats[f'{prefix}_BB_pct_b_5'] = bb_pctb

        feats[f'{prefix}_ATR_5'] = atr(max_px, min_px, price, period=5)

        for w in [5, 10]:
            if n > w:
                roc = (price.iloc[-1] / price.iloc[-w-1] - 1) * 100
                feats[f'{prefix}_ROC_{w}'] = roc
            else:
                feats[f'{prefix}_ROC_{w}'] = np.nan

        # Activity proxies
        feats[f'{prefix}_count_per_sec'] = n / total_dur if total_dur > 0 else np.nan
        feats[f'{prefix}_tw_avg_spread'] = (
            np.average(spread, weights=dur_s) if total_dur > 0 else np.nan
        )

        return feats

    # -----------------------------------------------------------------
    # Collaborative (cross‑source) features
    # -----------------------------------------------------------------
    def extract_collaborative_features(group):
        feats = {}
        plm_price = group['plm_price']
        bin_price = group['bin_price']
        plm_st_diff = group['plm_st_diff']
        bin_st_diff = group['bin_st_diff']
        plm_spread = group['plm_spread']
        bin_spread = group['bin_spread']
        plm_mean = group['plm_mean']
        bin_mean = group['bin_mean']
        plm_dur = group['plm_duration_s']
        bin_dur = group['bin_duration_s']

        n_plm = len(plm_price)
        n_bin = len(bin_price)
        # (In practice both should have the same length, but we allow mismatch)

        # --- Price level differences (at micro‑interval level) ---
        price_diff = plm_price - bin_price
        feats['collab_price_diff_mean'] = price_diff.mean()
        feats['collab_price_diff_std'] = price_diff.std()
        feats['collab_price_diff_min'] = price_diff.min()
        feats['collab_price_diff_max'] = price_diff.max()
        feats['collab_price_diff_last'] = price_diff.iloc[-1]

        # Ratio
        ratio = plm_price / bin_price
        feats['collab_price_ratio_mean'] = ratio.mean()
        feats['collab_price_ratio_std'] = ratio.std()
        feats['collab_price_ratio_last'] = ratio.iloc[-1]

        # --- OHLC differences (using computed OHLC values) ---
        plm_open = group['plm_st_price'].iloc[0]
        bin_open = group['bin_st_price'].iloc[0]
        feats['collab_open_diff'] = plm_open - bin_open

        plm_high = group['plm_max'].max()
        bin_high = group['bin_max'].max()
        feats['collab_high_diff'] = plm_high - bin_high

        plm_low = group['plm_min'].min()
        bin_low = group['bin_min'].min()
        feats['collab_low_diff'] = plm_low - bin_low

        plm_close = plm_price.iloc[-1]
        bin_close = bin_price.iloc[-1]
        feats['collab_close_diff'] = plm_close - bin_close

        feats['collab_range_diff'] = (plm_high - plm_low) - (bin_high - bin_low)

        # --- Mean price differences ---
        feats['collab_mean_price_diff'] = plm_mean.mean() - bin_mean.mean()

        # --- Spread divergence ---
        spread_diff = plm_spread - bin_spread
        feats['collab_spread_diff_mean'] = spread_diff.mean()
        feats['collab_spread_diff_std'] = spread_diff.std()
        feats['collab_spread_diff_max'] = spread_diff.max()
        feats['collab_total_spread_plm'] = plm_spread.sum()
        feats['collab_total_spread_bin'] = bin_spread.sum()

        # --- St_diff (micro‑return) comparison ---
        st_diff_corr = (
            plm_st_diff.corr(bin_st_diff) if len(plm_st_diff) > 2 else np.nan
        )
        feats['collab_st_diff_corr'] = st_diff_corr

        # Fraction of same sign
        both_positive = (plm_st_diff > 0) & (bin_st_diff > 0)
        both_negative = (plm_st_diff < 0) & (bin_st_diff < 0)
        feats['collab_same_sign_frac'] = (both_positive | both_negative).mean()

        # --- Full price series correlation ---
        feats['collab_price_corr'] = (
            plm_price.corr(bin_price) if len(plm_price) > 2 else np.nan
        )

        # --- Lead‑lag cross‑correlation (lag 1) ---
        if len(plm_price) > 2:
            feats['collab_cross_corr_plm_lead1'] = plm_price.iloc[1:].corr(
                bin_price.iloc[:-1]
            )
            feats['collab_cross_corr_bin_lead1'] = bin_price.iloc[1:].corr(
                plm_price.iloc[:-1]
            )
        else:
            feats['collab_cross_corr_plm_lead1'] = np.nan
            feats['collab_cross_corr_bin_lead1'] = np.nan

        # --- Activity / duration ratios ---
        if n_bin > 0:
            feats['collab_count_ratio'] = n_plm / n_bin
        else:
            feats['collab_count_ratio'] = np.nan

        total_dur_plm = plm_dur.sum()
        total_dur_bin = bin_dur.sum()
        if total_dur_bin > 0:
            feats['collab_total_duration_ratio'] = total_dur_plm / total_dur_bin
        else:
            feats['collab_total_duration_ratio'] = np.nan

        # --- VWAP differences (using simple means; weighted VWAP is inside source features) ---
        # We'll take from group direct (no need to recompute)
        feats['collab_vwap_mean_diff'] = (
            np.average(plm_mean, weights=plm_dur) if total_dur_plm > 0 else np.nan
        ) - (
            np.average(bin_mean, weights=bin_dur) if total_dur_bin > 0 else np.nan
        )

        # --- Trend slope difference ---
        # Recompute slopes if possible
        secs = (group['fix_ts'] - group['fix_ts'].iloc[0]) / 1000.0
        if len(plm_price) >= 2:
            A = np.vstack([secs, np.ones(len(secs))]).T
            slope_plm = np.linalg.lstsq(A, plm_price.values, rcond=None)[0][0]
            slope_bin = np.linalg.lstsq(A, bin_price.values, rcond=None)[0][0]
            feats['collab_trend_slope_diff'] = slope_plm - slope_bin
        else:
            feats['collab_trend_slope_diff'] = np.nan

        return feats

    def create_target(group):
        target_price = group["target_price"].iloc[0]
        start_price = group["plm_st_price"].iloc[0]
        bin_target = int(target_price > start_price) * 2 - 1
        target = bin_target if abs(target_price - start_price) >= target_threshold else 0

        return {"target": target, "bin_target": bin_target}

    # Master apply function
    def process_group(group):
        feats = {}
        feats.update(extract_one_source(group, 'plm'))
        feats.update(extract_one_source(group, 'bin'))
        feats.update(extract_collaborative_features(group))
        if with_target:
            feats.update(create_target(group))

        return pd.Series(feats)

    # Apply per interval
    features_df = df.groupby('interval_idx').apply(process_group).reset_index()
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if drop_interval_idx:
        features_df.drop("interval_idx", axis=1, inplace=True)
    else:
        features_df.sort_values("interval_idx", ascending=False, inplace=True)

    return features_df
