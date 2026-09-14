import math
import pandas as pd

from .utils import check_pos_int

PRICE_COLS = [
    "plm_price", "plm_st_price", "plm_spread", "plm_st_diff", "plm_mean",
    "plm_min", "plm_max", "plm_min_diff", "plm_max_diff", "plm_offset",
    "bin_price", "bin_st_price", "bin_spread", "bin_st_diff", "bin_mean",
    "bin_min", "bin_max", "bin_min_diff", "bin_max_diff", "bin_offset"
    # "cnb_price", "cnb_spread", "cnb_st_diff", "cnb_mean", "cnb_min", "cnb_max",
]
# first agg no more then 2 sec after event start, last agg no less then 2 sec before event ends
FILTER_SECONDS_THRESHOLD = 2
MIN_EVENT_CNT_PRICES = 250
MIN_INTERVAL_ROWS_PERC = 0.8


class SQLMaker:
    @staticmethod
    def get_events_query(n_events: int = None):
        query = """
            SELECT event_start_ts, start_price, close_price
            FROM polymarket.events_info
            WHERE start_price IS NOT NULL
            ORDER BY event_start_ts DESC
        """
        if check_pos_int(n_events):
            query += f"LIMIT {n_events}"
        return query

    @staticmethod
    def get_event_prices_query(event_start_ts: int, window_seconds: int):
        if not (check_pos_int(event_start_ts) and check_pos_int(window_seconds)):
            raise RuntimeError("Invalid arguments, all should be int")

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
        return query

    @staticmethod
    def get_sampled_prices_query(window_seconds: int, event_duration_sec: int, n_events: int = None):
        if not (check_pos_int(window_seconds) and check_pos_int(event_duration_sec)):
            raise RuntimeError("Invalid arguments, all should be int")
        if window_seconds >= event_duration_sec:
            raise RuntimeError("Invalid arguments, window_seconds should be less then event_duration_sec")

        target_offset = event_duration_sec - window_seconds
        target_interval_offset = math.ceil(target_offset / window_seconds)
        min_rows_interval = int(window_seconds * MIN_INTERVAL_ROWS_PERC)

        cols_str = ", ".join(PRICE_COLS)

        events_query = SQLMaker.get_events_query(n_events=n_events)
        query = f"""
        with cte_events as ({events_query}),
        cte_prices as (
            SELECT 
                agg.time, 
                agg.event_start_ts, 
                agg.fix_ts,
                {cols_str}
            FROM polymarket.price_agg as agg
            JOIN cte_events as ev 
                ON agg.event_start_ts = ev.event_start_ts
            WHERE (agg.event_start_ts + {event_duration_sec} + 1) * 1000 > agg.fix_ts
        ),
        cte_filter_events as (
            select 
                event_start_ts,
                first_fix_ts,
                last_fix_ts,
                cnt_prices
            from (
                select 
                    event_start_ts, 
                    MIN(fix_ts) as first_fix_ts, 
                    MAX(fix_ts) as last_fix_ts,
                    COUNT(fix_ts) as cnt_prices
                from cte_prices
                group by event_start_ts
            )
            where 1=1
                and (first_fix_ts - event_start_ts * 1000) <= {FILTER_SECONDS_THRESHOLD} * 1000
                and ((event_start_ts + {event_duration_sec}) * 1000 - last_fix_ts) <= {FILTER_SECONDS_THRESHOLD} * 1000
                and cnt_prices >= {MIN_EVENT_CNT_PRICES}
        ),
        cte_period_fix_ts as (
            select
                (MAX(last_fix_ts) - {target_offset} * 1000) as data_end_ts
            from cte_filter_events
        ),
        cte_prices_grouping as (
            SELECT
                FLOOR((prd.data_end_ts - prc.fix_ts) / ({window_seconds} * 1000)) as interval_idx,
                prc.*
            FROM cte_prices as prc
            JOIN cte_filter_events flt
                ON prc.event_start_ts = flt.event_start_ts
            CROSS JOIN cte_period_fix_ts prd
        ),
        cte_target_index as (
            SELECT
                interval_idx,
                (MAX(fix_ts) + {target_offset} * 1000) as target_ts,
                (interval_idx - {target_interval_offset}) as target_interval_idx,
                COUNT(*) as cnt_rows
            FROM cte_prices_grouping
            GROUP BY interval_idx
        ),
        cte_collect_targets as (
            SELECT DISTINCT
                trg_ind.interval_idx,
                FIRST_VALUE(prc.plm_price) OVER (
                    PARTITION BY trg_ind.interval_idx
                    ORDER BY ABS(prc.fix_ts - trg_ind.target_ts)
                ) as target_price
            FROM cte_target_index as trg_ind
            JOIN cte_prices_grouping as prc
                ON trg_ind.target_interval_idx = prc.interval_idx
                    AND trg_ind.target_ts <= prc.fix_ts + {FILTER_SECONDS_THRESHOLD} * 1000
                    AND trg_ind.target_ts >= prc.fix_ts - {FILTER_SECONDS_THRESHOLD} * 1000
            WHERE trg_ind.cnt_rows >= {min_rows_interval}
        )
        SELECT
            prc.interval_idx,
            prc.time, 
            prc.event_start_ts, 
            prc.fix_ts,
            {cols_str},
            trg.target_price
        FROM cte_prices_grouping prc
        JOIN cte_collect_targets trg
            ON prc.interval_idx = trg.interval_idx
        ORDER BY prc.interval_idx DESC, prc.time ASC
        """
        return query


if __name__ == '__main__':
    print(SQLMaker.get_sampled_prices_query(window_seconds=120, event_duration_sec=300, n_events=None))
