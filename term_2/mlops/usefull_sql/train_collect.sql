with cte_events as (
    SELECT event_start_ts, start_price, close_price
    FROM polymarket.events_info
    WHERE start_price IS NOT NULL
    ORDER BY event_start_ts DESC
),
cte_prices as (
    SELECT
        agg.time,
        agg.event_start_ts,
        agg.fix_ts,
        plm_price
    FROM polymarket.price_agg as agg
    JOIN cte_events as ev
        ON agg.event_start_ts = ev.event_start_ts
    WHERE (agg.event_start_ts + 300 + 1) * 1000 > agg.fix_ts
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
        and (first_fix_ts - event_start_ts * 1000) <= 2 * 1000
        and ((event_start_ts + 300) * 1000 - last_fix_ts) <= 2 * 1000
        and cnt_prices >= 250
),
cte_period_fix_ts as (
    select
        (MAX(last_fix_ts) - 180 * 1000) as data_end_ts
    from cte_filter_events
),
cte_prices_grouping as (
    SELECT
        FLOOR((prd.data_end_ts - prc.fix_ts) / (120 * 1000)) as interval_idx,
        prc.*
    FROM cte_prices as prc
    JOIN cte_filter_events flt
        ON prc.event_start_ts = flt.event_start_ts
    CROSS JOIN cte_period_fix_ts prd
),
cte_target_index as (
    SELECT
        interval_idx,
        (MAX(fix_ts) + 180 * 1000) as target_ts,
        (interval_idx - 2) as target_interval_idx,
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
            AND ABS(trg_ind.target_ts - prc.fix_ts) <= 2 * 1000
    WHERE trg_ind.cnt_rows >= 96
),
data_collected as (
    SELECT
        prc.interval_idx,
        prc.time,
        prc.event_start_ts,
        prc.fix_ts,
        plm_price,
        trg.target_price
    FROM cte_prices_grouping prc
    JOIN cte_collect_targets trg
        ON prc.interval_idx = trg.interval_idx
)
SELECT
    COUNT(DISTINCT interval_idx) as cnt_intervals,
    COUNT(DISTINCT event_start_ts) as cnt_events,
    SUM(CASE WHEN target_price < 10 THEN 1 ELSE 0 END) as incorrect_target
FROM data_collected