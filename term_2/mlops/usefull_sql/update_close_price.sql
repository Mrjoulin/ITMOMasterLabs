UPDATE polymarket.events_info as ev
SET close_price = subquery.close_price
FROM (
    SELECT
        event_start_ts,
        close_price
    FROM (
        SELECT
            event_start_ts,
            LEAD(event_start_ts) OVER (PARTITION BY 1 ORDER BY event_start_ts ASC) as next_event_ts,
            LEAD(start_price) OVER (PARTITION BY 1 ORDER BY event_start_ts ASC) as close_price
        FROM polymarket.events_info
        WHERE start_price IS NOT NULL AND close_price IS NULL
    )
    WHERE next_event_ts IS NOT NULL AND next_event_ts - event_start_ts <= 300
) AS subquery
WHERE ev.event_start_ts = subquery.event_start_ts;
