UPDATE polymarket.bets as b
SET was_correct = (b.bet_side_int = subquery.win_side)
FROM (
    SELECT
        event_start_ts,
        CASE WHEN close_price >= start_price THEN 1 ELSE -1 END as win_side
    FROM polymarket.events_info
    WHERE start_price IS NOT NULL AND close_price IS NOT NULL
) AS subquery
WHERE 1=1
    AND b.was_correct IS NULL
    AND b.bet_side <> 'NO'
    AND b.event_start_ts = subquery.event_start_ts;
