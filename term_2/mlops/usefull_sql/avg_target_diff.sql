select
    thr,
    COUNT(*) as cnt,
    avg(abs(close_price - start_price)) as avg_diff,
    sum(case when abs(close_price - start_price) >= thr then 1 else 0 end) as trg_cnt,
    avg(case when abs(close_price - start_price) >= thr then abs(close_price - start_price) end) as avg_trg,
    sum(case when abs(close_price - start_price) < thr then 1 else 0 end) as zero_cnt,
    avg(case when abs(close_price - start_price) < thr then abs(close_price - start_price) end) as avg_zero_diff
from events_info
cross join (
    select thr from (values (30), (35), (40), (42), (44), (46), (47), (48), (49), (50)) as t (thr)
) as t
where close_price is not null and start_price is not null
group by thr
order by thr;