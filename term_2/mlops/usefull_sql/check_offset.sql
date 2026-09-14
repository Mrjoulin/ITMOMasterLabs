select
    count(*) as cnt_ev,
    avg(st_fix_ts - ev_ts) as avg_st_offset,
    min(st_fix_ts - ev_ts) as min_st_offset,
    max(st_fix_ts - ev_ts) as max_st_offset,
    avg(ev_ts + 300000 - end_fix_ts) as avg_end_offset,
    min(ev_ts + 300000 - end_fix_ts) as min_end_offset,
    max(ev_ts + 300000 - end_fix_ts) as max_end_offset,
    avg(cnt_agg) as avg_cnt_agg
from(
    select
      event_start_ts * 1000 as ev_ts,
      min(fix_ts) as st_fix_ts,
      max(fix_ts) as end_fix_ts,
      count(event_start_ts) as cnt_agg
    from price_agg
    where fix_ts < (event_start_ts + 301) * 1000
    group by event_start_ts
)
where 1=1
    and st_fix_ts - ev_ts <= 2000
    and ev_ts + 300000 - end_fix_ts <= 2000
    and cnt_agg >= 200;


select
    ev_ts / 1000 as event_start_ts,
    fix_ts as first_fix_ts,
    cnt_agg,
    (fix_ts - ev_ts) / 1000 as offset_sec
from (
    select distinct
      event_start_ts * 1000 as ev_ts,
      first_value(fix_ts) over w as fix_ts,
      count(event_start_ts) over w as cnt_agg
    from price_agg
    WINDOW w AS (
        partition by event_start_ts
        order by fix_ts asc
        range between unbounded preceding and unbounded following
    )
)
where cnt_agg >= 200
ORDER BY offset_sec DESC
LIMIT 10;
