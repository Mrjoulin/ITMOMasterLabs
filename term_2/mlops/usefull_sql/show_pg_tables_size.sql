select
  table_name,
  pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) AS total_relation_size,
  pg_size_pretty(pg_indexes_size(quote_ident(table_name))) AS indexes_size,
  pg_total_relation_size(quote_ident(table_name)) AS total_relation_size_bytes
from information_schema.tables
where table_schema = 'polymarket'
order by 4 desc;

SELECT
    table_name,
    pg_size_pretty(hypertable_size(table_name)) as tbl_size
FROM (VALUES ('raw_prices'), ('order_book'), ('price_agg')) as t (table_name);
