-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

----------------- ORDER BOOK TABLE --------------------
-- One row per (event, outcome token, observation). Raw, executable top-of-book only -
-- any smoothing must happen at read time, never on the way in.
--
-- Rewritten 2026-07-30. The previous schema (up_price/down_price/... columns) was
-- unusable: up_/down_ were keyed off the price_change `side` field, which is a BOOK
-- side and not an outcome, so outcome identity was discarded; and the values were a
-- 50-message rolling MEAN, so they were never executable quotes.
-- See research/2026-07-30-profitability-analysis.md

-- Idempotent migration: if polymarket.order_book exists in a SUPERSEDED shape, move it
-- aside (never delete data) so the canonical name can hold the current schema.
--
-- Generations, identified by a column unique to each:
--   up_price  -> original layout (direction discarded, prices averaged) -> order_book_old
--   asset_id  -> first corrected layout, but ~200 B/row and both legs   -> order_book_firstcut
--   up_bid_t  -> current layout, nothing to do
--
-- Safe to leave in place; it is a no-op once migrated. Remove after the transition.
DO $$
DECLARE
    archive_name TEXT;
    existing_rows BIGINT;
    idx RECORD;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'polymarket' AND table_name = 'order_book'
    ) THEN
        RETURN;                                     -- fresh install
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'polymarket' AND table_name = 'order_book'
          AND column_name = 'up_bid_t'
    ) THEN
        RETURN;                                     -- already current
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'polymarket' AND table_name = 'order_book'
          AND column_name = 'up_price'
    ) THEN
        archive_name := 'order_book_old';
    ELSIF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'polymarket' AND table_name = 'order_book'
          AND column_name = 'asset_id'
    ) THEN
        archive_name := 'order_book_firstcut';
    ELSE
        RAISE EXCEPTION
            'polymarket.order_book exists with an unrecognised schema. '
            'Inspect it and move it aside manually before deploying.';
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'polymarket' AND table_name = archive_name
    ) THEN
        -- Archive already taken, so this table can only be a stray re-creation by an
        -- old collector still running during the rollout. Drop it only if it is empty.
        EXECUTE 'SELECT count(*) FROM polymarket.order_book' INTO existing_rows;
        IF existing_rows = 0 THEN
            RAISE NOTICE 'Dropping empty superseded order_book (% already archived)', archive_name;
            DROP TABLE polymarket.order_book CASCADE;
        ELSE
            RAISE EXCEPTION
                'polymarket.order_book (superseded schema, % rows) and polymarket.% both '
                'exist. Resolve manually before deploying.', existing_rows, archive_name;
        END IF;
        RETURN;
    END IF;

    -- Index names are schema-scoped, so every index has to move too or the new table
    -- cannot claim its own names (order_book_time_idx in particular is auto-created by
    -- create_hypertable for BOTH the old and the new table).
    FOR idx IN
        SELECT indexname FROM pg_indexes
        WHERE schemaname = 'polymarket' AND tablename = 'order_book'
    LOOP
        EXECUTE format(
            'ALTER INDEX polymarket.%I RENAME TO %I',
            idx.indexname, left(idx.indexname || '_' || replace(archive_name, 'order_book_', ''), 63)
        );
    END LOOP;

    EXECUTE format('ALTER TABLE polymarket.order_book RENAME TO %I', archive_name);
    RAISE NOTICE 'Archived superseded order_book -> %', archive_name;
END $$;

-- Storage notes (measured 2026-07-30, see research/data-collection-fix.md):
--   * Only the UP leg is stored. The two token books of a binary market are exact
--     mirrors - prices AND sizes matched on 14865/14865 live samples - so the DOWN leg
--     is 100% derivable and storing it was a pure 2x waste. Use the
--     order_book_quotes view below to get both legs back.
--   * No asset_id column: it is a ~77-char string, constant per (event, outcome), and
--     recoverable from events_info. It alone was ~80 of ~200 bytes per row.
--   * No separate collect_ts: `time` IS the collection time, to microsecond precision.
--   * Prices are SMALLINT in units of 0.0001 ("price ticks"), which covers every tick
--     size Polymarket uses and avoids both NUMERIC bloat and float-equality traps.
--     The view exposes them as proper decimals.
-- Net effect: 199.9 -> 68.3 bytes/row, and 5090 -> ~1145 rows/event.

CREATE TABLE IF NOT EXISTS polymarket.order_book (
    time TIMESTAMPTZ NOT NULL,       -- our receipt time (also the hypertable dimension)
    event_start_ts INT NOT NULL,     -- epoch seconds of the 5-minute window
    up_bid_t SMALLINT,               -- UP best bid, in units of 0.0001
    up_ask_t SMALLINT,               -- UP best ask, in units of 0.0001
    up_bid_size REAL,                -- size resting at UP best bid
    up_ask_size REAL,                -- size resting at UP best ask
    up_bid_depth REAL,               -- cumulative size within 1 cent of UP best bid
    up_ask_depth REAL,               -- cumulative size within 1 cent of UP best ask
    src_lag_ms INT                   -- receipt time minus exchange timestamp
);

-- 1-hour chunks to match the rest of the pipeline
SELECT create_hypertable('polymarket.order_book', 'time', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);

ALTER TABLE polymarket.order_book SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_segmentby = 'event_start_ts'
);

-- 4h (not 24h): keeps the uncompressed working set small. ~4.3x measured on
-- deliberately-random data, so a conservative floor.
SELECT add_compression_policy('polymarket.order_book', compress_after => '4 hours'::interval, if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_order_book_event_ts ON polymarket.order_book (event_start_ts, time DESC);

-- Both legs as decimals, for humans and analysis. DOWN is derived by mirroring.
CREATE OR REPLACE VIEW polymarket.order_book_quotes AS
SELECT
    time,
    event_start_ts,
    (up_bid_t / 10000.0)::NUMERIC(6,4) AS up_bid,
    (up_ask_t / 10000.0)::NUMERIC(6,4) AS up_ask,
    up_bid_size,
    up_ask_size,
    ((10000 - up_ask_t) / 10000.0)::NUMERIC(6,4) AS down_bid,
    ((10000 - up_bid_t) / 10000.0)::NUMERIC(6,4) AS down_ask,
    up_ask_size AS down_bid_size,
    up_bid_size AS down_ask_size,
    ((up_bid_t + up_ask_t) / 20000.0)::NUMERIC(6,4) AS up_mid,
    ((up_ask_t - up_bid_t) / 10000.0)::NUMERIC(6,4) AS spread,
    up_bid_depth,
    up_ask_depth,
    src_lag_ms,
    (EXTRACT(EPOCH FROM time) - event_start_ts)::NUMERIC(8,3) AS elapsed_s
FROM polymarket.order_book;
