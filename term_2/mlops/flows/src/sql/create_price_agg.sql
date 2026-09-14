-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

----------------- PRICE AGG TABLE --------------------

-- Create hypertable for main features storage
CREATE TABLE IF NOT EXISTS polymarket.price_agg (
    time TIMESTAMPTZ NOT NULL,
    event_start_ts BIGINT NOT NULL,
    fix_ts BIGINT NOT NULL,
    plm_st_price NUMERIC(20,8),
    plm_price NUMERIC(20,8),
    plm_offset NUMERIC(20,8),
    plm_mean NUMERIC(20,8),
    plm_min NUMERIC(20,8),
    plm_max NUMERIC(20,8),
    plm_spread NUMERIC(20,8),
    plm_st_diff NUMERIC(20,8),
    plm_min_diff NUMERIC(20,8),
    plm_max_diff NUMERIC(20,8),
    bin_st_price NUMERIC(20,8),
    bin_price NUMERIC(20,8),
    bin_offset NUMERIC(20,8),
    bin_mean NUMERIC(20,8),
    bin_min NUMERIC(20,8),
    bin_max NUMERIC(20,8),
    bin_spread NUMERIC(20,8),
    bin_st_diff NUMERIC(20,8),
    bin_min_diff NUMERIC(20,8),
    bin_max_diff NUMERIC(20,8),
    cnb_st_price NUMERIC(20,8),
    cnb_price NUMERIC(20,8),
    cnb_offset NUMERIC(20,8),
    cnb_mean NUMERIC(20,8),
    cnb_min NUMERIC(20,8),
    cnb_max NUMERIC(20,8),
    cnb_spread NUMERIC(20,8),
    cnb_st_diff NUMERIC(20,8),
    cnb_min_diff NUMERIC(20,8),
    cnb_max_diff NUMERIC(20,8)
);

-- Event-relative state, added 2026-07-30.
-- CAUTION on the columns above: `*_st_price` / `*_st_diff` are scoped to a single
-- ~1-second micro-batch (the aggregator resets its buffer after every aggregation),
-- so they are momentum, NOT distance from the event's strike. They were mistaken for
-- distance-to-strike; see research/2026-07-30-profitability-analysis.md.
-- The columns below are the real event-relative quantities. Added rather than renamed
-- so the currently-deployed model keeps working; prefer these for new models.
ALTER TABLE polymarket.price_agg ADD COLUMN IF NOT EXISTS plm_event_st_price NUMERIC(20,8);
ALTER TABLE polymarket.price_agg ADD COLUMN IF NOT EXISTS plm_dist_to_strike NUMERIC(20,8);
ALTER TABLE polymarket.price_agg ADD COLUMN IF NOT EXISTS plm_elapsed_s NUMERIC(20,8);
ALTER TABLE polymarket.price_agg ADD COLUMN IF NOT EXISTS bin_event_st_price NUMERIC(20,8);
ALTER TABLE polymarket.price_agg ADD COLUMN IF NOT EXISTS bin_dist_to_strike NUMERIC(20,8);
ALTER TABLE polymarket.price_agg ADD COLUMN IF NOT EXISTS bin_elapsed_s NUMERIC(20,8);
ALTER TABLE polymarket.price_agg ADD COLUMN IF NOT EXISTS cnb_event_st_price NUMERIC(20,8);
ALTER TABLE polymarket.price_agg ADD COLUMN IF NOT EXISTS cnb_dist_to_strike NUMERIC(20,8);
ALTER TABLE polymarket.price_agg ADD COLUMN IF NOT EXISTS cnb_elapsed_s NUMERIC(20,8);

-- Convert to hypertable with 1-hour chunks for ultra-fast queries
SELECT create_hypertable('polymarket.price_agg', 'time', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);

-- Enable compression for older data
ALTER TABLE polymarket.price_agg SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_segmentby = 'event_start_ts'
);

-- Add compression policy (compress after 24 hours)
SELECT add_compression_policy('polymarket.price_agg', compress_after => '24 hours'::interval, if_not_exists => TRUE);

-- Add retention policy (keep data for 90 days)
-- SELECT add_retention_policy('polymarket.price_agg', drop_after => '90 days'::interval, if_not_exists => TRUE);

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_features_event_ts ON polymarket.price_agg (event_start_ts, time DESC);
CREATE INDEX IF NOT EXISTS idx_features_time_event ON polymarket.price_agg (time DESC, event_start_ts);

