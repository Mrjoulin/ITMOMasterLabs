-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

----------------- PRICE RAW TABLE --------------------

-- Create hypertable for main features storage
CREATE TABLE IF NOT EXISTS polymarket.raw_prices (
    time TIMESTAMPTZ NOT NULL,
    collect_ts BIGINT NOT NULL,
    event_start_ts BIGINT NOT NULL,
    agg_ts BIGINT NOT NULL,
    source_ts BIGINT NOT NULL,
    source VARCHAR(3) NOT NULL,
    price NUMERIC(20,8) NOT NULL
);

-- Convert to hypertable with 1-hour chunks for ultra-fast queries
SELECT create_hypertable('polymarket.raw_prices', 'time', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);

-- Enable compression for older data
ALTER TABLE polymarket.raw_prices SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_segmentby = 'event_start_ts'
);

-- Add compression policy (compress after 24 hours)
SELECT add_compression_policy('polymarket.raw_prices', compress_after => '24 hours'::interval, if_not_exists => TRUE);

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_raw_prices_event_ts ON polymarket.raw_prices (event_start_ts, time DESC);
--CREATE INDEX IF NOT EXISTS idx_raw_prices_agg_ts ON polymarket.raw_prices (agg_ts);
--CREATE INDEX IF NOT EXISTS idx_raw_prices_source_ts ON polymarket.raw_prices (source_ts);
