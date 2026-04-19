-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create schema for Polymarket data
CREATE SCHEMA IF NOT EXISTS polymarket;

-- Create hypertable for main features storage
CREATE TABLE IF NOT EXISTS polymarket.features (
    time TIMESTAMPTZ NOT NULL,
    event_start_timestamp BIGINT NOT NULL,
    start_price NUMERIC(20,8),
    collection_timestamp BIGINT NOT NULL,
    current_price NUMERIC(20,8),
    polymarket_ask_yes NUMERIC(10,4),
    polymarket_bid_yes NUMERIC(10,4),
    polymarket_ask_no NUMERIC(10,4),
    polymarket_bid_no NUMERIC(10,4),
    spread_yes NUMERIC(10,4),
    spread_no NUMERIC(10,4),
    mid_price_yes NUMERIC(10,4),
    mid_price_no NUMERIC(10,4),
    price_change_1m NUMERIC(10,4),
    price_change_5m NUMERIC(10,4),
    volume_weighted_price NUMERIC(20,8)
);

-- Convert to hypertable with 1-hour chunks for ultra-fast queries
SELECT create_hypertable('polymarket.features', 'time', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);

-- Enable compression for older data
ALTER TABLE polymarket.features SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_segmentby = 'event_start_timestamp'
);

-- Add compression policy (compress after 24 hours)
SELECT add_compression_policy('polymarket.features', compress_after => '24 hours'::interval, if_not_exists => TRUE);

-- Add retention policy (keep data for 90 days)
SELECT add_retention_policy('polymarket.features', drop_after => '90 days'::interval, if_not_exists => TRUE);

-- Create indexes for ultra-fast queries
CREATE INDEX IF NOT EXISTS idx_features_event_ts ON polymarket.features (event_start_timestamp, time DESC);
CREATE INDEX IF NOT EXISTS idx_features_collection_ts ON polymarket.features (collection_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_features_time_event ON polymarket.features (time DESC, event_start_timestamp);

-- Create materialized view for real-time aggregations
CREATE MATERIALIZED VIEW IF NOT EXISTS polymarket.features_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket(INTERVAL '1 hour', time) AS bucket,
    event_start_timestamp,
    AVG(current_price) as avg_price,
    MIN(current_price) as min_price,
    MAX(current_price) as max_price,
    AVG(spread_yes) as avg_spread_yes,
    AVG(spread_no) as avg_spread_no,
    AVG(mid_price_yes) as avg_mid_price_yes,
    AVG(mid_price_no) as avg_mid_price_no,
    COUNT(*) as data_points
FROM polymarket.features
GROUP BY bucket, event_start_timestamp;

-- Add refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('polymarket.features_hourly',
    start_offset => INTERVAL '1 month',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists => TRUE);

-- Create stats table for pipeline monitoring
CREATE TABLE IF NOT EXISTS polymarket.pipeline_metrics (
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dag_run_id TEXT,
    event_count INTEGER,
    api_requests INTEGER,
    processing_time_ms INTEGER,
    error_count INTEGER,
    PRIMARY KEY (time, dag_run_id)
);

SELECT create_hypertable('polymarket.pipeline_metrics', 'time', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

-- Grant permissions (will be updated based on actual user)
GRANT ALL PRIVILEGES ON SCHEMA polymarket TO ${POSTGRES_USER};
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA polymarket TO ${POSTGRES_USER};
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA polymarket TO ${POSTGRES_USER};

-- Create function to generate partition name based on timestamp
CREATE OR REPLACE FUNCTION polymarket.get_partition_name(epoch_ms BIGINT)
RETURNS TEXT AS $$
DECLARE
    partition_date TEXT;
BEGIN
    partition_date := to_char(to_timestamp(epoch_ms), 'YYYY_MM');
    RETURN 'features_' || partition_date;
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- Create function to insert data with automatic feature calculation
CREATE OR REPLACE FUNCTION polymarket.insert_features(
    p_time TIMESTAMPTZ,
    p_event_start_timestamp BIGINT,
    p_start_price NUMERIC,
    p_collection_timestamp BIGINT,
    p_current_price NUMERIC,
    p_polymarket_ask_yes NUMERIC,
    p_polymarket_bid_yes NUMERIC,
    p_polymarket_ask_no NUMERIC,
    p_polymarket_bid_no NUMERIC
)
RETURNS VOID AS $$
DECLARE
    v_spread_yes NUMERIC;
    v_spread_no NUMERIC;
    v_mid_price_yes NUMERIC;
    v_mid_price_no NUMERIC;
BEGIN
    -- Calculate derived features
    v_spread_yes := p_polymarket_ask_yes - p_polymarket_bid_yes;
    v_spread_no := p_polymarket_ask_no - p_polymarket_bid_no;
    v_mid_price_yes := (p_polymarket_ask_yes + p_polymarket_bid_yes) / 2;
    v_mid_price_no := (p_polymarket_ask_no + p_polymarket_bid_no) / 2;

    -- Insert into main table
    INSERT INTO polymarket.features (
        time,
        event_start_timestamp,
        start_price,
        collection_timestamp,
        current_price,
        polymarket_ask_yes,
        polymarket_bid_yes,
        polymarket_ask_no,
        polymarket_bid_no,
        spread_yes,
        spread_no,
        mid_price_yes,
        mid_price_no
    ) VALUES (
        p_time,
        p_event_start_timestamp,
        p_start_price,
        p_collection_timestamp,
        p_current_price,
        p_polymarket_ask_yes,
        p_polymarket_bid_yes,
        p_polymarket_ask_no,
        p_polymarket_bid_no,
        v_spread_yes,
        v_spread_no,
        v_mid_price_yes,
        v_mid_price_no
    );
END;
$$ LANGUAGE plpgsql;