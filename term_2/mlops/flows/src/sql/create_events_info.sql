------- EVENTS Table ---------

CREATE TABLE IF NOT EXISTS polymarket.events_info (
    event_start_ts BIGINT PRIMARY KEY,
    start_price NUMERIC(20,8),
    close_price NUMERIC(20,8),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    event_id VARCHAR(16),
    event_slug VARCHAR(32),
    market_id VARCHAR(16),
    yes_token_id VARCHAR(128),
    no_token_id VARCHAR(128)
);
CREATE INDEX IF NOT EXISTS idx_events_info_event_start ON polymarket.events_info (event_start_ts);

-- Create trigger to auto update `updated_at` field when row updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trigger_update_updated_at
    BEFORE UPDATE ON polymarket.events_info
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
