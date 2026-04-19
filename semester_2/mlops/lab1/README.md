# Polymarket 5-Minute BTC Data Pipeline

Ultra-fast data pipeline for collecting Polymarket BTC price prediction data for 5-minute intervals. Built for speed and reliability with TimescaleDB, Airflow, and PostgreSQL.

## 🚀 Features

- **5-minute real-time data collection**: Automated DAG runs every 5 minutes
- **Ultra-fast inserts**: Async PostgreSQL with connection pooling and batch inserts
- **Time-series optimization**: TimescaleDB hypertable with 1-hour chunks
- **Precision rate limiting**: Max 1 req/sec, min 100ms interval compliance
- **Ordered partitioning**: Data partitioned by event timestamps for query performance
- **Derived features**: Auto-calculated spreads, mid-prices, VWAP, momentum
- **Full monitoring**: Prometheus metrics and Airflow observability

## 📋 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Airflow Scheduler (5min)                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │ Extract                     │
        │ - Polymarket API            │
        │ - Rate limiting (1 req/s)   │
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │ Transform                   │
        │ - Feature calculation       │
        │ - Spread & mid-price        │
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │ Load                        │
        │ - Async PostgreSQL          │
        │ - Batch inserts             │
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │ TimescaleDB                 │
        │ - Hourly chunks             │
        │ - Auto-partitioning         │
        └───────────────────────────────┘
```

## 🛠️ Prerequisites

- Docker & Docker Compose
- Python 3.11+
- UV package manager (`pip install uv`)
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

## ⚡ Quick Start

### 1. Clone & Setup

```bash
git clone <repository-url>
cd polymarket-pipeline
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` with your Polymarket API key:

```bash
# Database
POSTGRES_USER=polymarket
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=polymarket

# Polymarket API (required)
POLYMARKET_API_KEY=your_api_key_here

# Airflow
AIRFLOW_USER=admin
AIRFLOW_PASSWORD=your_airflow_password
```

### 3. Build & Run

```bash
# Fast setup with uv
uv sync

# Start entire stack
docker compose up -d

# Follow logs
docker compose logs -f airflow-init
```

### 4. Verify Deployment

```bash
# Wait for initialization (~2-3 minutes)
# Check Airflow UI at http://localhost:8080 (login: admin/your_password)
# Check Prometheus at http://localhost:9090

# Verify PostgreSQL connection
docker compose exec postgres psql -U polymarket -d polymarket -c "\dt polymarket.*"

# Check DAG status
docker compose exec airflow-webserver airflow dags list
```

## 📊 Data Schema

### Main Features Table

```sql
CREATE TABLE polymarket.features (
    time TIMESTAMPTZ NOT NULL,              -- Collection time
    event_start_timestamp BIGINT NOT NULL,  -- 5-min interval start
    start_price NUMERIC(20,8),              -- Baseline price
    collection_timestamp BIGINT,            -- Collection timestamp (ms)
    current_price NUMERIC(20,8),           -- Current BTC price
    polymarket_ask_yes NUMERIC(10,4),       -- YES ask (0-1)
    polymarket_bid_yes NUMERIC(10,4),     -- YES bid (0-1)
    polymarket_ask_no NUMERIC(10,4),       -- NO ask (0-1)
    polymarket_bid_no NUMERIC(10,4),       -- NO bid (0-1)
    spread_yes NUMERIC(10,4),               -- YES spread
    spread_no NUMERIC(10,4),               -- NO spread
    mid_price_yes NUMERIC(10,4),          -- YES mid-price
    mid_price_no NUMERIC(10,4),           -- NO mid-price
    price_change_1m NUMERIC(10,4),         -- 1-min price change (%)
    price_change_5m NUMERIC(10,4),         -- 5-min price change (%)
    volume_weighted_price NUMERIC(20,8)     -- VWAP
) USING TIMESCALEDB;
```

### Continuous Aggregates (Hourly)

```sql
-- Materialized view for hourly aggregations
SELECT * FROM polymarket.features_hourly WHERE bucket >= NOW() - INTERVAL '24 hours';
```

## 🚀 Performance Tuning

The pipeline is optimized for **maximum speed**:

- **PostgreSQL**: Shared buffers 2GB, synchronous_commit=off, JIT enabled
- **TimescaleDB**: 1-hour chunks for ultra-fast queries
- **Batch inserts**: 1000 records per batch, async executemany
- **Connection pooling**: Min 10, max 20 connections
- **HTTP/2**: Single connection for all API calls
- **Prepared statements**: Cached SQL plans

## 🔍 Monitoring

### Metrics Available

- `polymarket_pipeline_events_total`: Events processed per run
- `polymarket_pipeline_api_requests_total`: API calls made
- `polymarket_pipeline_processing_time_ms`: Processing duration
- `polymarket_pipeline_errors_total`: Error count

### Grafana Dashboard (Optional)

```bash
# Add Grafana to docker-compose.yml if needed
docker compose up -d grafana
# Import dashboard dashboard.json
```

## 📈 Query Examples

### Get data for specific interval

```sql
SELECT * FROM polymarket.features
WHERE event_start_timestamp = 1772731200
ORDER BY time DESC
LIMIT 100;
```

### Hourly aggregation

```sql
SELECT
    time_bucket('1 hour', time) AS hour,
    AVG(current_price) as avg_price,
    AVG(spread_yes) as avg_spread_yes,
    AVG(mid_price_yes) as avg_mid_price_yes,
    COUNT(*) as data_points
FROM polymarket.features
WHERE time >= NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour DESC;
```

### Compressed data (older than 24h)

```sql
SELECT * FROM polymarket.features
WHERE time < NOW() - INTERVAL '24 hours'
ORDER BY time DESC;
```
---
Automatically compressed by TimescaleDB

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| postgres | 5432 | TimescaleDB with hypertable |
| redis | 6379 | Celery message broker |
| airflow-webserver | 8080 | Airflow web UI |
| airflow-scheduler | - | DAG scheduler |
| airflow-worker | - | Task executor |
| airflow-triggerer | - | Deferrable operators |
| prometheus | 9090 | Metrics collection |

## 🎯 Testing

```bash
# Run pipeline tests
uv run pytest tests/ -v

# Manual DAG trigger
docker compose exec airflow-webserver airflow dags trigger polymarket_btc_5m_pipeline

# Check specific DAG run
docker compose exec airflow-webserver airflow dags list-runs -d polymarket_btc_5m_pipeline
```

## 🔧 Manual DAG Run

1. Open Airflow UI: http://localhost:8080
2. Login with credentials from `.env`
3. Find `polymarket_btc_5m_pipeline`
4. Click ▶️ to trigger manually
5. Monitor in Graph View or Grid View

## ⚠️ Troubleshooting

### DAG not appearing

```bash
# Restart scheduler to pick up new DAG
docker compose restart airflow-scheduler

# Check DAG parse errors
docker compose exec airflow-scheduler python -m airflow dags list-import-errors
```

### Database connection issues

```bash
# Check PostgreSQL logs
docker compose logs postgres

# Verify connection
docker compose exec airflow-worker python -c "import asyncpg; asyncio.run(asyncpg.connect(password='...'))"
```

### Rate limit errors

- Verify `POLYMARKET_API_KEY` in `.env`
- Check logs: `docker compose logs -f airflow-worker`
- API limited to 1 req/sec with 100ms min interval (per Polymarket docs)

### Slow performance

```bash
# Check resource usage
docker stats

# PostgreSQL tuning already applied in postgresql.conf
# For more speed, increase WORKER_CONCURRENCY in .env
```

## 📦 Project Structure

```
.
├── dags/
│   └── polymarket_btc_pipeline.py    # Main Airflow DAG
├── docker/
│   ├── postgres/
│   │   ├── init.sql                  # TimescaleDB setup
│   │   └── postgresql.conf           # Performance tuning
│   └── airflow/
│       └── Dockerfile                # Optimized Airflow image
├── src/
│   ├── api/
│   │   └── polymarket_client.py     # API client + rate limiting
│   ├── db/
│   │   └── postgresql_client.py      # Async DB client
│   ├── models/
│   │   └── polymarket_data.py        # Data models
│   └── processors/
│       └── feature_calculator.py       # Feature engineering
├── config/
│   └── config.py                     # Centralized config
├── monitoring/
│   └── prometheus.yml                 # Metrics config
├── migrations/                        # Database migrations
├── tests/                           # Unit & integration tests
├── docker-compose.yml               # Full stack definition
├── pyproject.toml                   # UV package management
└── .env.example                     # Environment template
```

## 🔒 Security

- **Secrets**: Never commit `.env` file (already in `.gitignore`)
- **Passwords**: Generate strong passwords for production
- **Network**: Services isolated in `polymarket-network`
- **API keys**: Rotate periodically and use environment variables

## 📊 Data Retention

- **Raw data**: 90 days (automatic deletion by TimescaleDB policy)
- **Compressed data**: Retained permanently
- **Aggregated**: Hourly aggregates stored indefinitely

## 📝 API Rate Limits

Per Polymarket documentation:
- Maximum: 1 request per second
- Minimum: 100ms between requests
- Pipeline automatically enforces these limits

## 🤝 Contributing

```bash
# Development setup
uv sync
cd docker
docker compose up -d postgres redis  # Start dependencies

# Run pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files
```

## 📄 License

MIT License - see LICENSE file

## 📞 Support

- GitHub Issues: [Create issue](https://github.com/your-org/polymarket-pipeline/issues)
- Polymarket API Docs: [API Reference](https://docs.polymarket.com/api-reference/introduction)

## 🎯 Performance Targets

- **Data collection**: < 10 seconds per 5-minute interval
- **API requests**: 100ms - 1000ms per request (Python rate limiter)
- **Database inserts**: > 10,000 records/second with batching
- **Query latency**: < 100ms for time-range queries
- **Airflow overhead**: < 30 seconds total DAG runtime