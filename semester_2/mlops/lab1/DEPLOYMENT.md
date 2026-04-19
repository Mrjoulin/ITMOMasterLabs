# Polymarket Pipeline - Quick Deployment Guide

## 🚀 Deployment Steps

### 1. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env with your API key
nano .env

# Required settings:
POLYMARKET_API_KEY=your_api_key_here
POSTGRES_PASSWORD=secure_random_password
AIRFLOW_PASSWORD=secure_random_password
```

### 2. Install Dependencies

```bash
# Install UV if not already installed
pip install uv

# Create lock file and install all dependencies
uv sync
```

This will create `uv.lock` and install:
- apache-airflow[postgres] (2.9.0)
- asyncpg (0.30.0) - async PostgreSQL
- httpx[http2] (0.27.0) - async HTTP client
- pydantic (2.5+) - data validation
- timescaledb (PostgreSQL extension)
- All other dependencies from pyproject.toml

### 3. Validate Setup

```bash
# Run validation script
python validate_setup.py
```

This will check:
- Environment variables configured
- Docker compose file structure
- Source code files present
- DAG syntax valid
- Package imports working
- API connectivity (if key provided)

### 4. Build Docker Images

```bash
# Build all images (takes 3-5 minutes)
docker compose build

# Verify images created
docker images | grep polymarket
```

### 5. Start the Stack

```bash
# Start all services in detached mode
docker compose up -d

# Follow initialization logs
docker compose logs -f airflow-init

# Wait for "airflow-init" to complete (~2-3 minutes)
# You should see: "airflow users create ... Admin User ..."
```

### 6. Verify Services

Check all services are running:

```bash
docker compose ps

# Expected output:
# NAME                           STATUS
# polymarket-postgres            healthy
# polymarket-redis               healthy
# polymarket-airflow-init        exited (0)
# polymarket-airflow-scheduler   healthy
# polymarket-airflow-webserver   healthy
# polymarket-airflow-worker      healthy
# polymarket-prometheus          healthy
```

### 7. Initialize Database

The TimescaleDB hypertable is automatically created by `docker/postgres/init.sql`.

Verify creation:

```bash
docker compose exec postgres psql -U polymarket -d polymarket -c "\dx"

# Should show:
# - timescaledb (extension)
# - plpgsql (default)
```

Check table structure:

```bash
docker compose exec postgres psql -U polymarket -d polymarket -c "\dt polymarket.*"

# Should show:
# polymarket.features (hypertable)
# polymarket.features_hourly (materialized view)
# polymarket.pipeline_metrics (hypertable)
```

### 8. Access Airflow Web UI

Open browser to: http://localhost:8080

Login with credentials from `.env`:
- Username: `admin` (or AIRFLOW_USER value)
- Password: your AIRFLOW_PASSWORD

### 9. Verify DAG in Airflow

In Airflow UI:
1. **DAGs** page should show `polymarket_btc_5m_pipeline`
2. Initially it's **Paused** (catchup=False prevents automatic runs)
3. Click the ⏯️ pause button to **Unpause** it
4. Schedule is `*/5 * * * *` (every 5 minutes)

### 10. Run First Manual Trigger

Trigger a manual run to test:

```bash
# Via Airflow UI:
# 1. Click DAG name
# 2. Click Graph View
# 3. Click ▶️ "Trigger DAG"

# Or via CLI:
docker compose exec airflow-webserver airflow dags trigger polymarket_btc_5m_pipeline
```

Monitor the run:
- **Grid View** shows historical runs
- **Graph View** shows task dependencies
- **Logs** for each task (extract, transform, load)

### 11. Verify Data Flow

Check if data is being collected:

```bash
# Query the database
docker compose exec postgres psql -U polymarket -d polymarket

# In psql:
SELECT COUNT(*) FROM polymarket.features;
SELECT time, event_start_timestamp, current_price FROM polymarket.features ORDER BY time DESC LIMIT 10;
```

Expected: Count increases every 5 minutes after first successful run

### 12. Monitor Metrics

Open Prometheus: http://localhost:9090

Try these queries:

```promql
# Events processed per DAG run
polymarket_pipeline_events_total

# API requests made
polymarket_pipeline_api_requests_total

# Processing time
polymarket_pipeline_processing_time_ms

# Errors
polymarket_pipeline_errors_total
```

### 13. Verify Rate Limiting

Check worker logs to confirm rate limiting:

```bash
docker compose logs -f airflow-worker | grep "rate"

# Should see:
# "acquired rate limit" with delays between 100-1000ms
```

## ✅ Deployment Validation Checklist

- [ ] All containers running: `docker compose ps`
- [ ] Airflow UI accessible at localhost:8080
- [ ] Polaris API key configured in .env
- [ ] DAG visible in Airflow UI
- [ ] Database hypertable created: `\dt polymarket.*`
- [ ] First manual DAG run completed successfully
- [ ] Data appearing in polymarket.features table
- [ ] Prometheus metrics visible at localhost:9090
- [ ] Rate limiting delays visible in logs (100-1000ms)
- [ ] No errors in airflow-worker logs

## 🔧 Troubleshooting

### Docker service fails to start

```bash
docker compose logs <service-name>  # Check logs
docker compose restart <service-name>  # Restart service
```

### Database connection fails

```bash
# Check PostgreSQL is ready
docker compose exec postgres pg_isready -U polymarket

# Verify connection from Airflow
docker compose exec airflow-worker python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect(
        host='postgres',
        port=5432,
        user='polymarket',
        password='your_password',
        database='polymarket'
    )
    print('Connection successful')
    await conn.close()
asyncio.run(test())"
```

### API authentication fails

```bash
# Test API key
export POLYMARKET_API_KEY=<your-key>
python -c "
import asyncio
from src.api.polymarket_client import PolymarketClient
async def test():
    async with PolymarketClient() as client:
        price = await client.get_current_btc_price()
        print(f'API working, BTC price: {price}')
asyncio.run(test())"
```

### DAG not appearing

```bash
# Check DAG import errors
docker compose exec airflow-webserver airflow dags list-import-errors

# Restart scheduler
docker compose restart airflow-scheduler
```

### High processing latency

```bash
# Check resource usage
docker stats

# Increase resources in docker-compose.yml
docker compose up -d  # Recreate with new limits
```

## 🎯 Performance Benchmarks

Expected performance after deployment:

| Metric | Target | Monitoring Method |
|--------|--------|-------------------|
| Total Pipeline Runtime | < 30s | Airflow UI |
| API Request Speed | 100-1000ms | Logs |
| Database Insert Rate | > 10k rows/sec | Calculation |
| Query Latency (1h data) | < 100ms | Prometheus |
| Memory Usage | < 2GB per service | Docker stats |

## 📊 Next Steps After Deployment

1. **Let it run for 24 hours** to accumulate baseline data
2. **Analyze first aggregation**: Check `polymarket.features_hourly` MV
3. **Tune Grafana dashboard** (optional) for visualization
4. **Set up alerts** in Prometheus for failures
5. **Adjust batch size** based on actual data volume
6. **Monitor compression** after 24 hours (data auto-compresses)

## 🔐 Security

- API keys stored only in `.env` (never committed)
- PostgreSQL password in `.env` (strong random string)
- Airflow web UI password protected
- All services in private Docker network
- No external access to PostgreSQL (port 5432 not exposed to host by default)