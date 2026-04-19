# Architecture

**Analysis Date:** 2026-04-19

## Pattern Overview

**Overall:** Modular async data‑pipeline built on Apache Airflow DAGs with a clean separation between configuration, API integration, database access, domain models, and feature‑processing logic.

**Key Characteristics:**
- Asynchronous I/O for external API calls.
- Connection pooling for high‑throughput PostgreSQL/TimescaleDB writes.
- Rate‑limiting and retry logic wrapped around external services.
- Declarative DAG using Airflow `@dag` and `@task` decorators.

## Layers

**Configuration:**
- Purpose: Centralised environment‑driven settings.
- Location: `airflow/dags/config/config.py`
- Contains: runtime limits, DB/Redis credentials, API endpoints, performance tuning.
- Depends on: OS environment variables.
- Used by: All other layers.

**DAG Definitions (Orchestration Layer):**
- Purpose: Declare workflow, schedule, and task dependencies.
- Location: `airflow/dags/simple_dag.py`, `airflow/dags/polymarket_btc_pipeline.py`
- Contains: Airflow DAG objects, task functions.
- Depends on: Configuration, API client, DB client, Processors.
- Used by: Airflow scheduler.

**API Integration Layer:**
- Purpose: Communicate with Polymarket HTTP API with rate‑limiting and retries.
- Location: `airflow/dags/src/api/polymarket_client.py`
- Contains: `PolymarketClient`, `RateLimiter`.
- Depends on: `config.POLYMARKET_CONFIG`, `MAX_REQUESTS_PER_SECOND`.
- Used by: Extraction task in DAG.

**Database Access Layer:**
- Purpose: High‑performance async PostgreSQL/TimescaleDB interaction.
- Location: `airflow/dags/src/db/postgresql_client.py`
- Contains: `PostgreSQLClient` with connection pool and bulk insert helpers.
- Depends on: `config.DB_CONFIG`, `config.PERFORMANCE_CONFIG`.
- Used by: Load task in DAG and by feature calculators that need recent price data.

**Domain Models Layer:**
- Purpose: Typed data contracts for events, order books, and processed features.
- Location: `airflow/dags/src/models/polymarket_data.py`
- Contains: `PolymarketEvent`, `PolymarketOrderBook`, `ProcessedFeatures`, `PipelineMetrics`.
- Depends on: `pydantic` for validation.
- Used by: API client, processors, DAG tasks.

**Feature Processor Layer:**
- Purpose: Derive business‑logic metrics from raw data.
- Location: `airflow/dags/src/processors/feature_calculator.py`
- Contains: `FeatureCalculator` with spread, VWAP, volatility, momentum, etc.
- Depends on: `PostgreSQLClient` for historical price queries.
- Used by: Transform task in DAG.

**Validators / Utils (Supporting Layer):**
- Purpose: Input validation and helper functions (currently minimal).
- Location: `airflow/dags/src/validators/` (empty) and `airflow/dags/utils/`.
- Depends on: None currently.

## Data Flow

**Polymarket BTC Pipeline:**
1. **Extract** – `polymarket_btc_pipeline.extract` calls `PolymarketClient` to fetch event, order‑book, and BTC price concurrently.
2. **Transform** – `polymarket_btc_pipeline.transform` builds `ProcessedFeatures` then enriches them via `FeatureCalculator` (price change, VWAP, volatility, etc.).
3. **Load** – `polymarket_btc_pipeline.load` uses `PostgreSQLClient.insert_batch` to bulk‑insert into TimescaleDB hypertable.

## Key Abstractions

**PolymarketClient:** Async wrapper with built‑in rate limiter and tenacity retries.
**PostgreSQLClient:** Async connection‑pool manager exposing bulk‑insert and recent‑price queries.
**FeatureCalculator:** Stateless service object that computes derived metrics, optionally enriching a dict of base features.

## Entry Points

**Simple Example DAG:** `airflow/dags/simple_dag.py` – basic BashOperator workflow.
**Polymarket BTC Pipeline DAG:** `airflow/dags/polymarket_btc_pipeline.py` – production data pipeline, scheduled via `DAG_CONFIG["schedule_interval"]`.

## Error Handling

- API errors are captured and logged via `structlog`; retries are handled by `tenacity`.
- Database errors propagate to the DAG task, causing Airflow to mark the task as failed and trigger retries per `DAG_CONFIG`.
- Validation errors in Pydantic models raise `ValidationError` early, preventing bad data from entering the pipeline.

## Cross‑Cutting Concerns

**Logging:** `structlog` is used throughout (`logger = structlog.get_logger(__name__)`).
**Rate Limiting:** Centralised in `RateLimiter` (max 1 req/sec, configurable via env).
**Configuration:** All runtime values come from environment variables through `config.py`.
**Observability:** Metrics emitted via `structlog` and Prometheus ports defined in `MONITORING_CONFIG`.

---

*Architecture analysis: 2026-04-19*