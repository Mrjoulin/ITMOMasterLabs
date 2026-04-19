# External Integrations

**Analysis Date:** 2026-04-19

## APIs & External Services

**Market Data:**
- Polymarket API - provides event, order book, and price data.
  - SDK/Client: Custom `PolymarketClient` defined in `airflow/dags/src/api/polymarket_client.py` using `httpx`.
  - Auth: Bearer token supplied via `POLYMARKET_API_KEY` environment variable (referenced in `docker-compose.yml` and DAG code).

**Observability:**
- Prometheus - metrics collection for Airflow and custom pipeline metrics.
  - Client: `prometheus-client` Python library.
  - Endpoint: Exposed by the `prometheus` service in `docker-compose.yml` on port `9090`.

## Data Storage

**Databases:**
- TimescaleDB (PostgreSQL 16) - time‑series storage for market events and analytics.
  - Connection string built from `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` env vars (see `docker-compose.yml`).
  - Client: `asyncpg` for async queries, `sqlalchemy` for ORM models.

**Cache:**
- Redis (7‑alpine) - Celery broker/result backend and lightweight cache.
  - Connection: `redis://:@redis:6379/0` defined in `docker-compose.yml`.
  - Client: `aiocache` (configured in code) and Airflow Celery settings.

**File Storage:**
- Local filesystem only (DAG files, logs, and Airflow plugins are mounted via Docker volumes).

## Authentication & Identity

**Auth Provider:**
- Custom token‑based auth for Polymarket API (Bearer token). No third‑party identity provider.
- Airflow API JWT authentication configured with `AIRFLOW__API_AUTH__JWT_SECRET` and `AIRFLOW__API_AUTH__JWT_ISSUER` env vars.

## Monitoring & Observability

**Error Tracking:**
- Not detected (no external Sentry or similar service).

**Logs:**
- Structured logging via `structlog` in Python code.
- Airflow logs are written to the `airflow/logs` volume and can be scraped by Prometheus if configured.

## CI/CD & Deployment

**Hosting:**
- Docker containers orchestrated via `docker-compose.yml`. No external CI/CD platform detected in the repository.

**CI Pipeline:**
- Not detected (no GitHub Actions or other CI config files).

## Environment Configuration

**Required env vars:**
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- `POLYMARKET_API_KEY`
- `AIRFLOW__API_AUTH__JWT_SECRET`, `AIRFLOW__API_AUTH__JWT_ISSUER`
- `AIRFLOW_UID` (optional, defaults to 50000)
- `ENV_FILE_PATH` (optional, points to `.env` file)

**Secrets location:**
- Expected to be provided via a `.env` file (referenced in `docker-compose.yml`) or injected as environment variables in the deployment environment.

## Webhooks & Callbacks

**Incoming:**
- None detected.

**Outgoing:**
- Polymarket API calls are outbound HTTP requests from `PolymarketClient`.

---

*Integration audit: 2026-04-19*