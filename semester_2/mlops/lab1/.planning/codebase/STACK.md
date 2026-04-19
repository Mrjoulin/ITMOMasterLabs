# Technology Stack

**Analysis Date:** 2026-04-19

## Languages

**Primary:**
- Python 3.11+ (required by `pyproject.toml`) – used throughout the pipeline and DAG code.

**Secondary:**
- Not detected

## Runtime

**Environment:**
- Docker image `apache/airflow:latest-python3.14` (Python 3.14 runtime in containers)

**Package Manager:**
- `uv` – fast, lockfile‑based installer used in `docker/airflow/Dockerfile`.
- Lockfile: present (`uv.lock`)

## Frameworks

**Core:**
- Apache Airflow `>=2.9.0` – orchestrates DAGs and tasks.

**Testing:**
- pytest `>=8.0.0` – unit, integration, and benchmark tests (`dev` extra).

**Build/Dev:**
- ruff `>=0.0` (configured in `pyproject.toml` under `[tool.ruff]`) – linting and import sorting.
- mypy `>=1.8.0` – static type checking.
- pre‑commit `>=3.6.0` – hook management.

## Key Dependencies

**Critical:**
- `asyncpg` – async PostgreSQL driver.
- `httpx[http2]` – async HTTP client with HTTP/2 support (used in `PolymarketClient`).
- `pydantic` – data validation and settings management.
- `structlog` – structured logging.
- `python‑dotenv` – loads environment variables.
- `prometheus-client` – metrics exposition.
- `aiocache` – async caching abstraction.
- `sqlalchemy` – ORM / core DB layer.
- `alembic` – DB migrations.
- `clickhouse-driver` – ClickHouse connectivity (present but not used in current DAGs).

**Infrastructure:**
- `timescale/timescaledb` (PostgreSQL 16) – time‑series storage, referenced in `docker-compose.yml`.
- `redis:7-alpine` – caching and Celery broker.
- `prom/prometheus` – monitoring.

## Configuration

**Environment:**
- `.env.example` (present) lists required variables such as `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POLYMARKET_API_KEY`, `AIRFLOW__API_AUTH__JWT_SECRET`, etc. Values are loaded via `python-dotenv`.

**Build:**
- Dockerfile at `docker/airflow/Dockerfile` builds the image, copies `pyproject.toml` and `uv.lock`, and runs `uv pip install .`.
- `docker-compose.yml` defines services, volumes, and resource limits.

## Platform Requirements

**Development:**
- Docker Engine + Docker Compose.
- Python 3.11+ locally for linting and testing.
- `uv` installed (`pip install uv`).

**Production:**
- Container runtime (Docker/Kubernetes) capable of running the Airflow image.
- PostgreSQL 16 (Timescale) and Redis instances.
- Prometheus for metrics collection.

---

*Stack analysis: 2026-04-19*