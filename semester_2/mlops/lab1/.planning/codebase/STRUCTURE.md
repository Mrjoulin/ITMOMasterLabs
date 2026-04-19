# Codebase Structure

**Analysis Date:** 2026-04-19

## Directory Layout

```
[project-root]/
├── airflow/                     # Airflow runtime configuration
│   └── dags/                    # DAG definitions and pipeline code
│       ├── __init__.py
│       ├── simple_dag.py        # Tutorial/example DAG
│       ├── polymarket_btc_pipeline.py  # Production pipeline DAG
│       ├── config/               # Central configuration module
│       │   ├── __init__.py
│       │   └── config.py
│       ├── src/                  # Core Python package for the pipeline
│       │   ├── __init__.py
│       │   ├── api/              # External API client layer
│       │   │   ├── __init__.py
│       │   │   └── polymarket_client.py
│       │   ├── db/               # Async PostgreSQL/TimescaleDB client
│       │   │   ├── __init__.py
│       │   │   └── postgresql_client.py
│       │   ├── models/           # Pydantic data models
│       │   │   ├── __init__.py
│       │   │   └── polymarket_data.py
│       │   ├── processors/       # Feature calculation & enrichment
│       │   │   ├── __init__.py
│       │   │   └── feature_calculator.py
│       │   └── validators/       # Input validation helpers (currently empty)
│       └── utils/                # Miscellaneous utilities (currently empty)
├── docker/                     # Container definitions
│   ├── airflow/Dockerfile
│   └── postgres/ ...
├── tests/                      # Test suite (pytest based)
├── .planning/                  # Auto‑generated planning docs (this directory)
│   └── codebase/                # Contains ARCHITECTURE.md, STRUCTURE.md, etc.
├── pyproject.toml               # Project metadata and dependencies
├── uv.lock                      # UV lockfile
├── docker-compose.yml           # Docker compose for Airflow & Postgres
├── main.py                      # Placeholder script (not part of pipeline)
└── README.md                    # Project overview
```

## Directory Purposes

**`airflow/`** – Holds all Airflow‑related files. The `dags/` subdirectory is the only folder Airflow scans for DAG definitions.

**`airflow/dags/config/`** – Centralised configuration accessed via `from config import …`. Keeps secrets out of code; values are pulled from environment variables.

**`airflow/dags/src/`** – Implements the business logic of the pipeline.
- **`api/`** – Asynchronous HTTP client for Polymarket, encapsulated in `PolymarketClient` (`airflow/dags/src/api/polymarket_client.py`).
- **`db/`** – Async PostgreSQL client with connection pooling (`airflow/dags/src/db/postgresql_client.py`).
- **`models/`** – Typed Pydantic models representing events, order books, processed features, and metrics (`airflow/dags/src/models/polymarket_data.py`).
- **`processors/`** – Feature engineering utilities (`airflow/dags/src/processors/feature_calculator.py`).
- **`validators/`** – Intended place for custom validation helpers (currently empty).

**`docker/`** – Dockerfiles and ancillary scripts for containerised development and deployment.

**`tests/`** – pytest test suite (unit, integration, and potential e2e tests).

## Key File Locations

**Entry Points:**
- ``airflow/dags/simple_dag.py`` – Simple example DAG.
- ``airflow/dags/polymarket_btc_pipeline.py`` – Main production DAG.

**Configuration:** ``airflow/dags/config/config.py`` – All runtime settings.

**Core Logic:**
- API client: ``airflow/dags/src/api/polymarket_client.py``
- DB client: ``airflow/dags/src/db/postgresql_client.py``
- Models: ``airflow/dags/src/models/polymarket_data.py``
- Feature calculator: ``airflow/dags/src/processors/feature_calculator.py``

## Naming Conventions

**Files:** snake_case with `.py` extension (e.g., `polymarket_client.py`).
**Directories:** singular snake_case matching the layer purpose (e.g., `api`, `db`, `models`).
**Classes:** PascalCase (e.g., `PolymarketClient`).
**Functions / Methods:** snake_case; async functions suffixed with `_async` only when the name would clash with a sync counterpart (not used here).

## Where to Add New Code

- **New DAG:** Place a new `.py` file under `airflow/dags/` and import shared config/models as needed.
- **New API integration:** Add a module under `airflow/dags/src/api/` and expose it via `__init__.py`.
- **New database entity:** Extend `airflow/dags/src/models/` with a new Pydantic model and add CRUD helpers in `airflow/dags/src/db/`.
- **New feature calculation:** Add a method to `FeatureCalculator` or create a new processor module under `airflow/dags/src/processors/`.
- **Tests:** Create a corresponding test file under `tests/` mirroring the source path, e.g., `tests/src/api/test_polymarket_client.py`.

## Special Directories

**`airflow/dags/utils/`** – Holds utility scripts that are not part of the core pipeline (currently empty). Generated files are not committed.
**`tests/`** – Fully committed test suite; follows the same package layout as `src/` for easy import.

---

*Structure analysis: 2026-04-19*