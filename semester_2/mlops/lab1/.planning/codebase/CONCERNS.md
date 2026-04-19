# Codebase Concerns

**Analysis Date:** 2026-04-19

## Tech Debt

**Configuration Management:**
- Issue: Configuration dictionaries (`DB_CONFIG`, `POLYMARKET_CONFIG`, etc.) are mutable globals, violating the project's immutability rule.
- Files: `airflow/config/config.py`
- Impact: Accidental runtime mutation can cause inconsistent behavior across modules.
- Fix approach: Replace mutable dicts with frozen dataclasses or `MappingProxyType` and provide accessor functions.

**Hard-coded Defaults:**
- Issue: Default passwords, admin credentials, and JWT secret are hard-coded in code and Docker Compose.
- Files: `airflow/config/config.py`, `docker/airflow/init.py`, `docker-compose.yml`
- Impact: Deployments in insecure environments may expose credentials; defaults can be unintentionally used in production.
- Fix approach: Remove defaults; require explicit environment variables and fail fast if missing. Use a secret manager for production.

## Known Bugs

**PolymarketClient.get_event_by_url:**
- Issue: References undefined variables `timestamp` and `url`; unreachable logging after a `return` statement.
- File: `airflow/dags/src/api/polymarket_client.py`
- Symptoms: Raises `NameError` when fetching an event, causing DAG runs to fail.
- Workaround: None viable; function needs correction.
- Fix approach: Extract timestamp from `slug` correctly, assign to a variable, and pass the proper URL to the `PolymarketEvent` constructor. Remove dead code after the `return`.

## Security Considerations

**Default Secrets in Docker Compose:**
- Risk: `POSTGRES_PASSWORD`, `AIRFLOW_PASSWORD`, and `AIRFLOW__API_AUTH__JWT_SECRET` default to known values (`polymarket_pass`, `admin`, `airflow_jwt_secret`).
- Files: `docker-compose.yml`
- Current mitigation: Environment variables can override, but defaults remain if vars are absent.
- Recommendations: Eliminate default values; enforce presence via validation script (`validate_setup.py`). Add secret scanning in CI.

**Empty API Key Default:**
- Risk: `POLYMARKET_CONFIG['api_key']` defaults to an empty string, allowing unauthenticated API calls which may be rate-limited or rejected.
- File: `airflow/config/config.py`
- Recommendation: Treat missing API key as a fatal configuration error and abort startup.

## Performance Bottlenecks

**Sequential Rate Limiter:**
- Issue: `RateLimiter` uses a single semaphore and lock, forcing all requests to be processed sequentially even when the configured rate allows concurrency.
- File: `airflow/dags/src/api/polymarket_client.py`
- Impact: Limits throughput; could become a bottleneck for high-frequency data collection.
- Fix approach: Use token-bucket algorithm or async sleep without locking when rate permits concurrent requests.

## Fragile Areas

**Airflow Init Script Defaults:**
- Files: `docker/airflow/init.py`
- Why fragile: Relies on environment defaults for admin user/password and DB credentials; any change in defaults requires code changes.
- Safe modification: Centralize credential handling in a single config module and reference it.
- Test coverage: No unit tests covering admin user creation logic.

## Scaling Limits

**PostgreSQL Resource Limits:**
- Current capacity: Docker Compose caps PostgreSQL memory at 4 GB and CPUs at 2.
- Limit: May be insufficient for large market data ingest.
- Scaling path: Raise Docker resource limits; consider external managed RDS for production.

## Dependencies at Risk

**tenacity library:**
- Risk: No pinned version in `pyproject.toml`; future releases could change retry semantics.
- Impact: Potential silent behavioural changes.
- Recommendation: Pin version and add integration tests for retry logic.

## Missing Critical Features

**Schema Validation for Environment Variables:**
- Problem: `validate_setup.py` only checks presence; does not enforce type/format (e.g., URL validation for `POLYMARKET_BASE_URL`).
- Blocks: Misconfiguration can cause runtime errors.

## Test Coverage Gaps

**PolymarketClient:**
- What's not tested: `get_event_by_url`, `get_order_book`, rate-limiter behavior.
- Files: `airflow/dags/src/api/polymarket_client.py`
- Risk: Undetected bugs like the undefined variables.
- Priority: High - add unit tests covering success and error paths.

---

*Concerns audit: 2026-04-19*