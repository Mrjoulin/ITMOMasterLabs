# Coding Conventions

**Analysis Date:** 2026-04-19

## Naming Patterns

**Files:**
- snake_case with `.py` extension, e.g., `polymarket_client.py`, `feature_calculator.py`
- Directories use snake_case, e.g., `airflow/dags/src/api`

**Classes:**
- PascalCase, e.g., `RateLimiter`, `PolymarketClient`, `FeatureCalculator`

**Functions / Methods:**
- snake_case, e.g., `extract_timestamp_from_url`, `calculate_spread`
- Verb‑noun phrasing, short and descriptive

**Variables:**
- snake_case, e.g., `max_requests_per_second`, `event_start_timestamp`
- Constants in upper‑case with underscores, e.g., `MAX_REQUESTS_PER_SECOND`, `DB_CONFIG`

**Types:**
- Use `typing` annotations, import from `typing` as needed, e.g., `Optional[str]`, `dict[str, Any]`

## Code Style

**Formatting:**
- `ruff` is the primary formatter/linter (`[tool.ruff.format]` enforces double quotes, LF line endings, space‑indent)
- Line length target 100 characters (`[tool.ruff] line-length = 100`), `E501` is ignored

**Linting:**
- `ruff` with selected rules: `E`, `W`, `F`, `I`, `B`, `C4`, `UP`, `T`, `PT`
- Enforces import sorting, no unused imports, bug‑bear checks, pyupgrade, and pytest‑style checks

## Import Organization

**Order:**
1. Standard library imports (e.g., `import asyncio`, `import time`)
2. Third‑party packages (e.g., `import httpx`, `import structlog`, `from tenacity import retry`)
3. Local project modules (e.g., `from config import POLYMARKET_CONFIG`, `from src.models import PolymarketEvent`)

**Path Aliases:**
- No custom path aliases; the project adds the `src` directory to `sys.path` in tests only.

## Error Handling

**Patterns:**
- Wrap external calls in `try/except` blocks, log the error with `structlog`, and either re‑raise or return a safe fallback (`None`).
- Example from `polymarket_client.py`:
```python
try:
    response = await self._client.request(...)
    response.raise_for_status()
    return response
except httpx.HTTPStatusError as e:
    logger.error("api_request_failed", status_code=e.response.status_code, url=url)
    raise
```
- Functions that may fail return `None` and log at appropriate level.

## Logging

**Framework:** `structlog`

**Patterns:**
- Create a module‑level logger: `logger = structlog.get_logger(__name__)`
- Use structured events with key/value pairs, e.g., `logger.error("failed_to_fetch_event", url=url, error=str(e))`
- Log at appropriate severity (`debug`, `info`, `warning`, `error`).

## Comments

**When to Comment:**
- Docstrings for modules, classes, and public functions/methods.
- Inline comments only when the code is non‑obvious.

**Docstrings:** Triple double‑quotes, concise description, optional Args/Returns sections (as seen in `FeatureCalculator`).

## Function Design

**Size:** Functions typically < 30 lines; complex logic split into helper methods.

**Parameters:** Typed, with defaults where sensible (e.g., `max_requests_per_second: float = 1.0`).

**Return Values:** Typed; asynchronous functions use `async def` and return `awaitable` objects; error paths return `None`.

## Module Design

**Exports:** Modules expose public classes/functions; internal helpers are prefixed with an underscore.

**Barrel Files:** Not used; imports are explicit.

---

*Convention analysis: 2026-04-19*