# Testing Patterns

**Analysis Date:** 2026-04-19

## Test Framework

**Runner:**
- **pytest** (declared in `pyproject.toml` under `[project.optional-dependencies] dev`)
- Configuration file not present; default pytest settings are used.

**Assertion Library:** Built-in `assert` statements provided by pytest.

**Run Commands:**
```bash
# Run all tests via the helper script (includes env validation)
bash run_tests.sh

# Direct pytest invocation (equivalent)
pytest
```

## Test File Organization

**Location:** All test files reside in the top-level `tests/` directory.

**Naming:** Files follow the `test_*.py` pattern; individual test functions also start with `test_`.

**Structure Example (`tests/test_imports.py`):**
```python
"""Basic integration test to verify pipeline components can be imported."""

import sys
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_api_client_import():
    from api.polymarket_client import PolymarketClient, RateLimiter
    assert PolymarketClient is not None
    assert RateLimiter is not None
```

## Test Structure

**Suite Organization:** Each test file groups related import or functional checks. Tests are simple, focusing on importability, configuration loading, and basic script execution.

**Patterns:**
- Use of `sys.path.insert` to expose project modules to the test runner.
- Direct `assert` statements without additional helper libraries.
- Optional environment validation via `load_dotenv` (see `test_rate_limiting_configuration`).
- Conditional skipping when required files are missing (`pytest.skip`).

## Mocking

**Framework:** None currently used; tests rely on real modules. Future tests should employ `pytest-mock` or `unittest.mock` for external services (e.g., HTTP calls) to keep unit tests fast and deterministic.

## Fixtures and Factories

**Test Data:** Not present yet. When needed, place reusable fixtures in `tests/conftest.py` or dedicated fixture modules.

## Coverage

**Requirements:** Project-wide minimum 80% coverage as defined in global testing guidelines.

**View Coverage:** The recommended command (from common rules) is:
```bash
pytest --cov=src --cov-report=term-missing
```

## Test Types

**Unit Tests:** Verify individual functions/modules (e.g., import tests, utility functions).

**Integration Tests:** Not yet defined; future work should include end-to-end DAG runs against a test PostgreSQL instance.

**E2E Tests:** Not currently implemented.

## Common Patterns

**Async Testing:** When testing async code, use `pytest.mark.asyncio` and `await` inside the test function.

**Error Testing:** Use `with pytest.raises(ExpectedException):` pattern for exception validation.

---

*Testing analysis: 2026-04-19*