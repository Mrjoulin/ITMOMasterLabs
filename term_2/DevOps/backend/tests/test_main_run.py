import runpy
import sys
import unittest


def test_main_runs_uvicorn(monkeypatch):
    """Run the project module as __main__ while stubbing out uvicorn.run.

    This covers the guarded __main__ block without starting a real server.
    """
    # Ensure we reload the module freshly
    if "main" in sys.modules:
        del sys.modules["main"]

    called = {}

    try:
        import uvicorn

        def fake_run(*args, **kwargs):
            called["args"] = args
            called["kwargs"] = kwargs

        monkeypatch.setattr(uvicorn, "run", fake_run)
    except Exception:
        # If uvicorn is not available as a module in the test env, insert
        # a fake module into sys.modules with a run attribute.
        import types

        uv = types.SimpleNamespace()

        def fake_run(*args, **kwargs):
            called["args"] = args
            called["kwargs"] = kwargs

        uv.run = fake_run
        sys.modules["uvicorn"] = uv

    # Provide deterministic host/port via env
    monkeypatch.setenv("APP_HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "4242")

    # Execute module as script
    runpy.run_module("main", run_name="__main__")

    assert "kwargs" in called
    # main.py sets host/port from env and passes them to uvicorn.run
    assert called["kwargs"].get("host") == "127.0.0.1"
    assert int(called["kwargs"].get("port")) == 4242
