import sys
import importlib
import unittest
from types import SimpleNamespace


def test_import_main_does_not_call_uvicorn(monkeypatch):
    # Ensure a fresh import of main
    if "main" in sys.modules:
        del sys.modules["main"]

    called = {"called": False}

    def fake_run(*args, **kwargs):
        called["called"] = True

    # Inject a fake uvicorn module that would mark if run() is called
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))

    import main as main_mod
    importlib.reload(main_mod)

    assert hasattr(main_mod, "app")
    # Importing main must not trigger uvicorn.run
    assert called["called"] is False
