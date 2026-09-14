import os
import pytest

import security.keys as keys


def test_get_private_key_from_env(monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", "raw-private")
    assert keys.get_private_key() == "raw-private"


def test_get_private_key_from_path(tmp_path, monkeypatch):
    p = tmp_path / "priv.pem"
    p.write_text("file-private")
    monkeypatch.delenv("PRIVATE_KEY", raising=False)
    monkeypatch.setenv("PRIVATE_KEY_PATH", str(p))
    assert keys.get_private_key() == "file-private"


def test_get_private_key_missing_raises(monkeypatch):
    monkeypatch.delenv("PRIVATE_KEY", raising=False)
    monkeypatch.delenv("PRIVATE_KEY_PATH", raising=False)
    with pytest.raises(FileNotFoundError):
        keys.get_private_key()


def test_get_public_key_from_env(monkeypatch):
    monkeypatch.setenv("PUBLIC_KEY", "raw-public")
    assert keys.get_public_key() == "raw-public"


def test_get_public_key_from_path(tmp_path, monkeypatch):
    p = tmp_path / "pub.pem"
    p.write_text("file-public")
    monkeypatch.delenv("PUBLIC_KEY", raising=False)
    monkeypatch.setenv("PUBLIC_KEY_PATH", str(p))
    assert keys.get_public_key() == "file-public"


def test_get_public_key_missing_raises(monkeypatch):
    monkeypatch.delenv("PUBLIC_KEY", raising=False)
    monkeypatch.delenv("PUBLIC_KEY_PATH", raising=False)
    with pytest.raises(FileNotFoundError):
        keys.get_public_key()
