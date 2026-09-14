import pytest
import secrets
from pathlib import Path

from src.core import security
from src.core.config import settings


def test_load_pem_accepts_raw_and_file(tmp_path):
    raw = "-----BEGIN TEST KEY-----\nabc\n-----END TEST KEY-----"
    assert security._load_pem(raw) == raw.encode()

    p = tmp_path / "key.pem"
    p.write_text(raw)
    assert security._load_pem(str(p)) == raw.encode()


def test_load_pem_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        security._load_pem(str(tmp_path / "nope.pem"))


def test_create_access_token_no_keys_returns_dev_token(monkeypatch):
    # Ensure settings have no keys so create_access_token falls back to dev token
    monkeypatch.setattr(settings, "SECRET_KEY", "", raising=False)
    monkeypatch.setattr(settings, "PUBLIC_KEY", "", raising=False)
    # Clear any cached jwks
    monkeypatch.setattr(security, "SECRET_JWK", None, raising=False)
    monkeypatch.setattr(security, "PUBLIC_JWK", None, raising=False)

    token = security.create_access_token("subject")
    assert isinstance(token, str)
    assert token.startswith("dev-token-")


def test_ensure_jwks_raises_when_missing(monkeypatch):
    monkeypatch.setattr(settings, "SECRET_KEY", "", raising=False)
    monkeypatch.setattr(settings, "PUBLIC_KEY", "", raising=False)
    monkeypatch.setattr(security, "SECRET_JWK", None, raising=False)
    monkeypatch.setattr(security, "PUBLIC_JWK", None, raising=False)

    with pytest.raises(RuntimeError):
        security._ensure_jwks()


def test_password_hash_and_verify():
    # Use a runtime-generated password to avoid hard-coded credentials in tests
    pwd = secrets.token_urlsafe(16)
    hashed = security.get_password_hash(pwd)
    assert security.verify_password(pwd, hashed) is True
    # Verify that a different password does not validate
    assert security.verify_password(pwd + "x", hashed) is False
