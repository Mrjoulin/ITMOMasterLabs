import pytest
import unittest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from src.core import security
from src.core import config


def _generate_rsa_pair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv.decode(), pub.decode()


def test_create_and_decode_jwt_with_real_keys(monkeypatch):
    priv_pem, pub_pem = _generate_rsa_pair()
    # patch settings to provide raw PEMs
    monkeypatch.setattr(config.settings, "SECRET_KEY", priv_pem, raising=False)
    monkeypatch.setattr(config.settings, "PUBLIC_KEY", pub_pem, raising=False)
    # clear cached jwks
    monkeypatch.setattr(security, "SECRET_JWK", None, raising=False)
    monkeypatch.setattr(security, "PUBLIC_JWK", None, raising=False)

    token = security.create_access_token("me@example.com")
    assert isinstance(token, str)

    payload = security.decode_access_token(token)
    assert payload.get("sub") == "me@example.com"


def test_decode_invalid_token_raises(monkeypatch):
    priv_pem, pub_pem = _generate_rsa_pair()
    monkeypatch.setattr(config.settings, "SECRET_KEY", priv_pem, raising=False)
    monkeypatch.setattr(config.settings, "PUBLIC_KEY", pub_pem, raising=False)
    monkeypatch.setattr(security, "SECRET_JWK", None, raising=False)
    monkeypatch.setattr(security, "PUBLIC_JWK", None, raising=False)

    with pytest.raises(ValueError):
        security.decode_access_token("not-a-valid-token")
