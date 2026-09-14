import sys
import os
from unittest.mock import Mock

import pytest

from fastapi.testclient import TestClient

# Add the backend directory to the Python path so tests can import src and main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_session():
	"""Reusable Mock for SQLModel Session-like behavior.

	Tests can configure return values on `mock_session.exec` and `mock_session.get`.
	"""
	return Mock()


@pytest.fixture
def user():
	from src.models.userinfo import UserInfo

	return UserInfo(id=1, email="test@example.com", full_name="Test User")


@pytest.fixture
def rsa_keys():
	# Generate ephemeral RSA keypair for tests that need real PEMs
	from cryptography.hazmat.primitives.asymmetric import rsa
	from cryptography.hazmat.primitives import serialization

	key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
	priv = key.private_bytes(
		encoding=serialization.Encoding.PEM,
		format=serialization.PrivateFormat.PKCS8,
		encryption_algorithm=serialization.NoEncryption(),
	).decode()
	pub = key.public_key().public_bytes(
		encoding=serialization.Encoding.PEM,
		format=serialization.PublicFormat.SubjectPublicKeyInfo,
	).decode()
	return priv, pub


@pytest.fixture
def client(mock_session):
	"""TestClient with `get_session` dependency overridden to yield `mock_session`.

	Use this for lightweight endpoint tests that need a mocked DB session.
	"""
	import importlib

	import main as main_mod
	from src.core import database as database_mod

	def _override_get_session():
		yield mock_session

	main_mod.app.dependency_overrides[database_mod.get_session] = _override_get_session
	client = TestClient(main_mod.app)
	yield client
	# Clean up override after test
	main_mod.app.dependency_overrides.pop(database_mod.get_session, None)

