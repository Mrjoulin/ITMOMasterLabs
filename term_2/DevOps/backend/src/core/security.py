from datetime import datetime, timedelta, timezone
from typing import Any, Union
import uuid

from passlib.context import CryptContext
from pathlib import Path

from src.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# JWT client and JWKs are created lazily to avoid importing cryptography at
# module import time (which can cause PyO3/cryptography initialization errors
# during test collection).
_jwt_client = None

ALGORITHM = "RS256"
ACCESS_TOKEN_EXPIRE_HOURS = 168


def _load_pem(value: str) -> bytes:
    """Accept either raw PEM content or a filesystem path to a PEM file."""
    if value.strip().startswith("-----BEGIN"):
        return value.encode()

    p = Path(value)
    if p.exists():
        return p.read_bytes()

    raise FileNotFoundError(f"Key not found at {value}")


SECRET_JWK = None
PUBLIC_JWK = None


def _ensure_jwks() -> None:
    """Lazily load JWKs from settings. Raises RuntimeError when keys are missing.

    Loading is deferred so test collection/import-time doesn't require configured
    secrets or the `cryptography` Rust extension being initialized.
    """
    global SECRET_JWK, PUBLIC_JWK, _jwt_client
    if SECRET_JWK is not None and PUBLIC_JWK is not None and _jwt_client is not None:
        return

    if not settings.SECRET_KEY or not settings.PUBLIC_KEY:
        # Leave uninitialized; calling code should handle missing keys or tests may patch
        raise RuntimeError("JWT keys are not configured. Set SECRET_KEY and PUBLIC_KEY environment variables.")

    # Import jwt and the jwk loader only when keys are required.
    from jwt import JWT, jwk_from_pem as _jwk_from_pem

    _jwt_client = JWT()
    SECRET_JWK = _jwk_from_pem(_load_pem(settings.SECRET_KEY))
    PUBLIC_JWK = _jwk_from_pem(_load_pem(settings.PUBLIC_KEY))


def create_access_token(
    subject: Union[str, Any], expires_delta: timedelta = None
) -> str:
    try:
        _ensure_jwks()
    except RuntimeError:
        # In test / dev environments where RSA keys are not provided, return a
        # harmless opaque token. Tests should mock `decode_access_token` when
        # they need to validate JWT content.
        return f"dev-token-{uuid.uuid4().hex}"

    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode = {"exp": int(expire.timestamp()), "sub": str(subject)}
    encoded_jwt = _jwt_client.encode(to_encode, SECRET_JWK, alg=ALGORITHM)
    return encoded_jwt


def decode_access_token(access_token: str) -> Any:
    # _ensure_jwks will raise RuntimeError if keys are missing; let that propagate
    # so callers/tests can handle or mock accordingly.
    _ensure_jwks()
    try:
        message_received = _jwt_client.decode(access_token, PUBLIC_JWK, do_time_check=True)
        return message_received
    except Exception as e:
        # Import jwt exceptions lazily as well (jwt package is loaded by _ensure_jwks)
        from jwt.exceptions import JWTDecodeError

        if isinstance(e, JWTDecodeError):
            raise ValueError("Incorrect JWT") from e
        raise


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)
