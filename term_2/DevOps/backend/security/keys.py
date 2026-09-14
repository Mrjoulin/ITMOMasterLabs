import os
from pathlib import Path


def _read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Key file not found: {path}")
    return p.read_text()


def get_private_key() -> str:
    """
    Priority:
    1. ENV (GitHub Secrets / production)
    2. File path (local dev)
    """
    key = os.getenv("PRIVATE_KEY")
    if key:
        return key

    path = os.getenv("PRIVATE_KEY_PATH")
    if path:
        return _read_file(path)

    raise FileNotFoundError("Private key not provided via PRIVATE_KEY or PRIVATE_KEY_PATH")


def get_public_key() -> str:
    key = os.getenv("PUBLIC_KEY")
    if key:
        return key
    path = os.getenv("PUBLIC_KEY_PATH")
    if path:
        return _read_file(path)

    raise FileNotFoundError("Public key not provided via PUBLIC_KEY or PUBLIC_KEY_PATH")
