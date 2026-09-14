from unittest.mock import Mock
import pytest
from sqlmodel import Session
from fastapi import HTTPException


def test_get_current_user_missing_sub_raises(monkeypatch):
    # Patch decode_access_token used inside src.routes.user
    monkeypatch.setattr("src.routes.user.decode_access_token", lambda t: {})
    from src.routes.user import get_current_user

    with pytest.raises(HTTPException) as exc:
        get_current_user(session=Mock(spec=Session), token="t")
    assert exc.value.status_code == 401


def test_get_current_user_decode_raises(monkeypatch):
    def _raise(t):
        raise ValueError("bad token")

    monkeypatch.setattr("src.routes.user.decode_access_token", _raise)
    from src.routes.user import get_current_user

    with pytest.raises(HTTPException) as exc:
        get_current_user(session=Mock(spec=Session), token="t")
    assert exc.value.status_code == 401
