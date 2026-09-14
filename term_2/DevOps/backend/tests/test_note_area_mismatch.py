from unittest.mock import Mock
import pytest
from fastapi import HTTPException

from src.routes.note import read_notes
from src.models.area import Area
from src.models.userinfo import UserInfo


def test_read_notes_area_mismatch_raises():
    mock_session = Mock()
    # session.get will return an area owned by another user
    mock_session.get.return_value = Area(id=1, user_id=99, name="A", color="c")
    user = UserInfo(id=1)

    with pytest.raises(HTTPException):
        read_notes(session=mock_session, offset=0, limit=10, area_id=1, current_user=user)
