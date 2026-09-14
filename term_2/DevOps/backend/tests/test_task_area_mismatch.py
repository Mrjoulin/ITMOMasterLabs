from unittest.mock import Mock
import pytest
from fastapi import HTTPException

from src.routes.task import read_tasks
from src.models.area import Area
from src.models.userinfo import UserInfo


def test_read_tasks_area_mismatch_raises():
    mock_session = Mock()
    mock_session.get.return_value = Area(id=2, user_id=99, name="A", color="c")
    user = UserInfo(id=1)

    with pytest.raises(HTTPException):
        read_tasks(session=mock_session, offset=0, limit=10, area_id=2, current_user=user)
