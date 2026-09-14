from unittest.mock import Mock
import pytest

from src.routes.task import create_task
from src.models.task import TaskCreate
from src.models.area import Area
from src.models.userinfo import UserInfo


def test_create_task_area_mismatch_raises():
    mock_session = Mock()
    # session.get will return an area that does not belong to the user
    mock_session.get.return_value = Area(id=1, user_id=99, name="A", color="c")
    user = UserInfo(id=1)
    task_in = TaskCreate(title="T", description="d", area_id=1)

    with pytest.raises(Exception):
        create_task(session=mock_session, task_in=task_in, current_user=user)
