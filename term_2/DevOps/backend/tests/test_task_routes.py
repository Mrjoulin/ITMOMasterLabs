from unittest.mock import Mock
import pytest
from fastapi import HTTPException

from src.routes.task import (
    create_task,
    read_task,
    delete_task,
    update_task,
    check_correct_area_id,
)
from src.models.task import Task, TaskCreate, TaskUpdate
from src.models.userinfo import UserInfo
from src.models.area import Area


def test_check_correct_area_id_ok():
    mock_session = Mock()
    area = Area(id=1, user_id=5, name="X", color="c")
    mock_session.get.return_value = area

    res = check_correct_area_id(mock_session, area_id=1, user_id=5)
    assert res is area


def test_check_correct_area_id_not_found_raises():
    mock_session = Mock()
    mock_session.get.return_value = None

    with pytest.raises(HTTPException) as exc:
        check_correct_area_id(mock_session, area_id=1, user_id=5)
    assert exc.value.status_code == 404


def test_create_task_success():
    mock_session = Mock()

    def refresh_side(obj):
        obj.id = 7

    mock_session.refresh.side_effect = refresh_side
    user = UserInfo(id=3)
    task_in = TaskCreate(title="T", description="d", area_id=1)

    # session.get must return an area for check_correct_area_id
    mock_session.get.return_value = Area(id=1, user_id=3, name="X", color="c")

    task = create_task(session=mock_session, task_in=task_in, current_user=user)

    assert mock_session.add.called
    assert mock_session.commit.called
    assert getattr(task, "id", None) == 7
    assert task.user_id == user.id


def test_read_task_not_found_raises():
    mock_session = Mock()
    mock_session.get.return_value = None
    user = UserInfo(id=1)

    with pytest.raises(HTTPException) as exc:
        read_task(session=mock_session, task_id=1, current_user=user)
    assert exc.value.status_code == 404


def test_update_task_owner_mismatch_raises():
    mock_session = Mock()
    task = Task(id=1, user_id=99, title="X", description=None)
    mock_session.get.return_value = task
    user = UserInfo(id=2)

    with pytest.raises(HTTPException) as exc:
        update_task(session=mock_session, task_id=1, task_in=TaskUpdate(title="New"), current_user=user)
    assert exc.value.status_code == 404


def test_delete_task_success():
    mock_session = Mock()
    task = Task(id=1, user_id=2, title="X", description=None)
    mock_session.get.return_value = task
    user = UserInfo(id=2)

    res = delete_task(session=mock_session, task_id=1, current_user=user)

    assert res == {"ok": True}
    mock_session.delete.assert_called_once_with(task)
    mock_session.commit.assert_called_once()
