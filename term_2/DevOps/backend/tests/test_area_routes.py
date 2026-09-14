from unittest.mock import Mock
import pytest
from fastapi import HTTPException

from src.routes.area import create_area, read_area, delete_area, update_area
from src.models.area import AreaCreate, Area
from src.models.userinfo import UserInfo


def test_read_area_not_found_raises():
    mock_session = Mock()
    mock_session.get.return_value = None
    user = UserInfo(id=1, email="a")

    with pytest.raises(HTTPException) as exc:
        read_area(session=mock_session, area_id=1, current_user=user)
    assert exc.value.status_code == 404


def test_create_area_success_calls_db():
    mock_session = Mock()

    def refresh_side(obj):
        obj.id = 5

    mock_session.refresh.side_effect = refresh_side
    user = UserInfo(id=2)
    area_in = AreaCreate(name="Test", color="red")

    area = create_area(session=mock_session, area_in=area_in, current_user=user)

    assert mock_session.add.called
    assert mock_session.commit.called
    assert getattr(area, "id", None) == 5
    assert area.user_id == user.id


def test_delete_area_success():
    mock_session = Mock()
    area = Area(id=1, user_id=2, name="X", color="c")
    mock_session.get.return_value = area
    user = UserInfo(id=2)

    res = delete_area(session=mock_session, area_id=1, current_user=user)

    assert res == {"ok": True}
    mock_session.delete.assert_called_once_with(area)
    mock_session.commit.assert_called_once()


def test_update_area_owner_mismatch_raises():
    mock_session = Mock()
    area = Area(id=1, user_id=99, name="X", color="c")
    mock_session.get.return_value = area
    user = UserInfo(id=2)

    with pytest.raises(HTTPException) as exc:
        update_area(session=mock_session, area_id=1, area_in=AreaCreate(name="New", color="y"), current_user=user)
    assert exc.value.status_code == 404
