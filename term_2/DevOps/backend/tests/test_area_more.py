from unittest.mock import Mock

from src.routes.area import read_areas, update_area
from src.models.area import Area, AreaUpdate
from src.models.userinfo import UserInfo


def _make_query_result(items):
    m = Mock()
    m.all.return_value = items
    return m


def test_read_areas_returns_list():
    mock_session = Mock()
    a1 = Area(id=1, name="Work", color="blue", user_id=1)
    mock_session.exec.return_value = _make_query_result([a1])
    user = UserInfo(id=1, email="a@x.com", full_name="X")

    res = read_areas(session=mock_session, offset=0, limit=10, current_user=user)

    assert isinstance(res, list)
    assert res[0].id == 1


def test_update_area_success():
    mock_session = Mock()
    area = Area(id=2, user_id=2, name="Old", color="red")
    mock_session.get.return_value = area
    user = UserInfo(id=2)

    updated = update_area(session=mock_session, area_id=2, area_in=AreaUpdate(name="New"), current_user=user)

    assert updated.name == "New"
    mock_session.add.assert_called()
    mock_session.commit.assert_called()
