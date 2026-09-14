from unittest.mock import Mock
from src.routes.task import read_tasks
from src.models.task import Task
from src.models.userinfo import UserInfo
from src.models.area import Area


def _make_query_result(items):
    m = Mock()
    m.all.return_value = items
    return m


def test_read_tasks_with_area_id_filters_and_checks_area():
    mock_session = Mock()
    t = Task(id=1, title="T1", user_id=1)
    mock_session.exec.return_value = _make_query_result([t])
    # check_correct_area_id calls session.get
    mock_session.get.return_value = Area(id=1, user_id=1, name="A", color="c")

    user = UserInfo(id=1)
    results = read_tasks(session=mock_session, offset=0, limit=10, area_id=1, current_user=user)

    assert isinstance(results, list)
    assert results[0].id == 1
