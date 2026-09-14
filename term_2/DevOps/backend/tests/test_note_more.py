from unittest.mock import Mock

from src.routes.note import read_notes
from src.models.note import Note
from src.models.userinfo import UserInfo


def test_read_notes_no_area_returns_list():
    mock_session = Mock()
    n = Note(id=1, title="N1", content="c", user_id=1)
    m = Mock()
    m.all.return_value = [n]
    mock_session.exec.return_value = m
    user = UserInfo(id=1)

    res = read_notes(session=mock_session, offset=0, limit=10, area_id=None, current_user=user)

    assert isinstance(res, list)
    assert res[0].id == 1
