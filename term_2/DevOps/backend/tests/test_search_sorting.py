from unittest.mock import Mock

from src.routes.search import search_items
from src.models.task import Task, Priority
from src.models.note import Note
from src.models.userinfo import UserInfo


def _make_query_result(items):
    m = Mock()
    m.all.return_value = items
    return m


def test_search_task_sorting_by_priority_and_due_date():
    # Tasks with different priority/completion/due_date
    t_high = Task(id=1, title="high", description="x", completed=False, priority=Priority.HIGH, due_date="2026-01-01", user_id=1)
    t_med = Task(id=2, title="med", description="x", completed=False, priority=Priority.MEDIUM, due_date="2026-01-02", user_id=1)
    t_low_completed = Task(id=3, title="low", description="x", completed=True, priority=Priority.LOW, due_date=None, user_id=1)

    mock_session = Mock()
    mock_session.exec.return_value = _make_query_result([t_med, t_low_completed, t_high])

    user = UserInfo(id=1)
    results = search_items(query="x", item_type="task", limit=10, session=mock_session, current_user=user)

    # Order should prefer incomplete, then priority (HIGH first), then due_date
    ids = [r.id for r in results]
    assert ids[0] == 1
    assert ids[1] == 2
    assert 3 in ids


def test_search_limit_and_item_type_note():
    n1 = Note(id=10, title="N1", content="x", user_id=1)
    n2 = Note(id=11, title="N2", content="x", user_id=1)

    mock_session = Mock()
    # When searching notes only, search() should only call exec once for notes.
    mock_session.exec.return_value = _make_query_result([n1, n2])

    user = UserInfo(id=1)
    results = search_items(query="x", item_type="note", limit=1, session=mock_session, current_user=user)

    assert len(results) == 1
    # Notes are sorted by `updated_at` (newest first); n2 was created after n1
    assert results[0].id == 11


def test_search_item_type_all_returns_both():
    t = Task(id=5, title="T", description="x", user_id=1)
    n = Note(id=20, title="N", content="x", user_id=1)

    mock_session = Mock()
    mock_session.exec.side_effect = [_make_query_result([t]), _make_query_result([n])]

    user = UserInfo(id=1)
    results = search_items(query="x", item_type="all", limit=10, session=mock_session, current_user=user)

    ids = [r.id for r in results]
    assert 5 in ids and 20 in ids
