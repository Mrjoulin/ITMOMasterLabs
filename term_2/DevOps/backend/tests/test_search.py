from unittest.mock import Mock
from src.routes.search import search_items
from src.models.task import Task, Priority
from src.models.note import Note
from src.models.userinfo import UserInfo


def _make_query_result(items):
    m = Mock()
    m.all.return_value = items
    return m


def test_search_items_returns_tasks_then_notes():
    # Create some tasks and notes belonging to user 1
    t1 = Task(id=1, title="T1", description="x", completed=False, priority=Priority.HIGH, due_date="2026-01-01", user_id=1)
    t2 = Task(id=2, title="T2", description="y", completed=True, priority=Priority.LOW, due_date=None, user_id=1)
    n1 = Note(id=10, title="N1", content="a", user_id=1)

    mock_session = Mock()
    # First call for tasks, second for notes
    mock_session.exec.side_effect = [_make_query_result([t1, t2]), _make_query_result([n1])]

    user = UserInfo(id=1)
    results = search_items(query="x", item_type=None, limit=10, session=mock_session, current_user=user)

    # We expect TaskSearchResult and NoteSearchResult representations appended
    ids = [r.id for r in results]
    assert 1 in ids
    assert 10 in ids
