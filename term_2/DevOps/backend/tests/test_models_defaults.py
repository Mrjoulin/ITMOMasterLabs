from datetime import timezone

from src.models.area import Area
from src.models.note import Note
from src.models.task import Task
from src.models.userinfo import UserInfo


def test_models_have_timezone_aware_defaults():
    a = Area(name="X", color="c")
    n = Note(title="N", content="c")
    t = Task(title="T")
    u = UserInfo(email="e@example.com", full_name="F")

    assert a.created_at.tzinfo is not None
    assert n.created_at.tzinfo is not None
    assert t.created_at.tzinfo is not None
    assert u.created_at.tzinfo is not None

    # They should use UTC
    assert a.created_at.tzinfo == timezone.utc
    assert n.created_at.tzinfo == timezone.utc
    assert t.created_at.tzinfo == timezone.utc
    assert u.created_at.tzinfo == timezone.utc
