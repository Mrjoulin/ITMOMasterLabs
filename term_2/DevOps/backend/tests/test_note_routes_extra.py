from unittest.mock import Mock
import pytest
from fastapi import HTTPException

from src.routes.note import (
    create_note,
    read_note,
    delete_note,
    update_note,
    check_correct_area_id,
)
from src.models.note import Note, NoteCreate, NoteUpdate
from src.models.userinfo import UserInfo
from src.models.area import Area


def test_check_correct_area_id_ok():
    mock_session = Mock()
    area = Area(id=2, user_id=7, name="A", color="c")
    mock_session.get.return_value = area

    res = check_correct_area_id(mock_session, area_id=2, user_id=7)
    assert res is area


def test_create_note_success():
    mock_session = Mock()

    def refresh_side(obj):
        obj.id = 11

    mock_session.refresh.side_effect = refresh_side
    user = UserInfo(id=4)
    note_in = NoteCreate(title="N", content="c", area_id=1)

    mock_session.get.return_value = Area(id=1, user_id=4, name="Work", color="c")

    note = create_note(session=mock_session, note_in=note_in, current_user=user)

    assert mock_session.add.called
    assert mock_session.commit.called
    assert getattr(note, "id", None) == 11
    assert note.user_id == user.id


def test_read_note_not_found_raises():
    mock_session = Mock()
    mock_session.get.return_value = None
    user = UserInfo(id=1)

    with pytest.raises(HTTPException) as exc:
        read_note(session=mock_session, note_id=1, current_user=user)
    assert exc.value.status_code == 404


def test_update_note_area_change_checks_area():
    mock_session = Mock()
    note = Note(id=1, user_id=5, title="X", content="c")
    mock_session.get.return_value = note
    user = UserInfo(id=5)

    # When updating area_id, session.get needs to return an area owned by user
    mock_session.get.side_effect = [note, Area(id=2, user_id=5, name="A", color="c")]

    updated = update_note(session=mock_session, note_id=1, note_in=NoteUpdate(area_id=2), current_user=user)

    assert mock_session.commit.called
    assert updated.area_id == 2


def test_delete_note_success():
    mock_session = Mock()
    note = Note(id=1, user_id=3, title="X", content="c")
    mock_session.get.return_value = note
    user = UserInfo(id=3)

    res = delete_note(session=mock_session, note_id=1, current_user=user)

    assert res == {"ok": True}
    mock_session.delete.assert_called_once_with(note)
    mock_session.commit.assert_called_once()
