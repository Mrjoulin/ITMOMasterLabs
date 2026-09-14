from typing import Optional, Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from src.core.database import get_session
from src.core.constants import AREA_NOT_FOUND, NOTE_NOT_FOUND
from src.models.userinfo import UserInfo
from src.models.area import Area
from src.models.note import Note, NoteCreate, NotePublic, NoteUpdate
from src.routes.user import get_current_user

router = APIRouter()


def check_correct_area_id(session: Session, area_id: int, user_id: int) -> Area:
    area = session.get(Area, area_id)
    if not area or area.user_id != user_id:
        raise HTTPException(status_code=404, detail=AREA_NOT_FOUND)
    return area


@router.post("/notes/", response_model=NotePublic, responses={404: {"description": AREA_NOT_FOUND}})
def create_note(
    *,
    session: Annotated[Session, Depends(get_session)],
    note_in: NoteCreate,
    current_user: Annotated[UserInfo, Depends(get_current_user)],
):
    check_correct_area_id(session, area_id=note_in.area_id, user_id=current_user.id)

    note = Note.from_orm(note_in, update={"user_id": current_user.id})
    session.add(note)
    session.commit()
    session.refresh(note)
    return note


@router.get("/notes/", response_model=list[NotePublic], responses={404: {"description": AREA_NOT_FOUND}})
def read_notes(
    *,
    session: Annotated[Session, Depends(get_session)],
    offset: int = 0,
    limit: int = 100,
    area_id: Optional[int] = None,
    current_user: Annotated[UserInfo, Depends(get_current_user)],
):
    select_expr = select(Note).where(Note.user_id == current_user.id)
    if area_id is not None:
        check_correct_area_id(session, area_id=area_id, user_id=current_user.id)
        select_expr = select_expr.where(Note.area_id == area_id)

    select_expr = select_expr.order_by(Note.updated_at.desc())
    notes = session.exec(select_expr.offset(offset).limit(limit)).all()
    return notes


@router.get("/notes/{note_id}", response_model=NotePublic, responses={404: {"description": NOTE_NOT_FOUND}})
def read_note(
    *,
    session: Annotated[Session, Depends(get_session)],
    note_id: int,
    current_user: Annotated[UserInfo, Depends(get_current_user)],
):
    note = session.get(Note, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=NOTE_NOT_FOUND)
    return note


@router.patch("/notes/{note_id}", response_model=NotePublic, responses={404: {"description": NOTE_NOT_FOUND}})
def update_note(
    *,
    session: Annotated[Session, Depends(get_session)],
    note_id: int,
    note_in: NoteUpdate,
    current_user: Annotated[UserInfo, Depends(get_current_user)],
):
    note = session.get(Note, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=NOTE_NOT_FOUND)
    update_data = note_in.dict(exclude_unset=True)

    if "area_id" in update_data:
        check_correct_area_id(session, area_id=update_data["area_id"], user_id=current_user.id)

    for key, value in update_data.items():
        setattr(note, key, value)

    session.add(note)
    session.commit()
    session.refresh(note)
    return note


@router.delete("/notes/{note_id}", responses={404: {"description": NOTE_NOT_FOUND}})
def delete_note(
    *,
    session: Annotated[Session, Depends(get_session)],
    note_id: int,
    current_user: Annotated[UserInfo, Depends(get_current_user)],
):
    note = session.get(Note, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=NOTE_NOT_FOUND)
    session.delete(note)
    session.commit()
    return {"ok": True}