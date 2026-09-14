from typing import Optional, TYPE_CHECKING
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel, Relationship

if TYPE_CHECKING:
    from src.models.area import Area


class NoteBase(SQLModel):
    title: str
    content: Optional[str] = None


class Note(NoteBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    area_id: Optional[int] = Field(default=None, foreign_key="area.id")
    user_id: Optional[int] = Field(default=None, foreign_key="user_info.id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
        nullable=False,
    )

    area: Optional["Area"] = Relationship(back_populates="notes")


class NoteCreate(NoteBase):
    area_id: Optional[int] = None


class NotePublic(NoteBase):
    id: int
    area_id: Optional[int]
    created_at: datetime
    updated_at: datetime


class NoteUpdate(SQLModel):
    title: Optional[str] = None
    content: Optional[str] = None
    area_id: Optional[int] = None


class NoteSearchResult(NoteBase):
    id: int
    type: str = "note"