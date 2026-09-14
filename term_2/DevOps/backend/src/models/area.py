from typing import List, Optional, TYPE_CHECKING
from datetime import datetime, timezone

from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from src.models.task import Task
    from src.models.note import Note
    from src.models.userinfo import UserInfo


class AreaBase(SQLModel):
    name: str
    color: str


class Area(AreaBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="user_info.id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
        nullable=False,
    )

    tasks: List["Task"] = Relationship(back_populates="area")
    notes: List["Note"] = Relationship(back_populates="area")
    user_info: Optional["UserInfo"] = Relationship(back_populates="areas")


class AreaCreate(AreaBase):
    pass


class AreaPublic(AreaBase):
    id: int
    created_at: datetime
    updated_at: datetime


class AreaUpdate(SQLModel):
    name: Optional[str] = None
    color: Optional[str] = None
