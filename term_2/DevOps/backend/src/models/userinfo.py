from typing import List, Optional, TYPE_CHECKING
from datetime import datetime, timezone

from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from .area import Area


class UserBase(SQLModel):
    email: str = Field(unique=True, index=True)
    full_name: str


class UserInfo(UserBase, table=True):
    __tablename__ = "user_info"

    id: int = Field(default=None, primary_key=True)
    hashed_password: str
    last_login: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
        nullable=False,
    )

    areas: List["Area"] = Relationship(back_populates="user_info")


class UserCreate(UserBase):
    password: str


class UserPublic(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime


class UserToken(SQLModel):
    access_token: str
    token_type: str = "Bearer"
