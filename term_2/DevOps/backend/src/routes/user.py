from typing import Optional, Annotated
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import ValidationError
from sqlmodel import Session, select

from src.core.database import get_session
from src.core.security import (
    create_access_token, decode_access_token,
    verify_password, get_password_hash
)
from src.models.userinfo import UserInfo, UserCreate, UserToken, UserPublic
from src.models.area import Area

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/users/login")


def get_current_user(
    session: Annotated[Session, Depends(get_session)], token: Annotated[str, Depends(oauth2_scheme)]
) -> UserInfo:
    try:
        payload = decode_access_token(token)
        token_email: Optional[str] = payload.get("sub")
        if token_email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except ValidationError:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
        )
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
        )
    user = session.exec(select(UserInfo).where(UserInfo.email == token_email)).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials: UserInfo not found")
    return user


@router.post("/users/register", response_model=UserToken, responses={400: {"description": "User already exists"}})
def create_user(*, session: Annotated[Session, Depends(get_session)], user_in: UserCreate):
    user = session.exec(select(UserInfo).where(UserInfo.email == user_in.email)).first()
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this username already exists in the system.",
        )
    hashed_password = get_password_hash(user_in.password)
    user = UserInfo.from_orm(user_in, update={"hashed_password": hashed_password})
    session.add(user)
    session.commit()
    session.refresh(user)

    # Create a default "Work" area for the new user
    work_area = Area(name="Work", color="bg-blue-500", user_id=user.id)
    session.add(work_area)
    session.commit()
    session.refresh(work_area)

    access_token = create_access_token(subject=user.email)
    return UserToken(access_token=access_token)


@router.post("/users/login", responses={400: {"description": "Incorrect email or password"}})
def login(
    *,
    session: Annotated[Session, Depends(get_session)],
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
):
    user: UserInfo = session.exec(select(UserInfo).where(UserInfo.email == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=400, detail="Incorrect email or password"
        )

    user.last_login = datetime.now(timezone.utc)
    session.add(user)
    session.commit()

    access_token = create_access_token(subject=user.email)
    return UserToken(access_token=access_token)


@router.get("/users/me", response_model=UserPublic)
def read_users_me(current_user: Annotated[UserInfo, Depends(get_current_user)]):
    return current_user
