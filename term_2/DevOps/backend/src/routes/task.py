from typing import Optional, Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from src.core.database import get_session
from src.core.constants import AREA_NOT_FOUND, TASK_NOT_FOUND
from src.models.userinfo import UserInfo
from src.models.area import Area
from src.models.task import Task, TaskCreate, TaskPublic, TaskUpdate
from src.routes.user import get_current_user

router = APIRouter()


def check_correct_area_id(session: Session, area_id: int, user_id: int) -> Area:
    area = session.get(Area, area_id)
    if not area or area.user_id != user_id:
        raise HTTPException(status_code=404, detail=AREA_NOT_FOUND)
    return area


@router.post("/tasks/", response_model=TaskPublic, responses={404: {"description": AREA_NOT_FOUND}})
def create_task(
    *,
    session: Annotated[Session, Depends(get_session)],
    task_in: TaskCreate,
    current_user: Annotated[UserInfo, Depends(get_current_user)],
):
    check_correct_area_id(session, area_id=task_in.area_id, user_id=current_user.id)

    task = Task.from_orm(task_in, update={"user_id": current_user.id})
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


@router.get("/tasks/", response_model=list[TaskPublic], responses={404: {"description": AREA_NOT_FOUND}})
def read_tasks(
    *,
    session: Annotated[Session, Depends(get_session)],
    offset: int = 0,
    limit: int = 100,
    area_id: Optional[int] = None,
    current_user: Annotated[UserInfo, Depends(get_current_user)],
):
    select_expr = select(Task).where(Task.user_id == current_user.id)
    if area_id is not None:
        check_correct_area_id(session, area_id=area_id, user_id=current_user.id)
        select_expr = select_expr.where(Task.area_id == area_id)

    tasks = session.exec(select_expr.offset(offset).limit(limit)).all()
    return tasks


@router.get("/tasks/{task_id}", response_model=TaskPublic, responses={404: {"description": TASK_NOT_FOUND}})
def read_task(
    *,
    session: Annotated[Session, Depends(get_session)],
    task_id: int,
    current_user: Annotated[UserInfo, Depends(get_current_user)],
):
    task = session.get(Task, task_id)
    if not task or task.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=TASK_NOT_FOUND)
    return task


@router.patch("/tasks/{task_id}", response_model=TaskPublic, responses={404: {"description": TASK_NOT_FOUND}})
def update_task(
    *,
    session: Annotated[Session, Depends(get_session)],
    task_id: int,
    task_in: TaskUpdate,
    current_user: Annotated[UserInfo, Depends(get_current_user)],
):
    task = session.get(Task, task_id)
    if not task or task.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=TASK_NOT_FOUND)
    update_data = task_in.dict(exclude_unset=True)

    if "area_id" in update_data:
        check_correct_area_id(session, area_id=update_data["area_id"], user_id=current_user.id)

    for key, value in update_data.items():
        setattr(task, key, value)

    session.add(task)
    session.commit()
    session.refresh(task)
    return task


@router.delete("/tasks/{task_id}", responses={404: {"description": TASK_NOT_FOUND}})
def delete_task(
    *,
    session: Annotated[Session, Depends(get_session)],
    task_id: int,
    current_user: Annotated[UserInfo, Depends(get_current_user)],
):
    task = session.get(Task, task_id)
    if not task or task.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=TASK_NOT_FOUND)
    session.delete(task)
    session.commit()
    return {"ok": True}