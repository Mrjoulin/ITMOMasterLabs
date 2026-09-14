from typing import List, Optional, Union, Annotated
from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select, or_

from src.core.database import get_session
from src.models.task import Task, TaskSearchResult, Priority
from src.models.note import Note, NoteSearchResult
from src.models.userinfo import UserInfo
from src.routes.user import get_current_user


router = APIRouter()


@router.get("/search/")
def search_items(
    *,
    query: Annotated[str, Query(..., min_length=1)],
    item_type: Annotated[Optional[str], Query(description="Filter by item type: 'task' or 'note'")] = None,
    limit: Annotated[int, Query()] = 10,
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[UserInfo, Depends(get_current_user)],
) -> List[Union[TaskSearchResult, NoteSearchResult]]:
    
    search_pattern = f"%{query}%"
    results = []

    if item_type is None or item_type in ["task", "all"]:
        # Search for Tasks
        tasks_statement = select(Task).where(
            Task.user_id == current_user.id,
            or_(
                Task.title.ilike(search_pattern),
                Task.description.ilike(search_pattern)
            )
        )
        tasks = session.exec(tasks_statement).all()

        # Sort tasks by priority (High to Low) and then by due_date (earliest first)
        priority_order = {Priority.HIGH: 3, Priority.MEDIUM: 2, Priority.LOW: 1}
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (
                int(not t.completed),  # Active tasks comes first
                priority_order.get(t.priority, 0),  # Tasks without priority come last
                t.due_date if t.due_date else "9999-12-31"  # Tasks without due_date come last
            ),
            reverse=True
        )

        for task in sorted_tasks[:limit]:
            results.append(TaskSearchResult.model_validate(task))

        if len(results) >= limit:
            return results[:limit]

        limit -= len(results)

    if item_type is None or item_type in ["note", "all"]:
        # Search for Notes
        notes_statement = select(Note).where(
            Note.user_id == current_user.id,
            or_(
                Note.title.ilike(search_pattern),
                Note.content.ilike(search_pattern)
            )
        )
        notes = session.exec(notes_statement).all()

        # Sort notes by updated_at (newest first)
        sorted_notes = sorted(
            notes,
            key=lambda n: n.updated_at if n.updated_at else "1970-01-01T00:00:00Z",
            reverse=True
        )

        for note in sorted_notes[:limit]:
            results.append(NoteSearchResult.model_validate(note))

    return results
