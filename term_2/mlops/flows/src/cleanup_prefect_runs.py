import os
from datetime import datetime, timedelta

import structlog
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from utils.postgres import get_postgres_connection

# ---------------------------
# Configuration
# ---------------------------
MAX_AGE_HOURS = int(os.getenv("PREFECT_RUNS_MAX_AGE_HOURS", 48))
# Rows deleted per transaction to avoid long locks / WAL bloat.
BATCH_SIZE = int(os.getenv("PREFECT_CLEANUP_BATCH_SIZE", 5000))

logger = structlog.get_logger(__name__)


@task(name="Delete old events/logs", retries=3, retry_delay_seconds=1, cache_policy=NO_CACHE)
def _delete_orphan_table(conn, table: str, column: str, cutoff: datetime) -> int:
    """Range-delete rows from an orphan table (events / event_resources / log)."""
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {table} WHERE {column} < %s",
            (cutoff,),
        )
        deleted = cur.rowcount
    conn.commit()
    return deleted


@task(name="Collect old flow runs ids", retries=3, retry_delay_seconds=1, cache_policy=NO_CACHE)
def _collect_old_flow_run_ids(conn, cutoff: datetime) -> list:
    """Return flow_run ids whose start_time is older than the cutoff."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM flow_run WHERE start_time < %s",
            (cutoff,),
        )
        return [row[0] for row in cur.fetchall()]


@task(name="Delete runs info in batches", retries=3, retry_delay_seconds=1, cache_policy=NO_CACHE)
def _delete_in_batches(conn, table: str, column: str, ids: list, batch_size: int) -> int:
    """Delete rows where ``column`` is in ``ids``, in batches, returning total removed."""
    if not ids:
        return 0
    total = 0
    with conn.cursor() as cur:
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            cur.execute(
                f"DELETE FROM {table} WHERE {column} = ANY(%s::uuid[])",
                (batch,),
            )
            total += cur.rowcount
    conn.commit()
    return total


@task(name="Delete run states in batches", retries=3, retry_delay_seconds=1, cache_policy=NO_CACHE)
def _delete_task_run_states(conn, flow_run_ids: list, batch_size: int) -> int:
    """Delete task_run_state rows whose parent task_run belongs to the given flow runs."""
    if not flow_run_ids:
        return 0
    total = 0
    with conn.cursor() as cur:
        for start in range(0, len(flow_run_ids), batch_size):
            batch = flow_run_ids[start:start + batch_size]
            cur.execute(
                """
                    DELETE FROM task_run_state
                WHERE task_run_id IN (
                    SELECT id FROM task_run WHERE flow_run_id = ANY(%s::uuid[])
                )
                """,
                (batch,),
            )
            total += cur.rowcount
    conn.commit()
    return total


@flow(name="cleanup-prefect-runs", log_prints=True)
def cleanup_prefect_runs_flow(max_age_hours: int = MAX_AGE_HOURS):
    """Delete Prefect runs and their artifacts older than ``max_age_hours``.

    Prefect's API-driven deletion is slow for large backlogs and does not
    reclaim the biggest tables (``events``, ``event_resources``, ``log``),
    which are not FK-linked to runs. This flow cleans them directly in
    Postgres, leaf tables first, in batched transactions.
    """
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    logger.info(
        f"Starting Prefect cleanup: runs older than {max_age_hours}h "
        f"(cutoff {cutoff.isoformat()}), batch size {BATCH_SIZE}."
    )

    conn = get_postgres_connection()
    try:
        # 1. Orphan event tables + logs (range delete by timestamp).
        events_deleted = _delete_orphan_table(conn, "event_resources", "occurred", cutoff)
        logger.info(f"Deleted {events_deleted} row(s) from event_resources.")

        events_deleted = _delete_orphan_table(conn, "events", "occurred", cutoff)
        logger.info(f"Deleted {events_deleted} row(s) from events.")

        logs_deleted = _delete_orphan_table(conn, "log", "timestamp", cutoff)
        logger.info(f"Deleted {logs_deleted} row(s) from log.")

        # 2. Collect the flow_run ids we intend to drop, then delete their
        #    children in FK-safe order before dropping the runs themselves.
        old_run_ids = _collect_old_flow_run_ids(conn, cutoff)
        logger.info(f"Found {len(old_run_ids)} flow_run(s) older than cutoff.")

        if not old_run_ids:
            logger.info("No flow runs matched the age threshold, nothing more to clean.")
            return

        # task_run_state references task_run; resolve via task_run.flow_run_id.
        task_runs_deleted = _delete_in_batches(
            conn, "task_run", "flow_run_id", old_run_ids, BATCH_SIZE
        )
        logger.info(f"Deleted {task_runs_deleted} row(s) from task_run.")

        # task_run_state has no direct flow_run_id column; delete by joining
        # task_run. Do it in batches over the run ids.
        task_state_deleted = _delete_task_run_states(conn, old_run_ids, BATCH_SIZE)
        logger.info(f"Deleted {task_state_deleted} row(s) from task_run_state.")

        flow_state_deleted = _delete_in_batches(
            conn, "flow_run_state", "flow_run_id", old_run_ids, BATCH_SIZE
        )
        logger.info(f"Deleted {flow_state_deleted} row(s) from flow_run_state.")

        runs_deleted = _delete_in_batches(
            conn, "flow_run", "id", old_run_ids, BATCH_SIZE
        )

        logger.info(f"Deleted {runs_deleted} row(s) from flow_run.")
        logger.info(
            f"Prefect cleanup finished: removed {runs_deleted} flow run(s) and associated rows."
        )
    finally:
        conn.close()


if __name__ == "__main__":
    cleanup_prefect_runs_flow()
