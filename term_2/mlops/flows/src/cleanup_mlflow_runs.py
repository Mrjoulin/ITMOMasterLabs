import os
from datetime import datetime, timedelta, timezone

import structlog
import subprocess
from mlflow import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
import boto3

# ---------------------------
# Configuration
# ---------------------------
BACKEND_STORE_URI = os.getenv("MLFLOW_BACKEND_STORE_URI", "")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:4444")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "polymarket-btc-5m-rf")
S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://storage:9000")
S3_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "s3admin")
S3_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "s3admin")
S3_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_USE_SSL = os.getenv("MLFLOW_S3_IGNORE_TLS", "true").lower() in ("false", "no", "0")
S3_MLFLOW_BUCKET_NAME = os.getenv("S3_BUCKET", "mlflow")

MAX_AGE_HOURS = int(os.getenv("MLFLOW_RUNS_MAX_AGE_HOURS", 24))

logger = structlog.get_logger(__name__)


@task(name="Delete S3 old artifacts", retries=3, retry_delay_seconds=1, cache_policy=NO_CACHE)
def clean_s3_data(cutoff_date: datetime):
    s3_client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        region_name=S3_DEFAULT_REGION,
        use_ssl=S3_USE_SSL,
        verify=S3_USE_SSL
    )
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=S3_MLFLOW_BUCKET_NAME)

    deleted_count = 0
    for page in page_iterator:
        # Check if the bucket has contents
        if "Contents" not in page:
            continue

        # Prepare a batch of objects to delete (max 1000 per S3 API call)
        objects_to_delete = []
        for obj in page["Contents"]:
            obj_key = obj["Key"]
            last_modified = obj["LastModified"]

            # If the file is older than the cutoff, queue it for deletion
            if last_modified < cutoff_date:
                logger.info(f"Queueing for deletion: {obj_key} (Modified: {last_modified})")
                objects_to_delete.append({"Key": obj_key})

        # Execute the batch deletion
        if objects_to_delete:
            response = s3_client.delete_objects(
                Bucket=S3_MLFLOW_BUCKET_NAME, Delete={"Objects": objects_to_delete}
            )
            deleted_count += len(response.get("Deleted", []))

    logger.info(f"\nSuccessfully deleted {deleted_count} old files.")
    return deleted_count


@flow(name="cleanup-mlflow-runs", log_prints=True)
def cleanup_mlflow_runs_flow(max_age_hours: int = MAX_AGE_HOURS):
    """Delete MLflow runs and their artifacts older than ``max_age_hours``.

    For each stale run we remove the artifact files first (``runs:/<id>``) and
    then mark the run as deleted, so both the run record and the underlying
    artifact storage (S3 / MinIO) are reclaimed.
    """
    client = MlflowClient(tracking_uri=TRACKING_URI)

    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    except Exception as e:
        logger.error("experiment_lookup_failed", experiment=EXPERIMENT_NAME, error=str(e))
        raise RuntimeError(f"Could not look up experiment '{EXPERIMENT_NAME}'") from e

    if experiment is None:
        logger.info(f"Experiment '{EXPERIMENT_NAME}' does not exist, nothing to clean.")
        return

    # Fetch all runs (active + deleted) so we also tidy anything left behind
    # by prior partial deletions. ``search_runs`` is paginated by default; use
    # a generous cap to bound memory on very large experiments.
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=10000,
        order_by=["start_time ASC", "run_id"],
    )

    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    logger.info(
        f"Scanning {len(runs)} run(s) in experiment '{EXPERIMENT_NAME}' "
        f"older than {max_age_hours} hour(s) (cutoff {cutoff.isoformat()})."
    )

    runs_to_delete = []
    for run in runs[::-1]:
        run_id = run.info.run_id
        run_name = run.data.tags.get(MLFLOW_RUN_NAME, run_id)
        # ``start_time`` is unix epoch in milliseconds.
        start_dt = datetime.fromtimestamp(run.info.start_time / 1000)
        if start_dt < cutoff:
            runs_to_delete.append((run_id, run_name, start_dt))

    if not runs_to_delete:
        logger.info("No runs matched the age threshold, nothing to clean.")
        return

    removed = 0
    for run_id, run_name, start_dt in runs_to_delete:
        logger.info(f"Deleting run '{run_name}' ({run_id}), started {start_dt.isoformat()}")
        try:
            client.delete_run(run_id)
        except Exception as e:
            logger.warning("run_delete_failed", run_id=run_id, error=str(e))
            continue
        removed += 1

    logger.info(f"Cleanup finished: deleted {removed}/{len(runs_to_delete)} run(s).")
    logger.info("Run mlflow gc to delete data")
    status = subprocess.run([
        "mlflow", "gc",
        "--tracking-uri", TRACKING_URI,
        "--backend-store-uri", BACKEND_STORE_URI,
        "--older-than", "0s"
    ])
    if status.returncode == 0:
        logger.info(f"Subprocess call successfully, old runs deleted")
    else:
        logger.error(f"Subprocess call failed, finished with return code: {status.returncode}")

    try:
        clean_s3_data(cutoff_date=cutoff.astimezone(timezone.utc))
    except Exception as e:
        logger.error(f"Unable to clean S3 old artifacts: {e}")
        print(e)


if __name__ == "__main__":
    cleanup_mlflow_runs_flow()
