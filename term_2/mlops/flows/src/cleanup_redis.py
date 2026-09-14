# flows/cleanup_redis.py
import time
import os
import redis
from prefect import flow, task
from typing import List, Set, Optional

# ---------------------------
# Configuration
# ---------------------------
REDIS_HOST = os.getenv("PREFECT_REDIS_MESSAGING_HOST", "redis")
REDIS_PORT = int(os.getenv("PREFECT_REDIS_MESSAGING_PORT", 6379))
REDIS_DB = int(os.getenv("PREFECT_REDIS_MESSAGING_DB", 0))

# Default prefix to clean (can be overridden via flow parameter)
QUEUE_PREFIXES = ["data", "agg"]   # e.g., keys like "data:1734567890"

# Age threshold: 1 hour
AGE_THRESHOLD_SECONDS = 3600


def get_redis_client():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


@task
def scan_keys_with_prefix(prefix: str) -> List[str]:
    """Return all keys matching 'prefix:*' using SCAN (non‑blocking)."""
    r = get_redis_client()
    keys = []
    cursor = 0
    while True:
        cursor, batch = r.scan(cursor, match=f"{prefix}:*", count=1000)
        keys.extend(batch)
        if cursor == 0:
            break
    return keys


def extract_timestamp_from_key(key: str) -> Optional[int]:
    """
    Extract timestamp from key like 'prefix:1234567890'.
    Returns timestamp in seconds (int) or None if not found.
    """
    # Remove prefix and colon
    parts = key.split(":", 1)
    if len(parts) != 2:
        return None
    timestamp_str = parts[1]
    # Try to parse as integer (seconds)
    try:
        ts = int(timestamp_str)
        # If timestamp looks like milliseconds (13 digits), convert to seconds
        if ts > 1e12:
            ts = ts // 1000
        return ts
    except ValueError:
        return None


@task
def classify_keys(keys: List[str], max_age_seconds: int) -> tuple[Set[str], Set[str]]:
    """
    Returns two sets:
    - empty_keys: keys that exist but have zero length (based on type)
    - old_keys: keys whose embedded timestamp is older than max_age_seconds
    """
    r = get_redis_client()
    empty_keys = set()
    old_keys = set()
    now = time.time()

    for key in keys:
        # 1. Check if key is empty
        key_type = r.type(key)
        if key_type == "list":
            if r.llen(key) == 0:
                empty_keys.add(key)
        elif key_type == "hash":
            if r.hlen(key) == 0:
                empty_keys.add(key)
        elif key_type == "set":
            if r.scard(key) == 0:
                empty_keys.add(key)
        elif key_type == "string":
            if r.get(key) == "":
                empty_keys.add(key)
        elif key_type == "zset":
            if r.zcard(key) == 0:
                empty_keys.add(key)

        # 2. Check if key is old (by timestamp in name)
        ts = extract_timestamp_from_key(key)
        if ts is not None:
            if now - ts > max_age_seconds:
                old_keys.add(key)

    return empty_keys, old_keys


@task(log_prints=True)
def delete_keys(keys: Set[str], reason: str):
    """Delete a set of keys and log count."""
    if not keys:
        print(f"No keys to delete for reason: {reason}")
        return
    r = get_redis_client()
    r.delete(*keys)
    print(f"Deleted {len(keys)} {reason} keys: {', '.join(list(keys))}")


@task(log_prints=True)
def cleanup_redis(prefix: str, max_age_seconds: int = AGE_THRESHOLD_SECONDS):
    """
    Scheduled every hour: deletes empty Redis keys and keys older than `max_age_seconds`
    that match the given prefix.

    Args:
        prefix: Key prefix to scan (e.g., 'data', 'aggregated')
        max_age_seconds: Age threshold in seconds (default 3600 = 1 hour)
    """
    print(f"Starting Redis cleanup for prefix '{prefix}' with max age {max_age_seconds}s")

    # 1. Get all matching keys
    keys = scan_keys_with_prefix(prefix)
    if not keys:
        print("No keys found. Exiting.")
        return
    print(f"Found {len(keys)} keys with prefix '{prefix}'")

    # 2. Classify
    empty_keys, old_keys = classify_keys(keys, max_age_seconds)

    print(f"Identified {len(empty_keys)} empty keys, {len(old_keys)} old keys")

    # 3. Delete
    all_to_delete = empty_keys.union(old_keys)
    if all_to_delete:
        delete_keys(all_to_delete, "empty or old")
    else:
        print("No keys to delete.")

    # 4. Optional: Report remaining keys
    remaining_keys = set(keys) - all_to_delete
    print(f"Remaining keys after cleanup in '{prefix}': {len(remaining_keys)}")


@flow(name="redis-cleanup", log_prints=True)
def cleanup_redis_flow():
    for prefix in QUEUE_PREFIXES:
        cleanup_redis(prefix)


if __name__ == "__main__":
    cleanup_redis_flow()
