import os
from redis.asyncio import Redis
from prefect import task

# ---------------------------
# Configuration
# ---------------------------
REDIS_HOST = os.getenv("PREFECT_REDIS_MESSAGING_HOST", "localhost")
REDIS_PORT = int(os.getenv("PREFECT_REDIS_MESSAGING_PORT", 6379))
REDIS_DB = int(os.getenv("PREFECT_REDIS_MESSAGING_DB", 0))


@task(name="Get Redis Client", retries=3)
def get_redis_client() -> Redis:
    """Get or create the Redis client (connection pool)."""
    _client = Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        max_connections=10,
        socket_keepalive=True,
        socket_timeout=5,
        health_check_interval=30
    )
    return _client

