import os
import asyncio
from redis.asyncio import Redis, RedisError
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


async def publish_redis(redis_client: Redis, queue: str, msg: str, max_retries: int = 3, delay: float = 0.1):
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            await redis_client.rpush(queue, msg)
            return
        except RedisError as e:
            last_exception = e
            if attempt == max_retries:
                break  # no more retries

            # Exponential backoff with jitter (optional)
            await asyncio.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Failed to publish message after {max_retries} retries") from last_exception


async def update_value_redis(redis_client: Redis, key: str, value: str, max_retries: int = 3, delay: float = 0.1):
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            await redis_client.set(key, value)
            return
        except RedisError as e:
            last_exception = e
            if attempt == max_retries:
                break  # no more retries

            # Exponential backoff with jitter (optional)
            await asyncio.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Failed to update value after {max_retries} retries") from last_exception
