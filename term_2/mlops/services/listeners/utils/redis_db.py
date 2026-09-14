import os
import asyncio
import json
from typing import Optional, Union

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError
from .metrics import EVENTS_PUBLISHED_TOTAL

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_QUEUE_KEY = os.getenv("REDIS_QUEUE_KEY", "data")


class AsyncRedisPublisher:
    """
    Async publisher that maintains a stable Redis connection and retries failed publishes.
    """

    def __init__(
        self,
        redis_url: str = REDIS_URL,
        key_prefix: str = REDIS_QUEUE_KEY,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 2.0,
    ):
        """
        Args:
            redis_url: Redis connection URL.
            key_prefix: Prefix for Redis keys (e.g., "data" -> "data:interval_start").
            max_retries: Maximum number of retry attempts per publish.
            base_delay: Initial delay in seconds (exponential backoff).
            max_delay: Maximum delay between retries.
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        self._client: Optional[Redis] = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> Redis:
        """Get or create the Redis client (connection pool)."""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = redis.from_url(
                        self.redis_url,
                        decode_responses=True,
                        max_connections=10,
                        socket_keepalive=True,
                        socket_timeout=5,
                        health_check_interval=30,
                        retry_on_error=[redis.TimeoutError],
                    )
        return self._client

    async def publish(self, timestamp: Union[float, int], value: float, source_name: str) -> None:
        """
        Publish a message to the Redis queue keyed by the 5‑minute interval of `timestamp`.
        Automatically retries on failures.

        Args:
            timestamp: Unix timestamp (seconds, can be fractional).
            value: The value to store (must be JSON serializable).
            source_name: Name of the loader/source.
        """
        # Round down to nearest 5 minutes (300 seconds)
        timestamp_s = timestamp // 1000
        interval_start = (timestamp_s // 300) * 300
        queue_key = f"{self.key_prefix}:{interval_start}"

        payload = {"ts": timestamp, "v": value, "src": source_name}
        message = json.dumps(payload)

        last_exception = None
        delay = self.base_delay

        for attempt in range(self.max_retries + 1):
            try:
                client = await self._get_client()
                await client.rpush(queue_key, message)
                EVENTS_PUBLISHED_TOTAL.labels(source=source_name).inc()
                return  # success
            except RedisError as e:
                last_exception = e
                if attempt == self.max_retries:
                    break  # no more retries

                # Exponential backoff with jitter (optional)
                wait = min(delay, self.max_delay)
                await asyncio.sleep(wait)
                delay *= 2  # double for next attempt
                await self._reset_client()

        raise RuntimeError(
            f"Failed to publish message after {self.max_retries} retries"
        ) from last_exception

    async def _reset_client(self) -> None:
        """Close the existing client and set it to None so a new one is created."""
        async with self._lock:
            if self._client is not None:
                await self._client.close()
                self._client = None

    async def close(self) -> None:
        """Gracefully close the Redis client when shutting down the service."""
        await self._reset_client()
