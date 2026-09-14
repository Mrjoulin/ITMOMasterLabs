import os
import asyncio
from typing import Dict, Any

from prometheus_client import Counter, Gauge, CONTENT_TYPE_LATEST
from prometheus_client.core import REGISTRY
from prometheus_client.exposition import generate_latest

POLL_INTERVAL_S = float(os.getenv("METRICS_POLL_INTERVAL_S", "10"))
REDIS_QUEUE_PREFIXES = os.getenv("REDIS_QUEUE_PREFIXES", "data,agg")


def _parse_prefixes(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.replace(";", ",").replace(" ", ",").split(",")]
    return [p for p in parts if p]


EVENTS_PUBLISHED_TOTAL = Counter(
    "polymarket_events_total",
    "Total number of events published to Redis",
    labelnames=("source",),
)

REDIS_QUEUE_LENGTH = Gauge(
    "polymarket_redis_queue_length",
    "Length of Redis list (queue)",
    labelnames=("key",),
)


def render_metrics() -> bytes:
    return generate_latest(REGISTRY)


async def start_redis_queue_metrics_poller(redis_client: Any) -> None:
    """Periodically updates queue length gauges for Redis list-queues.

    Notes:
      - Uses SCAN to avoid blocking Redis.
      - For large keyspaces, consider narrowing the match pattern.
      - Keys that were deleted or emptied since the last cycle are removed
        from the gauge so the dashboard only reflects active, non-empty queues.
    """

    prefixes = _parse_prefixes(REDIS_QUEUE_PREFIXES)
    matches = [f"{p}:*" for p in prefixes]
    while True:
        try:
            client = await redis_client._get_client()
            seen: Dict[str, int] = {}

            for match in matches:
                cursor = 0
                while True:
                    cursor, keys = await client.scan(cursor=cursor, match=match, count=200)
                    if keys:
                        pipe = client.pipeline()
                        for k in keys:
                            pipe.llen(k)
                        lengths = await pipe.execute()
                        for k, ln in zip(keys, lengths):
                            seen[str(k)] = int(ln)
                    if cursor == 0:
                        break

            # Set / update gauges for keys that exist and are non-empty.
            for key, ln in seen.items():
                if ln > 0:
                    REDIS_QUEUE_LENGTH.labels(key=key).set(ln)

            # Remove gauges for keys that disappeared or became empty, so the
            # dashboard stops showing stale series.

            # ``_metrics`` keys are tuples of label values (1-tuple here).
            for label in list(REDIS_QUEUE_LENGTH._metrics.keys()):
                key = label[0]
                if key not in seen or seen[key] == 0:
                    REDIS_QUEUE_LENGTH.remove(*label)
        except Exception:
            pass

        await asyncio.sleep(POLL_INTERVAL_S)


async def metrics_http_handler(reader, writer) -> None:
    body = render_metrics()
    headers = [
        b"HTTP/1.1 200 OK",
        f"Content-Type: {CONTENT_TYPE_LATEST}".encode("utf-8"),
        f"Content-Length: {len(body)}".encode("utf-8"),
        b"Connection: close",
        b"",
        b"",
    ]
    writer.write(b"\r\n".join(headers) + body)
    await writer.drain()
    writer.close()
