import sys
import asyncio

from listeners import AbstractListener, LISTENERS
from utils import AsyncRedisPublisher, metrics_http_handler, start_redis_queue_metrics_poller


def get_listener_by_name(name: str) -> AbstractListener:
    name = name.strip().lower()
    listener = LISTENERS.get(name)

    if listener is None:
        raise ImportError(f"Listener {name} not found")

    return listener()


async def run_several_listeners(listeners_names: list[str]):
    redis_client = AsyncRedisPublisher()

    metrics_host = "0.0.0.0"
    metrics_port = int(sys.argv[sys.argv.index("--metrics-port") + 1]) if "--metrics-port" in sys.argv else 8000
    server = await asyncio.start_server(metrics_http_handler, metrics_host, metrics_port)

    listeners_tasks = [
        LISTENERS[listener_name](redis_client=redis_client).listen()
        for listener_name in listeners_names
        if listener_name in LISTENERS and LISTENERS[listener_name] is not None
    ]

    poller_task = start_redis_queue_metrics_poller(redis_client)
    async with server:
        await asyncio.gather(server.serve_forever(), poller_task, *listeners_tasks)


def start_listener():
    if len(sys.argv) < 2:
        raise ValueError("Provide listener name, should be one of: polymarket, binance, coinbase, all")

    listener_names = sys.argv[1:]
    if len(listener_names) == 1 or listener_names[1] is None:
        listener_name = listener_names[0]
        if listener_name != "all":
            listener = get_listener_by_name(listener_name)
            listener.start()
        else:
            asyncio.run(run_several_listeners(listeners_names=list(LISTENERS)))
    else:
        asyncio.run(run_several_listeners(listeners_names=listener_names))


if __name__ == '__main__':
    start_listener()
