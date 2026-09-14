import time
from datetime import datetime
from typing import Union, Optional

import asyncio
import threading


def get_cur_ts(precision: Optional[str] = None) -> Union[int, float]:
    if precision is None:
        return time.time()
    elif precision.startswith("sec"):
        return round(time.time())
    elif precision.startswith("milli"):
        return round(time.time() * 1000)
    else:
        return time.time()


def ts_to_dt(ts: int, to_str: bool = True) -> Union[datetime, str]:
    if ts > 1e12:
        ts = ts / 1000
    if to_str:
        return str(datetime.fromtimestamp(ts))
    else:
        return datetime.fromtimestamp(ts)


def run_in_thread(coroutine):
    threading.Thread(
        target=asyncio.run, args=(coroutine,), daemon=True
    ).start()


def queue_name(prefix: str, window_start: Union[int, str]) -> str:
    if isinstance(window_start, int) and window_start > 1e12:
        window_start = window_start // 1000
    return ":".join((prefix, str(window_start)))
