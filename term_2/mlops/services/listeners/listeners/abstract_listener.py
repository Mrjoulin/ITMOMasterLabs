from typing import Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import time
import json
import os

import websockets
import structlog
import asyncio

from utils.redis_db import AsyncRedisPublisher

LOG_PUBLISHED_EVERY = int(os.getenv("LOG_PUBLISHED_EVERY", -1))


class AbstractListener(ABC):
    def __init__(
            self, ws_url: str, sub_message: str, source_name: str,
            reconnect_timeout: float = 1, redis_client: AsyncRedisPublisher = None
    ):
        self.ws_url = ws_url
        self.sub_message = sub_message
        self.source_name = source_name
        self.reconnect_timeout = reconnect_timeout

        self.redis_client = redis_client or AsyncRedisPublisher()
        self.logger = structlog.get_logger(__name__)

        self.msg_cnt = 0
        self.msg_received_timestamp = 0

    def load_json_message(self, msg: str):
        try:
            return json.loads(msg)
        except Exception as e:
            self.logger.error("unable_to_parse_json", error=str(e), message=msg)
            return None

    @abstractmethod
    async def process_message(self, data: Dict[str, Any]) -> Optional[Tuple[int, float]]:
        pass

    async def listen(self):
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    await ws.send(self.sub_message)
                    self.logger.info("connected_to_ws", url=self.ws_url, sub_message=self.sub_message)

                    async for msg in ws:
                        self.msg_received_timestamp = round(time.time() * 1000)
                        self.msg_cnt += 1
                        try:
                            data = self.load_json_message(msg)

                            if not data:
                                continue

                            result = await self.process_message(data)

                            if result is not None:
                                await self.redis_client.publish(
                                    timestamp=result[0],
                                    value=result[1],
                                    source_name=self.source_name
                                )
                                if LOG_PUBLISHED_EVERY > 0 and self.msg_cnt % LOG_PUBLISHED_EVERY == 0:
                                    self.logger.info(
                                        "Published message", src=self.source_name, ts=result[0], value=result[1]
                                    )
                        except Exception as e:
                            self.logger.error(
                                "unable_to_process_message",
                                error=str(e), msg=msg, msg_cnt=self.msg_cnt, received_ts=self.msg_received_timestamp
                            )
            except Exception as e:
                self.logger.error("connection_exception", error=str(e))
                self.logger.info("reconnecting_in_timeout", timeout=self.reconnect_timeout)
                await asyncio.sleep(self.reconnect_timeout)

    def start(self):
        asyncio.run(self.listen())
