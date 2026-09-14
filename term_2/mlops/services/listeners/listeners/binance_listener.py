import os
import json
from typing import Optional, Dict, Tuple, Any

from .abstract_listener import AbstractListener


BINANCE_WS_URL = os.getenv("BINANCE_WS_URL", "wss://fstream.binance.com/public/ws")
SUB_MESSAGE = json.dumps({
    "method": "SUBSCRIBE",
    "params": ["btcusdt@bookTicker"],
    "id": 1
})
SOURCE_NAME = "BIN"


class BinanceListener(AbstractListener):
    def __init__(self, **kwargs):
        super().__init__(
            ws_url=BINANCE_WS_URL,
            sub_message=SUB_MESSAGE,
            source_name=SOURCE_NAME,
            **kwargs
        )

    async def process_message(self, data: Dict[str, Any]) -> Optional[Tuple[int, float]]:
        if 'b' in data and 'a' in data:
            bid_price, ask_price = float(data['b']), float(data['a'])
            mid_price = (bid_price + ask_price) / 2
            return self.msg_received_timestamp, mid_price
        return None