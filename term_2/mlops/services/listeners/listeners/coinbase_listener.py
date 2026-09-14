import os
import json
from typing import Optional, Dict, Tuple, Any
from dateutil.parser import isoparse

from .abstract_listener import AbstractListener


COINBASE_WS_URL = os.getenv("COINBASE_WS_URL", "wss://ws-feed.exchange.coinbase.com")
SUB_MESSAGE = json.dumps({
    "type": "subscribe",
    "product_ids": ["BTC-USD"],
    "channels": ["ticker"]
})
SOURCE_NAME = "CNB"


class CoinbaseListener(AbstractListener):
    def __init__(self, **kwargs):
        super().__init__(
            ws_url=COINBASE_WS_URL,
            sub_message=SUB_MESSAGE,
            source_name=SOURCE_NAME,
            **kwargs
        )

    async def process_message(self, data: Dict[str, Any]) -> Optional[Tuple[int, float]]:
        if data.get('type') == 'ticker':
            price = float(data['price'])
            dt = isoparse(data["time"])
            timestamp = round(dt.timestamp() * 1000)
            return timestamp, price
        return None
