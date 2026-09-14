import os
import json
from typing import Optional, Dict, Tuple, Any

from .abstract_listener import AbstractListener


POLYMARKET_WS_URL = os.getenv("POLYMARKET_WS_URL", "wss://ws-live-data.polymarket.com/")
SUB_MESSAGE = json.dumps({
    "action": "subscribe",
    "subscriptions": [
        {
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "filters": "{\"symbol\":\"btc/usd\"}"
        }
    ]
})
SOURCE_NAME = "PLM"


class PolymarketListener(AbstractListener):
    def __init__(self, **kwargs):
        super().__init__(
            ws_url=POLYMARKET_WS_URL,
            sub_message=SUB_MESSAGE,
            source_name=SOURCE_NAME,
            **kwargs
        )

    async def process_message(self, data: Dict[str, Any]) -> Optional[Tuple[int, float]]:
        if 'data' in data['payload'] and isinstance(data['payload']['data'], list):
            await self._process_hist_data(data)
            return None

        return data['payload']['timestamp'], data['payload']['value']

    async def _process_hist_data(self, data: Dict[str, Any]):
        pass
