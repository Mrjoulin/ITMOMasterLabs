import asyncio
import websockets
import json
import time
import structlog
import time

POLY_API = "https://api.polymarket.com"
WSS_MARKET = "wss://ws-live-data.polymarket.com/"
logger = structlog.get_logger(__name__)


# ---------- WEBSOCKET ----------
async def market_ws_listener():
    async with websockets.connect(WSS_MARKET) as ws:
        # Subscribe
        sub_msg = {
          "action": "subscribe",
          "subscriptions": [
            {
              "topic": "crypto_prices_chainlink",
              "type": "update",
              "filters": "{\"symbol\":\"btc/usd\"}"
            }
          ]
        }
        await ws.send(json.dumps(sub_msg))
        skip_first = True
        while True:
            msg = await ws.recv()
            if skip_first:
                skip_first = False
                continue
            data = json.loads(msg)
            try:
                orig_ts = data['payload']['timestamp'] / 1000
                ts_diff = data["timestamp"] / 1000 - orig_ts
                my_ts = time.time()
                logger.info("new_message", ts_diff=ts_diff, my_ts_diff=my_ts-orig_ts)
            except Exception as e:
                logger.info("new_message_parse_error", data=data)

if __name__ == "__main__":
    asyncio.run(market_ws_listener())
