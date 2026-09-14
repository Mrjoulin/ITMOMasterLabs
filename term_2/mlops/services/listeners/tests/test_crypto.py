#!/usr/bin/env python3
"""
Real-time BTC-USD Price Monitor

Connects to Binance and Coinbase WebSocket feeds using the 'websockets' library.
Displays real-time BTC-USD (BTC/USDT on Binance) price updates.
"""

import asyncio
import json
import ssl
from datetime import datetime
from dateutil.parser import isoparse

import websockets


async def binance_websocket_handler():
    """Connect to Binance WebSocket and subscribe to BTC/USDT bookTicker."""
    uri = "wss://fstream.binance.com/public/ws"
    # websockets library automatically handles ping/pong frames
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                # Subscribe to bookTicker stream for BTCUSDT
                subscribe_msg = {
                    "method": "SUBSCRIBE",
                    "params": ["btcusdt@bookTicker"],
                    "id": 1
                }
                await websocket.send(json.dumps(subscribe_msg))
                print("Binance: Subscribed to btcusdt@bookTicker")

                # Continuously receive messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        # bookTicker message contains best bid and ask
                        if 'b' in data and 'a' in data:
                            bid_price = float(data['b'])
                            ask_price = float(data['a'])
                            mid_price = (bid_price + ask_price) / 2
                            timestamp = datetime.now()
                            print(f"[{timestamp.strftime('%H:%M:%S')}] Binance BTC/USDT - "
                                  f"Bid: ${bid_price:,.2f} | Ask: ${ask_price:,.2f} | "
                                  f"Mid: ${mid_price:,.2f}")
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        print(f"Binance processing error: {e}")
        except (websockets.exceptions.ConnectionClosed, Exception) as e:
            print(f"Binance connection error: {e}, reconnecting in 5 seconds...")
            await asyncio.sleep(5)


async def coinbase_websocket_handler():
    """Connect to Coinbase WebSocket and subscribe to BTC-USD ticker."""
    uri = "wss://ws-feed.exchange.coinbase.com"
    # ssl_context = ssl.create_default_context()
    # ssl_context.check_hostname = False
    # ssl_context.verify_mode = ssl.CERT_NONE

    while True:
        try:
            async with websockets.connect(uri) as websocket:
                # Subscribe to ticker channel for BTC-USD
                subscribe_msg = {
                    "type": "subscribe",
                    "product_ids": ["BTC-USD"],
                    "channels": ["ticker"]
                }
                await websocket.send(json.dumps(subscribe_msg))
                print("Coinbase: Subscribed to BTC-USD ticker channel")

                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if data.get('type') == 'ticker':
                            price = float(data['price'])
                            timestamp = isoparse(data["time"])
                            print(f"[{timestamp.strftime('%H:%M:%S')}] Coinbase BTC-USD - Price: ${price:,.2f}")
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        print(f"Coinbase processing error: {e}")
        except (websockets.exceptions.ConnectionClosed, Exception) as e:
            print(f"Coinbase connection error: {e}, reconnecting in 5 seconds...")
            await asyncio.sleep(5)


async def main():
    """Run both exchange handlers concurrently."""
    print("Starting real-time BTC-USD price monitor...")
    print("Press Ctrl+C to stop\n")

    # Run both handlers simultaneously
    await asyncio.gather(
        binance_websocket_handler(),
        coinbase_websocket_handler()
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")