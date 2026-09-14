from .abstract_listener import AbstractListener

try:
    from .polymarket_listener import PolymarketListener
except ImportError:
    PolymarketListener = None
try:
    from .binance_listener import BinanceListener
except ImportError:
    BinanceListener = None
try:
    from .coinbase_listener import CoinbaseListener
except ImportError:
    CoinbaseListener = None

LISTENERS = {
    "polymarket": PolymarketListener,
    "binance": BinanceListener,
    "coinbase": CoinbaseListener
}

if not any(LISTENERS.values()):
    raise ImportError("No listeners found")
