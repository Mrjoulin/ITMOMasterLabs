"""Local Polymarket CLOB order-book reconstruction.

Why this exists: the websocket sends a full `book` snapshot per token plus a stream of
`price_change` deltas. A `price_change` entry tells you the new size at ONE price level
of ONE token's book - it does not hand you a usable top-of-book with sizes. To know what
you could actually trade, you have to keep the ladder locally and read the top off it.

Two rules that the previous collector broke, and that everything here is built around:

1. Outcome identity comes from `asset_id`, NEVER from the `side` field. `side` is the
   BOOK side (BUY = bid, SELL = ask) of that asset's own ladder.
2. Nothing is averaged on the way in. Quotes are stored raw so they stay executable.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Prices arrive as decimal strings ("0.01"); round when keying to avoid float dust.
PRICE_DP = 6
# Cumulative size within this distance of the top, as a liquidity measure.
DEPTH_WINDOW = 0.01

BID_SIDE = "BUY"
ASK_SIDE = "SELL"


@dataclass(frozen=True)
class L1Quote:
    """An immutable, executable top-of-book snapshot for one outcome token."""

    asset_id: str
    outcome: str
    src_ts: int
    collect_ts: int
    best_bid: Optional[float]
    best_ask: Optional[float]
    bid_size: Optional[float]
    ask_size: Optional[float]
    bid_depth_1c: Optional[float]
    ask_depth_1c: Optional[float]
    tick_size: Optional[float]

    @property
    def is_two_sided(self) -> bool:
        return self.best_bid is not None and self.best_ask is not None

    @property
    def mid(self) -> Optional[float]:
        if not self.is_two_sided:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> Optional[float]:
        if not self.is_two_sided:
            return None
        return self.best_ask - self.best_bid

    def quote_key(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """The part of the quote whose change is worth persisting a new row for."""
        return (self.best_bid, self.best_ask, self.bid_size, self.ask_size)


def _to_price(value: Any) -> float:
    return round(float(value), PRICE_DP)


def parse_book_snapshot(msg: Dict[str, Any]) -> Optional[Tuple[str, Dict[float, float], Dict[float, float], Optional[float]]]:
    """Parse a full `book` message -> (asset_id, bids, asks, tick_size).

    The arrays are NOT reliably ordered best-first, so callers must never assume
    position; we build ladders and take max(bid) / min(ask).
    """
    asset_id = msg.get("asset_id")
    if not asset_id:
        return None

    def ladder(entries: Any) -> Dict[float, float]:
        out: Dict[float, float] = {}
        for e in entries or []:
            try:
                size = float(e["size"])
                if size > 0:
                    out[_to_price(e["price"])] = size
            except (KeyError, TypeError, ValueError):
                continue
        return out

    tick = msg.get("tick_size")
    return (
        asset_id,
        ladder(msg.get("bids")),
        ladder(msg.get("asks")),
        float(tick) if tick is not None else None,
    )


def iter_price_changes(msg: Dict[str, Any]) -> Iterator[Tuple[str, str, float, float]]:
    """Yield (asset_id, side, price, size) from a `price_change` message.

    Each entry carries its own asset_id - that is the whole point. Entries without one
    are unusable and skipped rather than guessed at.
    """
    for change in msg.get("price_changes") or []:
        asset_id = change.get("asset_id")
        side = change.get("side")
        if not asset_id or side not in (BID_SIDE, ASK_SIDE):
            continue
        try:
            yield asset_id, side, _to_price(change["price"]), float(change["size"])
        except (KeyError, TypeError, ValueError):
            continue


class LocalBook:
    """Price->size ladders for a single outcome token.

    Deliberately mutable: this is updated a few hundred times a second, so rebuilding
    immutable ladders per delta would be wasteful. Everything it hands OUT (`L1Quote`)
    is immutable.
    """

    __slots__ = ("asset_id", "outcome", "bids", "asks", "tick_size", "src_ts")

    def __init__(self, asset_id: str, outcome: str) -> None:
        self.asset_id = asset_id
        self.outcome = outcome
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.tick_size: Optional[float] = None
        self.src_ts: int = 0

    def apply_snapshot(
        self, bids: Dict[float, float], asks: Dict[float, float],
        tick_size: Optional[float], src_ts: int
    ) -> None:
        self.bids = dict(bids)
        self.asks = dict(asks)
        if tick_size is not None:
            self.tick_size = tick_size
        self.src_ts = max(self.src_ts, src_ts)

    def apply_level(self, side: str, price: float, size: float, src_ts: int) -> None:
        ladder = self.bids if side == BID_SIDE else self.asks
        if size <= 0:
            ladder.pop(price, None)
        else:
            ladder[price] = size
        self.src_ts = max(self.src_ts, src_ts)

    def _depth(self, ladder: Dict[float, float], top: float, is_bid: bool) -> float:
        if is_bid:
            return sum(s for p, s in ladder.items() if p >= top - DEPTH_WINDOW - 1e-9)
        return sum(s for p, s in ladder.items() if p <= top + DEPTH_WINDOW + 1e-9)

    def l1(self, collect_ts: int) -> L1Quote:
        best_bid = max(self.bids) if self.bids else None
        best_ask = min(self.asks) if self.asks else None
        return L1Quote(
            asset_id=self.asset_id,
            outcome=self.outcome,
            src_ts=self.src_ts,
            collect_ts=collect_ts,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_size=self.bids.get(best_bid) if best_bid is not None else None,
            ask_size=self.asks.get(best_ask) if best_ask is not None else None,
            bid_depth_1c=self._depth(self.bids, best_bid, True) if best_bid is not None else None,
            ask_depth_1c=self._depth(self.asks, best_ask, False) if best_ask is not None else None,
            tick_size=self.tick_size,
        )


class BookRegistry:
    """Holds one LocalBook per outcome token of a single market."""

    def __init__(self, asset_to_outcome: Dict[str, str]) -> None:
        self._books: Dict[str, LocalBook] = {
            aid: LocalBook(aid, outcome) for aid, outcome in asset_to_outcome.items()
        }

    def get(self, asset_id: str) -> Optional[LocalBook]:
        return self._books.get(asset_id)

    def apply_message(self, msg: Dict[str, Any]) -> List[str]:
        """Apply one websocket message; return the asset ids it touched.

        Deliberately does NOT derive L1 quotes. The feed runs at ~270 messages/second
        and callers normally only need to know that state moved - deriving quotes here
        that the caller then discards would be the most wasteful thing in this path.
        Call `canonical_quote` / `l1` when a quote is actually needed.
        """
        event_type = msg.get("event_type")
        try:
            src_ts = int(msg.get("timestamp") or 0)
        except (TypeError, ValueError):
            src_ts = 0

        touched: List[str] = []
        if event_type == "book":
            parsed = parse_book_snapshot(msg)
            if parsed is None:
                return []
            asset_id, bids, asks, tick = parsed
            book = self.get(asset_id)
            if book is None:
                return []
            book.apply_snapshot(bids, asks, tick, src_ts)
            touched.append(asset_id)
        elif event_type == "price_change":
            for asset_id, side, price, size in iter_price_changes(msg):
                book = self.get(asset_id)
                if book is None:
                    continue
                book.apply_level(side, price, size, src_ts)
                if asset_id not in touched:
                    touched.append(asset_id)
        return touched

    def snapshot(self, collect_ts: int) -> Dict[str, L1Quote]:
        """Current L1 for every token, keyed by OUTCOME ('UP'/'DOWN')."""
        return {b.outcome: b.l1(collect_ts) for b in self._books.values()}

    def canonical_quote(self, outcome: str, collect_ts: int) -> Optional[L1Quote]:
        """L1 for `outcome`, merged with the mirrored complement book.

        A binary CLOB's two token books are exact mirrors: buying UP at p is the same
        trade as selling DOWN at 1-p. Measured live 2026-07-30: prices AND sizes
        mirrored on 14865/14865 samples. So we only need to STORE one leg - the other is
        derivable - but we merge both ladders when deriving it, so the quote stays correct
        even if a message only touched the complement token.

        Where the two representations disagree, max(bid)/min(ask) wins: that is the
        genuinely best executable price, since either route is available to you.
        """
        own = next((b for b in self._books.values() if b.outcome == outcome), None)
        if own is None:
            return None
        other = next((b for b in self._books.values() if b.outcome != outcome), None)

        bids = dict(own.bids)
        asks = dict(own.asks)
        if other is not None:
            for price, size in other.asks.items():
                mirrored = round(1.0 - price, PRICE_DP)
                bids[mirrored] = max(bids.get(mirrored, 0.0), size)
            for price, size in other.bids.items():
                mirrored = round(1.0 - price, PRICE_DP)
                asks[mirrored] = max(asks.get(mirrored, 0.0), size)

        merged = LocalBook(own.asset_id, outcome)
        merged.apply_snapshot(bids, asks, own.tick_size, max(
            own.src_ts, other.src_ts if other is not None else 0
        ))
        return merged.l1(collect_ts)
