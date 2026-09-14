"""Tests for what the collector persists and publishes: a single UP leg with
tick-encoded prices, and a Redis payload that derives the DOWN leg."""

import json

import pytest

from clients.order_book_state import L1Quote
from polymarket_orderbook import (
    HEARTBEAT_MS,
    MIN_STORE_INTERVAL_MS,
    build_subscription,
    l1_to_row,
    redis_payload,
    should_store,
    to_price_ticks,
)
from utils.postgres import ORDER_BOOK_COLUMNS


def quote(**kw) -> L1Quote:
    base = {
        "asset_id": "tok", "outcome": "UP", "src_ts": 1_000_000, "collect_ts": 1_000_040,
        "best_bid": 0.47, "best_ask": 0.48, "bid_size": 448.6, "ask_size": 268.9,
        "bid_depth_1c": 671.5, "ask_depth_1c": 478.6, "tick_size": 0.01,
    }
    base.update(kw)
    return L1Quote(**base)


@pytest.mark.unit
@pytest.mark.parametrize(("price", "ticks"), [
    (0.47, 4700), (0.48, 4800), (0.01, 100), (0.99, 9900),
    (0.005, 50), (0.0001, 1), (None, None),
])
def test_to_price_ticks(price, ticks):
    assert to_price_ticks(price) == ticks


@pytest.mark.unit
def test_price_ticks_fit_smallint():
    # SMALLINT max is 32767; a probability can never exceed 1.0 -> 10000
    assert to_price_ticks(1.0) == 10000


@pytest.mark.unit
def test_price_ticks_preserve_thousandth_ticks():
    """Polymarket uses 0.001 ticks near the extremes (observed up_ask 0.9990 live),
    so a cents-based encoding would silently truncate real quotes."""
    assert to_price_ticks(0.999) == 9990
    assert to_price_ticks(0.001) == 10


@pytest.mark.unit
def test_row_matches_declared_columns():
    """The INSERT uses an explicit column list, so the row must supply exactly those."""
    row = l1_to_row(quote(), event_start_ts=1785415800)
    # collect_ts becomes the `time` column, everything else is a declared column
    assert set(row) == set(ORDER_BOOK_COLUMNS) | {"collect_ts"}


@pytest.mark.unit
def test_row_encodes_up_leg_only():
    row = l1_to_row(quote(), event_start_ts=1785415800)
    assert row["up_bid_t"] == 4700
    assert row["up_ask_t"] == 4800
    assert row["up_bid_size"] == 448.6
    assert row["src_lag_ms"] == 40
    # no redundant fields: the DOWN leg, asset id and duplicate timestamps are all gone
    assert not any(k.startswith("down_") for k in row)
    assert "asset_id" not in row
    assert "outcome" not in row
    assert "tick_size" not in row
    assert "src_ts" not in row


@pytest.mark.unit
def test_row_tolerates_one_sided_book():
    row = l1_to_row(quote(best_bid=None, bid_size=None, bid_depth_1c=None), 1785415800)
    assert row["up_bid_t"] is None
    assert row["up_ask_t"] == 4800


@pytest.mark.unit
def test_missing_src_ts_yields_null_lag_not_garbage():
    row = l1_to_row(quote(src_ts=0), event_start_ts=1785415800)
    assert row["src_lag_ms"] is None


@pytest.mark.unit
def test_should_store_on_change_after_rate_limit():
    q = quote()
    assert should_store(q, last_key=None, last_ms=0, collect_ts=MIN_STORE_INTERVAL_MS)


@pytest.mark.unit
def test_should_not_store_unchanged_quote_before_heartbeat():
    q = quote()
    assert not should_store(q, last_key=q.quote_key(), last_ms=1000, collect_ts=2000)


@pytest.mark.unit
def test_should_store_unchanged_quote_on_heartbeat():
    q = quote()
    assert should_store(q, last_key=q.quote_key(), last_ms=1000,
                        collect_ts=1000 + HEARTBEAT_MS)


@pytest.mark.unit
def test_should_not_store_change_inside_rate_limit():
    q = quote()
    assert not should_store(q, last_key=("x", "y", "z", "w"), last_ms=1000,
                            collect_ts=1000 + MIN_STORE_INTERVAL_MS - 1)


@pytest.mark.unit
def test_subscription_includes_both_tokens():
    sub = build_subscription({"a": "UP", "b": "DOWN"})
    assert '"a"' in sub
    assert '"b"' in sub


# --- Redis payload: both legs derived from one canonical UP quote -------------------

OUTCOME_ASSETS = {"UP": "up-token", "DOWN": "down-token"}


def payload(**kw):
    return json.loads(redis_payload(quote(**kw), OUTCOME_ASSETS,
                                    event_start_ts=1785415800, collect_ts=1_000_100))


@pytest.mark.unit
def test_payload_carries_asset_id_per_leg():
    """make_bets reads the token from here instead of querying Postgres."""
    p = payload()
    assert p["UP"]["asset_id"] == "up-token"
    assert p["DOWN"]["asset_id"] == "down-token"


@pytest.mark.unit
def test_payload_down_leg_is_derived_mirror():
    p = payload()          # UP bid 0.47 / ask 0.48
    assert p["DOWN"]["best_bid"] == pytest.approx(0.52)   # 1 - up_ask
    assert p["DOWN"]["best_ask"] == pytest.approx(0.53)   # 1 - up_bid
    assert p["DOWN"]["bid_size"] == 268.9                 # mirrors up ask_size
    assert p["DOWN"]["ask_size"] == 448.6                 # mirrors up bid_size


@pytest.mark.unit
def test_payload_legs_sum_to_one_by_construction():
    for bid, ask in [(0.47, 0.48), (0.01, 0.02), (0.99, 0.999), (0.5, 0.51)]:
        p = payload(best_bid=bid, best_ask=ask)
        assert p["UP"]["best_ask"] + p["DOWN"]["best_bid"] == pytest.approx(1.0)
        assert p["UP"]["best_bid"] + p["DOWN"]["best_ask"] == pytest.approx(1.0)


@pytest.mark.unit
def test_payload_one_sided_book_yields_nulls_not_zeros():
    p = payload(best_bid=None, bid_size=None)
    assert p["UP"]["best_bid"] is None
    assert p["UP"]["mid"] is None       # must not silently become ask/2
    assert p["UP"]["spread"] is None
    assert p["DOWN"]["best_ask"] is None
    assert p["DOWN"]["best_bid"] == pytest.approx(0.52)


@pytest.mark.unit
def test_payload_declares_schema_and_event():
    p = payload()
    assert p["schema"] == "l1.v1"
    assert p["event_start_ts"] == 1785415800
    assert p["collect_ts"] == 1_000_100
