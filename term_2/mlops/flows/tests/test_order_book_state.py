"""Tests for local order-book reconstruction.

The shapes here are copied from real messages captured off
wss://ws-subscriptions-frontend-clob.polymarket.com/ws/market on 2026-07-30, including
the exact case that produced the original bug: a price_change batch whose side=BUY entry
belongs to the DOWN token and whose side=SELL entry belongs to the UP token.
"""

import pytest

from clients.order_book_state import (
    BookRegistry,
    LocalBook,
    iter_price_changes,
    parse_book_snapshot,
)

UP_TOKEN = "92051065648500377265509348888365317688708395866730250575102499657669945618210"
DOWN_TOKEN = "59100000000000000000000000000000000000000000000000000000000009168366042"
ASSET_TO_OUTCOME = {UP_TOKEN: "UP", DOWN_TOKEN: "DOWN"}

# Real capture: side is the BOOK side of each asset's own ladder, and the BUY entry here
# is DOWN's bid. Keying on `side` (the old bug) would file DOWN's data under "up".
REAL_PRICE_CHANGE = {
    "event_type": "price_change",
    "market": "0xmarket",
    "timestamp": "1785410450123",
    "price_changes": [
        {"asset_id": DOWN_TOKEN, "side": "BUY", "price": "0.96", "size": "15114.38",
         "best_bid": "0.96", "best_ask": "0.97", "hash": "h1"},
        {"asset_id": UP_TOKEN, "side": "SELL", "price": "0.04", "size": "15114.38",
         "best_bid": "0.03", "best_ask": "0.04", "hash": "h2"},
    ],
}

REAL_BOOK_UP = {
    "event_type": "book",
    "asset_id": UP_TOKEN,
    "timestamp": "1785410450000",
    "tick_size": "0.01",
    # deliberately not best-first, matching the real feed
    "bids": [{"price": "0.01", "size": "10901.23"}, {"price": "0.03", "size": "500.0"},
             {"price": "0.02", "size": "1368.75"}],
    "asks": [{"price": "0.99", "size": "15109.38"}, {"price": "0.04", "size": "250.0"},
             {"price": "0.98", "size": "2695.99"}],
}


@pytest.mark.unit
def test_parse_book_snapshot_ignores_array_order():
    asset_id, bids, asks, tick = parse_book_snapshot(REAL_BOOK_UP)
    assert asset_id == UP_TOKEN
    assert tick == 0.01
    # best bid is the MAX bid and best ask the MIN ask, regardless of array position
    assert max(bids) == 0.03
    assert min(asks) == 0.04


@pytest.mark.unit
def test_parse_book_snapshot_drops_zero_size_levels():
    msg = dict(REAL_BOOK_UP, bids=[{"price": "0.05", "size": "0"},
                                   {"price": "0.03", "size": "10"}])
    _, bids, _, _ = parse_book_snapshot(msg)
    assert 0.05 not in bids
    assert bids[0.03] == 10


@pytest.mark.unit
def test_iter_price_changes_yields_asset_id_not_side():
    got = list(iter_price_changes(REAL_PRICE_CHANGE))
    assert len(got) == 2
    by_asset = {aid: (side, price, size) for aid, side, price, size in got}
    # the BUY entry belongs to DOWN -- this is the regression the old code got wrong
    assert by_asset[DOWN_TOKEN][0] == "BUY"
    assert by_asset[UP_TOKEN][0] == "SELL"


@pytest.mark.unit
def test_iter_price_changes_skips_entries_without_asset_id():
    msg = {"price_changes": [{"side": "BUY", "price": "0.5", "size": "1"}]}
    assert list(iter_price_changes(msg)) == []


@pytest.mark.unit
def test_local_book_l1_tracks_top_with_sizes():
    book = LocalBook(UP_TOKEN, "UP")
    _, bids, asks, tick = parse_book_snapshot(REAL_BOOK_UP)
    book.apply_snapshot(bids, asks, tick, src_ts=1785410450000)
    q = book.l1(collect_ts=1785410450500)
    assert q.best_bid == 0.03
    assert q.best_ask == 0.04
    assert q.bid_size == 500.0
    assert q.ask_size == 250.0
    assert q.mid == pytest.approx(0.035)
    assert q.spread == pytest.approx(0.01)
    assert q.outcome == "UP"


@pytest.mark.unit
def test_local_book_removes_level_on_zero_size():
    book = LocalBook(UP_TOKEN, "UP")
    _, bids, asks, tick = parse_book_snapshot(REAL_BOOK_UP)
    book.apply_snapshot(bids, asks, tick, src_ts=1)
    assert book.l1(2).best_bid == 0.03
    book.apply_level("BUY", 0.03, 0.0, src_ts=3)
    # top bid falls back to the next level down
    assert book.l1(4).best_bid == 0.02


@pytest.mark.unit
def test_depth_within_one_cent():
    book = LocalBook(UP_TOKEN, "UP")
    book.apply_snapshot(
        bids={0.50: 10.0, 0.49: 5.0, 0.45: 100.0},
        asks={0.51: 7.0, 0.52: 3.0, 0.60: 100.0},
        tick_size=0.01, src_ts=1,
    )
    q = book.l1(2)
    assert q.bid_depth_1c == pytest.approx(15.0)   # 0.50 + 0.49, not 0.45
    assert q.ask_depth_1c == pytest.approx(10.0)   # 0.51 + 0.52, not 0.60


@pytest.mark.unit
def test_registry_routes_price_change_to_correct_outcome():
    reg = BookRegistry(ASSET_TO_OUTCOME)
    touched = reg.apply_message(REAL_PRICE_CHANGE)
    assert set(touched) == {UP_TOKEN, DOWN_TOKEN}
    by_outcome = reg.snapshot(collect_ts=1)
    # DOWN got the BUY entry -> it is DOWN's bid at 0.96
    assert by_outcome["DOWN"].best_bid == 0.96
    # UP got the SELL entry -> it is UP's ask at 0.04
    assert by_outcome["UP"].best_ask == 0.04


@pytest.mark.unit
def test_apply_message_returns_ids_not_quotes():
    """The hot path runs at ~270 msg/s; apply_message must not derive quotes it may
    never be asked for."""
    reg = BookRegistry(ASSET_TO_OUTCOME)
    touched = reg.apply_message(REAL_PRICE_CHANGE)
    assert all(isinstance(t, str) for t in touched)


@pytest.mark.unit
def test_registry_legs_are_complementary():
    """UP ask and DOWN bid must sum to 1 -- the sanity check that direction is intact."""
    reg = BookRegistry(ASSET_TO_OUTCOME)
    reg.apply_message(REAL_PRICE_CHANGE)
    snap = reg.snapshot(collect_ts=2)
    assert snap["UP"].best_ask + snap["DOWN"].best_bid == pytest.approx(1.0)


@pytest.mark.unit
def test_registry_ignores_unknown_asset_and_event_types():
    reg = BookRegistry(ASSET_TO_OUTCOME)
    assert reg.apply_message({"event_type": "last_trade_price", "price": "0.5"}) == []
    stranger = {"event_type": "price_change", "timestamp": "1",
                "price_changes": [{"asset_id": "other", "side": "BUY",
                                   "price": "0.5", "size": "1"}]}
    assert reg.apply_message(stranger) == []


@pytest.mark.unit
def test_canonical_quote_merges_mirrored_complement():
    """Only the UP leg is stored, so deriving it must also work from DOWN's ladder."""
    reg = BookRegistry(ASSET_TO_OUTCOME)
    # only the DOWN token ticks: DOWN bid 0.96 / ask 0.97
    down_only = {
        "event_type": "price_change", "timestamp": "1",
        "price_changes": [
            {"asset_id": DOWN_TOKEN, "side": "BUY", "price": "0.96", "size": "500"},
            {"asset_id": DOWN_TOKEN, "side": "SELL", "price": "0.97", "size": "300"},
        ],
    }
    reg.apply_message(down_only)
    up = reg.canonical_quote("UP", collect_ts=11)
    # UP must be the exact mirror even though UP's own ladder never received anything
    assert up.best_bid == pytest.approx(0.03)   # 1 - 0.97
    assert up.best_ask == pytest.approx(0.04)   # 1 - 0.96
    assert up.bid_size == 300.0                 # mirrors DOWN's ask size
    assert up.ask_size == 500.0                 # mirrors DOWN's bid size


@pytest.mark.unit
def test_canonical_quote_takes_best_when_legs_disagree():
    reg = BookRegistry(ASSET_TO_OUTCOME)
    reg.apply_message({
        "event_type": "price_change", "timestamp": "1",
        "price_changes": [
            {"asset_id": UP_TOKEN, "side": "SELL", "price": "0.40", "size": "10"},
            # DOWN bid 0.65 implies an UP ask of 0.35 -- strictly better for a buyer
            {"asset_id": DOWN_TOKEN, "side": "BUY", "price": "0.65", "size": "20"},
        ],
    })
    up = reg.canonical_quote("UP", collect_ts=2)
    assert up.best_ask == pytest.approx(0.35)


@pytest.mark.unit
def test_canonical_quote_unknown_outcome_is_none():
    reg = BookRegistry(ASSET_TO_OUTCOME)
    assert reg.canonical_quote("SIDEWAYS", collect_ts=1) is None


@pytest.mark.unit
def test_snapshot_keys_are_outcomes_not_sides():
    reg = BookRegistry(ASSET_TO_OUTCOME)
    reg.apply_message(REAL_BOOK_UP)
    snap = reg.snapshot(collect_ts=2)
    assert sorted(snap) == ["DOWN", "UP"]
    assert snap["UP"].asset_id == UP_TOKEN
    assert snap["DOWN"].asset_id == DOWN_TOKEN


@pytest.mark.unit
def test_one_sided_book_reports_none_not_zero():
    book = LocalBook(UP_TOKEN, "UP")
    book.apply_snapshot(bids={}, asks={0.6: 5.0}, tick_size=0.01, src_ts=1)
    q = book.l1(2)
    assert q.best_bid is None
    assert q.best_ask == 0.6
    assert q.is_two_sided is False
    assert q.mid is None          # must not silently become 0.3
    assert q.spread is None
