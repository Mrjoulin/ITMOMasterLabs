from unittest.mock import MagicMock, patch

from py_clob_client_v2.exceptions import PolyApiException

from src.execution.order_executor import place_market_order

TOKEN_ID = "12345"


def _mock_client(post_order_return=None, post_order_side_effect=None, get_order_return=None, get_order_side_effect=None):
    client = MagicMock()
    client.get_tick_size.return_value = "0.01"
    client.get_neg_risk.return_value = False
    if post_order_side_effect is not None:
        client.create_and_post_market_order.side_effect = post_order_side_effect
    else:
        client.create_and_post_market_order.return_value = post_order_return
    if get_order_side_effect is not None:
        client.get_order.side_effect = get_order_side_effect
    else:
        client.get_order.return_value = get_order_return
    return client


@patch("src.execution.order_executor.get_clob_client")
def test_place_market_order_matched(mock_get_client):
    # Shape actually returned by get_order() for a resolved FAK order: a trade-like
    # record with human-decimal price/size_matched, not raw fixed-point amounts.
    mock_get_client.return_value = _mock_client(
        post_order_return={"orderID": "order-1", "transactionsHashes": ["0xabc"]},
        get_order_return={
            "id": "order-1",
            "status": "MATCHED",
            "original_size": "1.1235",
            "size_matched": "1.12359",
            "price": "0.8901",
        },
    )

    result = place_market_order.fn(token_id=TOKEN_ID, amount_usd=1.0)

    assert result.success is True
    assert result.order_id == "order-1"
    assert result.status == "MATCHED"
    assert result.filled_price == 0.8901
    assert result.filled_size == 1.12359
    assert result.tx_hashes == "0xabc"
    assert result.error_message is None


@patch("src.execution.order_executor.get_clob_client")
def test_place_market_order_zero_fill_is_not_success(mock_get_client):
    mock_get_client.return_value = _mock_client(
        post_order_return={"orderID": "order-2"},
        get_order_return={
            "id": "order-2",
            "status": "UNMATCHED",
            "original_size": "1.1235",
            "size_matched": "0",
            "price": "0",
        },
    )

    result = place_market_order.fn(token_id=TOKEN_ID, amount_usd=1.0)

    assert result.success is False
    assert result.filled_price is None
    assert result.filled_size is None
    assert result.error_message == "order not filled"


@patch("src.execution.order_executor.get_clob_client")
def test_place_market_order_partial_fill_is_success(mock_get_client):
    # FAK (unlike FOK) can partially fill: size_matched < original_size, still a success.
    mock_get_client.return_value = _mock_client(
        post_order_return={"orderID": "order-5", "transactionsHashes": ["0xdef"]},
        get_order_return={
            "id": "order-5",
            "status": "MATCHED",
            "original_size": "1.1235",
            "size_matched": "0.5",
            "price": "0.8901",
        },
    )

    result = place_market_order.fn(token_id=TOKEN_ID, amount_usd=1.0)

    assert result.success is True
    assert result.filled_price == 0.8901
    assert result.filled_size == 0.5


@patch("src.execution.order_executor.get_clob_client")
def test_place_market_order_matched_raw_amount_fallback_shape(mock_get_client):
    # Some get_order() responses (e.g. an open/resting order) may come back with raw
    # fixed-point amounts instead of size_matched/price - covered as a fallback.
    mock_get_client.return_value = _mock_client(
        post_order_return={"orderID": "order-4", "transactionsHashes": ["0xabc"]},
        get_order_return={
            "order_id": "order-4",
            "status": "matched",
            "makerAmount": "1000000",
            "takerAmount": "2000000",
            "filledTakerAmount": "2000000",
        },
    )

    result = place_market_order.fn(token_id=TOKEN_ID, amount_usd=1.0)

    assert result.success is True
    assert result.filled_price == 0.5
    assert result.filled_size == 2.0


@patch("src.execution.order_executor.get_clob_client")
def test_place_market_order_no_order_id_is_rejected(mock_get_client):
    mock_get_client.return_value = _mock_client(
        post_order_return={"status": "rejected", "errorMsg": "invalid order version, please use the latest clob-client"}
    )

    result = place_market_order.fn(token_id=TOKEN_ID, amount_usd=1.0)

    assert result.success is False
    assert result.order_id is None
    assert "invalid order version" in result.error_message


@patch("src.execution.order_executor.get_clob_client")
def test_place_market_order_resolution_check_fails_is_unconfirmed(mock_get_client):
    mock_get_client.return_value = _mock_client(
        post_order_return={"orderID": "order-3"},
        get_order_side_effect=RuntimeError("timeout"),
    )

    result = place_market_order.fn(token_id=TOKEN_ID, amount_usd=1.0)

    assert result.success is False
    assert result.order_id == "order-3"
    assert result.status == "UNCONFIRMED"
    assert "timeout" in result.error_message


@patch("src.execution.order_executor.get_clob_client")
def test_place_market_order_api_exception_insufficient_balance(mock_get_client):
    mock_get_client.return_value = _mock_client(
        post_order_side_effect=PolyApiException(error_msg="insufficient balance")
    )

    result = place_market_order.fn(token_id=TOKEN_ID, amount_usd=1.0)

    assert result.success is False
    assert result.status == "REJECTED"
    assert result.error_message == "insufficient balance"
    assert result.order_id is None


@patch("src.execution.order_executor.get_clob_client")
def test_place_market_order_unexpected_error(mock_get_client):
    mock_get_client.return_value = _mock_client(post_order_side_effect=RuntimeError("network down"))

    result = place_market_order.fn(token_id=TOKEN_ID, amount_usd=1.0)

    assert result.success is False
    assert result.status == "ERROR"
    assert "network down" in result.error_message
