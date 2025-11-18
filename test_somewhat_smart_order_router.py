"""Unit tests for the best_price_improvement order-router API"""

# Local library
from somewhat_smart_order_router import best_price_improvement


def test_normal_order():
    """Tests that a noraml buy order returns a valid exchange"""
    exchange = best_price_improvement(
        symbol="AAPL",
        side="B",
        quantity=100,
        limit_price=180.0,
        bid_price=179.9,
        ask_price=180.1,
        bid_size=500,
        ask_size=600,
    )

    assert isinstance(exchange, str)
    assert len(exchange) > 0


def test_corner_case():
    """
    Test that a small sell order still returns a valid exchange.

    This is a corner case with quantity 1 and minimal displayed size.
    """
    exch = best_price_improvement(
        symbol="AAPL",
        side="S",
        quantity=1,
        limit_price=180.0,
        bid_price=179.9,
        ask_price=180.1,
        bid_size=1,
        ask_size=1,
    )
    assert isinstance(exch, str)
