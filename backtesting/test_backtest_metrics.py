import unittest

from quant.backtesting.metrics import build_daily_mark_to_market_returns


class MarkToMarketReturnTests(unittest.TestCase):
    def test_long_position_is_marked_daily_then_replaced_by_realized_pnl(self):
        trade = {
            "epic": "GOLD",
            "trade_side": "BUY",
            "entry_price": 100.0,
            "entry_time": "2026-01-01T10:00:00",
            "exit_time": "2026-01-03T10:00:00",
            "pnl_amount": 40.0,
            "order_size": 10.0,
        }
        closes = [
            {"epic": "GOLD", "candle_day": "2026-01-01", "close_bid": 101.0, "close_ask": 102.0},
            {"epic": "GOLD", "candle_day": "2026-01-02", "close_bid": 103.0, "close_ask": 104.0},
            {"epic": "GOLD", "candle_day": "2026-01-03", "close_bid": 104.0, "close_ask": 105.0},
        ]

        returns = build_daily_mark_to_market_returns(
            trade_rows=[trade],
            completed_days=["2026-01-01", "2026-01-02", "2026-01-03"],
            daily_close_rows=closes,
            starting_capital=1_000.0,
        )

        self.assertEqual([day for day, _ in returns], [
            "2026-01-01", "2026-01-02", "2026-01-03"
        ])
        self.assertAlmostEqual(returns[0][1], 1010 / 1000 - 1)
        self.assertAlmostEqual(returns[1][1], 1030 / 1010 - 1)
        self.assertAlmostEqual(returns[2][1], 1040 / 1030 - 1)

    def test_short_position_uses_ask_and_carries_last_close(self):
        trade = {
            "epic": "EURUSD",
            "trade_side": "SELL",
            "entry_price": 100.0,
            "entry_time": "2026-01-01T10:00:00",
            "exit_time": None,
            "pnl_amount": 0.0,
            "order_size": 10.0,
        }
        closes = [
            {"epic": "EURUSD", "candle_day": "2026-01-01", "close_bid": 97.0, "close_ask": 98.0},
        ]

        returns = build_daily_mark_to_market_returns(
            trade_rows=[trade],
            completed_days=["2026-01-01", "2026-01-02"],
            daily_close_rows=closes,
            starting_capital=1_000.0,
        )

        self.assertAlmostEqual(returns[0][1], 1020 / 1000 - 1)
        self.assertAlmostEqual(returns[1][1], 0.0)

    def test_closed_trade_does_not_require_a_daily_mark(self):
        returns = build_daily_mark_to_market_returns(
            trade_rows=[{
                "epic": "GOLD",
                "trade_side": "BUY",
                "entry_price": 100.0,
                "entry_time": "2026-01-01T10:00:00",
                "exit_time": "2026-01-01T11:00:00",
                "pnl_amount": 25.0,
                "order_size": 10.0,
            }],
            completed_days=["2026-01-01"],
            daily_close_rows=[],
            starting_capital=1_000.0,
        )

        self.assertAlmostEqual(returns[0][1], 0.025)


if __name__ == "__main__":
    unittest.main()
