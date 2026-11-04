"""Tests for confirmed support/resistance pivot detection."""

import unittest

from quant.features.support_resistance.detection import build_pivot_candidates


def candles(values: list[tuple[float, float]]) -> list[dict[str, object]]:
    return [
        {"snapshot_time": f"t{index}", "high_mid": high, "low_mid": low}
        for index, (high, low) in enumerate(values)
    ]


class PivotDetectionTests(unittest.TestCase):
    def test_detects_high_and_records_when_it_becomes_confirmed(self):
        result = build_pivot_candidates(
            candles([(1, 0), (2, 1), (5, 2), (3, 1), (2, 0)]),
            side_bars=2,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["pivot_type"], "pivot_high")
        self.assertEqual(result[0]["pivot_price"], 5)
        self.assertEqual(result[0]["pivot_index"], 2)
        self.assertEqual(result[0]["confirmed_index"], 4)
        self.assertEqual(result[0]["confirmed_time"], "t4")

    def test_detects_pivot_low(self):
        result = build_pivot_candidates(
            candles([(5, 4), (4, 3), (3, 0), (4, 2), (5, 3)]),
            side_bars=2,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["pivot_type"], "pivot_low")
        self.assertEqual(result[0]["pivot_price"], 0)

    def test_high_takes_precedence_when_candle_is_both_pivot_types(self):
        result = build_pivot_candidates(
            candles([(2, 1), (5, 0), (2, 1)]),
            side_bars=1,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["pivot_type"], "pivot_high")
        self.assertEqual(result[0]["pivot_price"], 5)

    def test_requires_complete_left_and_right_windows(self):
        result = build_pivot_candidates(
            candles([(2, 1), (5, 0), (2, 1)]),
            side_bars=2,
        )

        self.assertEqual(result, [])

    def test_rejects_unsupported_source(self):
        with self.assertRaisesRegex(ValueError, "pivot_source"):
            build_pivot_candidates(
                candles([(2, 1), (5, 0), (2, 1)]),
                side_bars=1,
                source="close_open",
            )


if __name__ == "__main__":
    unittest.main()
