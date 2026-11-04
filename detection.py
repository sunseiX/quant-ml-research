"""Detect confirmed pivot highs and lows for support/resistance channels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


PIVOT_SOURCE_HIGH_LOW = "high_low"
SUPPORTED_PIVOT_SOURCES = {PIVOT_SOURCE_HIGH_LOW}


def _validate_settings(*, side_bars: int, source: str) -> None:
    if type(side_bars) is not int or side_bars < 1:
        raise ValueError("pivot_side_bars must be a positive integer")
    if source not in SUPPORTED_PIVOT_SOURCES:
        supported = ", ".join(sorted(SUPPORTED_PIVOT_SOURCES))
        raise ValueError(f"pivot_source must be one of: {supported}")


def build_pivot_candidates(
    candles: Sequence[Mapping[str, object]],
    *,
    side_bars: int,
    source: str = PIVOT_SOURCE_HIGH_LOW,
) -> list[dict[str, object]]:
    """Return pivots confirmed by ``side_bars`` candles on each side.

    A consumer must not use a candidate before ``confirmed_index``. Keeping
    confirmation explicit prevents historical snapshots from seeing pivots
    that were only identifiable using later candles.
    """
    _validate_settings(side_bars=side_bars, source=source)
    if len(candles) < side_bars * 2 + 1:
        return []

    candidates: list[dict[str, object]] = []
    for pivot_index in range(side_bars, len(candles) - side_bars):
        candle = candles[pivot_index]
        window = candles[pivot_index - side_bars : pivot_index + side_bars + 1]
        high = float(candle["high_mid"])
        low = float(candle["low_mid"])
        confirmed_index = pivot_index + side_bars

        is_pivot_high = high == max(float(row["high_mid"]) for row in window)
        is_pivot_low = low == min(float(row["low_mid"]) for row in window)
        if is_pivot_high:
            pivot_type = "pivot_high"
            price = high
        elif is_pivot_low:
            # Pine stores the high when one candle qualifies as both types.
            pivot_type = "pivot_low"
            price = low
        else:
            continue

        candidates.append(
            {
                "pivot_type": pivot_type,
                "pivot_price": price,
                "pivot_index": pivot_index,
                "pivot_time": candle["snapshot_time"],
                "confirmed_index": confirmed_index,
                "confirmed_time": candles[confirmed_index]["snapshot_time"],
            }
        )

    return candidates
