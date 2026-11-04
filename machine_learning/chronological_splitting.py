"""Chronological ML splits with trade-overlap purging and embargo gaps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
import math
import sqlite3

from quant.market_data.resolutions import get_resolution_minutes
from quant.market_data.pipeline_time import now_pipeline_timestamp


SPLIT_TABLE = "ml_confirmation_sample_splits"
MODEL_SPLITS = {"TRAIN", "VALIDATION", "TEST"}


def validate_split_settings(settings: Mapping) -> None:
    fractions = [
        settings.get("train_fraction"),
        settings.get("validation_fraction"),
        settings.get("test_fraction"),
    ]
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in fractions
    ):
        raise ValueError("ML split fractions must be finite numbers greater than zero")
    if not math.isclose(sum(map(float, fractions)), 1.0, abs_tol=1e-9):
        raise ValueError("ML train, validation, and test fractions must total 1.0")

    embargo_bars = settings.get("embargo_bars")
    if (
        not isinstance(embargo_bars, int)
        or isinstance(embargo_bars, bool)
        or embargo_bars < 0
    ):
        raise ValueError("ML embargo_bars must be a non-negative integer")


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_chronological_assignments(
    samples: Sequence[Mapping],
    *,
    resolution: str,
    settings: Mapping,
    test_start_time: str | None = None,
) -> tuple[list[tuple[int, str]], dict]:
    """Assign every sample to a model split or an explicit exclusion reason."""
    validate_split_settings(settings)
    if not samples:
        raise ValueError("Cannot split an empty ML dataset")

    ordered = sorted(samples, key=lambda row: (row["signal_time"], row["sample_id"]))
    first_signal = _parse(ordered[0]["signal_time"])
    last_signal = _parse(ordered[-1]["signal_time"])
    if first_signal >= last_signal:
        raise ValueError("ML samples must span more than one signal timestamp")

    total_span = last_signal - first_signal
    validation_boundary = first_signal + total_span * float(
        settings["train_fraction"]
    )
    embargo = timedelta(
        minutes=get_resolution_minutes(resolution) * int(settings["embargo_bars"])
    )
    if test_start_time is None:
        test_boundary = first_signal + total_span * (
            float(settings["train_fraction"])
            + float(settings["validation_fraction"])
        )
        test_start = test_boundary + embargo
    else:
        test_start = _parse(test_start_time)
        test_boundary = test_start - embargo
        if not first_signal < test_boundary < last_signal:
            raise ValueError(
                "Shared final holdout boundary falls outside the ML sample history"
            )
        development_fraction = (
            float(settings["train_fraction"])
            + float(settings["validation_fraction"])
        )
        validation_boundary = first_signal + (
            test_boundary - first_signal
        ) * (float(settings["train_fraction"]) / development_fraction)
    validation_start = validation_boundary + embargo

    assignments: list[tuple[int, str]] = []
    for sample in ordered:
        signal_time = _parse(sample["signal_time"])
        exit_time = _parse(sample["exit_time"])
        if exit_time < signal_time:
            raise ValueError(f"Sample {sample['sample_id']} exits before its signal")

        if signal_time < validation_boundary:
            split_name = (
                "PURGED_TRAIN"
                if exit_time >= validation_start
                else "TRAIN"
            )
        elif signal_time < validation_start:
            split_name = "EMBARGO_VALIDATION"
        elif signal_time < test_boundary:
            split_name = (
                "PURGED_VALIDATION"
                if exit_time >= test_start
                else "VALIDATION"
            )
        elif signal_time < test_start:
            split_name = "EMBARGO_TEST"
        else:
            split_name = "TEST"
        assignments.append((int(sample["sample_id"]), split_name))

    counts = {}
    for _, split_name in assignments:
        counts[split_name] = counts.get(split_name, 0) + 1
    for split_name in MODEL_SPLITS:
        if counts.get(split_name, 0) == 0:
            raise ValueError(f"Chronological split produced no {split_name} samples")

    return assignments, {
        "first_signal": ordered[0]["signal_time"],
        "last_signal": ordered[-1]["signal_time"],
        "validation_boundary": validation_boundary.isoformat(),
        "validation_start_after_embargo": validation_start.isoformat(),
        "test_boundary": test_boundary.isoformat(),
        "test_start_after_embargo": test_start.isoformat(),
        "counts": counts,
    }


def create_split_build_table(cursor: sqlite3.Cursor) -> None:
    cursor.execute(f"DROP TABLE IF EXISTS {SPLIT_TABLE}__build")
    cursor.execute(
        f"""
        CREATE TABLE {SPLIT_TABLE}__build (
            sample_id INTEGER PRIMARY KEY,
            experiment_name TEXT NOT NULL,
            split_name TEXT NOT NULL CHECK (split_name IN (
                'TRAIN', 'VALIDATION', 'TEST',
                'PURGED_TRAIN', 'PURGED_VALIDATION',
                'EMBARGO_VALIDATION', 'EMBARGO_TEST'
            )),
            created_at TEXT NOT NULL
        )
        """
    )


def replace_split_table(
    cursor: sqlite3.Cursor,
    *,
    experiment_name: str,
    assignments: Sequence[tuple[int, str]],
) -> None:
    created_at = now_pipeline_timestamp()
    cursor.executemany(
        f"""
        INSERT INTO {SPLIT_TABLE}__build (
            sample_id, experiment_name, split_name, created_at
        ) VALUES (?, ?, ?, ?)
        """,
        [
            (sample_id, experiment_name, split_name, created_at)
            for sample_id, split_name in assignments
        ],
    )
    cursor.execute(f"DROP TABLE IF EXISTS {SPLIT_TABLE}")
    cursor.execute(f"ALTER TABLE {SPLIT_TABLE}__build RENAME TO {SPLIT_TABLE}")
