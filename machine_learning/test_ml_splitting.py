"""Verify chronological ML splitting, purging, and embargo behavior."""

import unittest


from quant.machine_learning.training.data.chronological_splitting import (
    build_chronological_assignments,
    validate_split_settings,
)


class MlSplittingTests(unittest.TestCase):
    settings = {
        "train_fraction": 0.5,
        "validation_fraction": 0.25,
        "test_fraction": 0.25,
        "embargo_bars": 1,
    }

    def test_assigns_time_ordered_splits_with_embargo(self):
        samples = [
            {"sample_id": index + 1, "signal_time": f"2026-01-01T0{index}:00:00",
             "exit_time": f"2026-01-01T0{index}:30:00"}
            for index in range(8)
        ]

        assignments, summary = build_chronological_assignments(
            samples,
            resolution="HOUR",
            settings=self.settings,
        )
        assignment_by_id = dict(assignments)

        self.assertEqual(assignment_by_id[1], "TRAIN")
        self.assertEqual(assignment_by_id[5], "EMBARGO_VALIDATION")
        self.assertEqual(assignment_by_id[6], "VALIDATION")
        self.assertEqual(assignment_by_id[7], "EMBARGO_TEST")
        self.assertEqual(assignment_by_id[8], "TEST")
        self.assertEqual(summary["counts"]["TEST"], 1)

    def test_purges_trade_whose_outcome_overlaps_next_split(self):
        samples = [
            {"sample_id": 1, "signal_time": "2026-01-01T00:00:00",
             "exit_time": "2026-01-01T07:00:00"},
            {"sample_id": 2, "signal_time": "2026-01-01T01:00:00",
             "exit_time": "2026-01-01T01:30:00"},
            {"sample_id": 3, "signal_time": "2026-01-01T02:00:00",
             "exit_time": "2026-01-01T02:30:00"},
            {"sample_id": 4, "signal_time": "2026-01-01T03:00:00",
             "exit_time": "2026-01-01T03:30:00"},
            {"sample_id": 5, "signal_time": "2026-01-01T04:00:00",
             "exit_time": "2026-01-01T04:30:00"},
            {"sample_id": 6, "signal_time": "2026-01-01T05:00:00",
             "exit_time": "2026-01-01T05:30:00"},
            {"sample_id": 7, "signal_time": "2026-01-01T06:00:00",
             "exit_time": "2026-01-01T06:30:00"},
            {"sample_id": 8, "signal_time": "2026-01-01T07:00:00",
             "exit_time": "2026-01-01T07:30:00"},
        ]

        assignments, _ = build_chronological_assignments(
            samples,
            resolution="HOUR",
            settings=self.settings,
        )

        self.assertEqual(dict(assignments)[1], "PURGED_TRAIN")

    def test_uses_explicit_shared_final_holdout_start(self):
        samples = [
            {
                "sample_id": index + 1,
                "signal_time": f"2026-01-01T0{index}:00:00",
                "exit_time": f"2026-01-01T0{index}:30:00",
            }
            for index in range(8)
        ]

        assignments, summary = build_chronological_assignments(
            samples,
            resolution="HOUR",
            settings=self.settings,
            test_start_time="2026-01-01T07:00:00",
        )

        self.assertEqual(summary["test_start_after_embargo"], "2026-01-01T07:00:00")
        self.assertEqual(dict(assignments)[7], "EMBARGO_TEST")
        self.assertEqual(dict(assignments)[8], "TEST")

    def test_rejects_invalid_fraction_total(self):
        with self.assertRaisesRegex(ValueError, "must total 1.0"):
            validate_split_settings(
                self.settings | {"test_fraction": 0.30}
            )


if __name__ == "__main__":
    unittest.main()
