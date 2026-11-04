"""Verify baseline ML metrics and XGBoost training contracts."""

import unittest

import numpy as np


from quant.machine_learning.training.data.model_matrix import ModelMatrix
from quant.machine_learning.training.models.xgboost_binary import (
    evaluate_probabilities,
    train_baseline,
)


def model_matrix(split_name: str, offset: float = 0.0) -> ModelMatrix:
    values = np.asarray(
        [[index / 20.0 + offset, (index % 3) / 2.0] for index in range(20)],
        dtype=np.float32,
    )
    labels = np.asarray([0] * 10 + [1] * 10, dtype=np.int8)
    return ModelMatrix(
        split_name=split_name,
        feature_names=("trend", "regime"),
        values=values,
        labels=labels,
        sample_ids=np.arange(1, 21, dtype=np.int64),
        signal_times=tuple(f"2026-01-{index + 1:02d}" for index in range(20)),
        trade_sides=tuple("BUY" if index % 2 else "SELL" for index in range(20)),
        net_r_multiples=np.where(labels == 1, 2.0, -1.0),
    )


class MlTrainingTests(unittest.TestCase):
    def test_probability_metrics_reward_correct_ranking(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
        metrics = evaluate_probabilities(labels, probabilities)

        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)
        self.assertNotIn("diagnostic_threshold", metrics)
        self.assertNotIn("selected_count", metrics)

    def test_constant_probabilities_have_base_rate_average_precision(self):
        labels = np.asarray([0, 0, 0, 1], dtype=np.int8)
        probabilities = np.full(4, 0.25)

        metrics = evaluate_probabilities(labels, probabilities)

        self.assertAlmostEqual(metrics["average_precision"], 0.25)

    def test_trains_without_using_a_test_matrix(self):
        result = train_baseline(
            model_matrix("TRAIN"),
            model_matrix("VALIDATION", offset=0.01),
            settings={
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 2,
                "early_stopping_rounds": 3,
                "random_state": 42,
            },
        )

        self.assertEqual(result.validation_probabilities.shape, (20,))
        self.assertGreaterEqual(result.best_iteration, 0)
        self.assertIn("roc_auc", result.validation_metrics)


if __name__ == "__main__":
    unittest.main()
