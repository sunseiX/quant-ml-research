"""Train and evaluate the baseline XGBoost confirmation model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import xgboost as xgb

from quant.machine_learning.training.data.model_matrix import ModelMatrix


@dataclass(frozen=True)
class TrainingResult:
    booster: xgb.Booster
    best_iteration: int
    validation_probabilities: np.ndarray
    validation_metrics: dict[str, float | int]


def _validate_binary_labels(labels: np.ndarray, name: str) -> None:
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{name} labels must be a one-dimensional binary array")
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{name} labels must contain both classes")


def binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities.astype(float), 1e-15, 1.0 - 1e-15)
    return float(
        -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    )


def roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Calculate ROC-AUC with average ranks for tied probabilities."""
    order = np.argsort(probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    ranks = np.empty(len(probabilities), dtype=float)
    start = 0
    while start < len(probabilities):
        end = start + 1
        while (
            end < len(probabilities)
            and sorted_probabilities[end] == sorted_probabilities[start]
        ):
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        raise ValueError("ROC-AUC requires both label classes")
    rank_sum = float(ranks[positives].sum())
    return (
        rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[order]
    sorted_probabilities = probabilities[order]
    positive_count = int(sorted_labels.sum())
    if not positive_count:
        raise ValueError("Average precision requires positive labels")
    # Evaluate at the end of each tied-score group. Arbitrary ordering within
    # equal predictions must not create artificial ranking skill.
    group_ends = np.r_[
        np.flatnonzero(np.diff(sorted_probabilities)),
        len(sorted_probabilities) - 1,
    ]
    cumulative_positives = np.cumsum(sorted_labels)[group_ends]
    predicted_positives = group_ends + 1
    precision = cumulative_positives / predicted_positives
    recall = cumulative_positives / positive_count
    recall_increase = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increase * precision))


def evaluate_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int]:
    _validate_binary_labels(labels, "Evaluation")
    if probabilities.shape != labels.shape:
        raise ValueError("Probabilities and labels must have equal shape")
    if not np.isfinite(probabilities).all():
        raise ValueError("Evaluation inputs must be finite")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predicted probabilities must be between zero and one")

    return {
        "log_loss": binary_log_loss(labels, probabilities),
        "roc_auc": roc_auc(labels, probabilities),
        "average_precision": average_precision(labels, probabilities),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "base_positive_rate": float(labels.mean()),
        "mean_probability": float(probabilities.mean()),
    }


def train_baseline(
    train: ModelMatrix,
    validation: ModelMatrix,
    *,
    settings: Mapping,
) -> TrainingResult:
    if train.feature_names != validation.feature_names:
        raise ValueError("Train and validation feature orders do not match")
    if train.values.shape[1] != len(train.feature_names):
        raise ValueError("Train matrix width does not match its feature contract")
    if validation.values.shape[1] != len(validation.feature_names):
        raise ValueError("Validation matrix width does not match its feature contract")
    _validate_binary_labels(train.labels, "Train")
    _validate_binary_labels(validation.labels, "Validation")

    parameters = dict(settings)
    num_boost_round = int(parameters.pop("n_estimators"))
    early_stopping_rounds = int(parameters.pop("early_stopping_rounds"))
    parameters["seed"] = int(parameters.pop("random_state"))
    if num_boost_round <= 0 or early_stopping_rounds <= 0:
        raise ValueError("XGBoost rounds and early stopping must be positive")

    train_data = xgb.DMatrix(
        train.values,
        label=train.labels,
        feature_names=list(train.feature_names),
    )
    validation_data = xgb.DMatrix(
        validation.values,
        label=validation.labels,
        feature_names=list(validation.feature_names),
    )
    booster = xgb.train(
        parameters,
        train_data,
        num_boost_round=num_boost_round,
        evals=[(train_data, "train"), (validation_data, "validation")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    best_iteration = int(booster.best_iteration)
    probabilities = booster.predict(
        validation_data,
        iteration_range=(0, best_iteration + 1),
    )
    metrics = evaluate_probabilities(
        validation.labels,
        probabilities,
    )
    return TrainingResult(
        booster=booster,
        best_iteration=best_iteration,
        validation_probabilities=probabilities,
        validation_metrics=metrics,
    )
