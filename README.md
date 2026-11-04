# Quant ML Research — Code Samples

Selected code from a larger quantitative trading project. These examples show how I approach feature engineering, model training, optimisation, and backtest evaluation.

## What to look at

| Area | Code | Tests | Focus |
| --- | --- | --- | --- |
| XGBoost training | `machine_learning/xgboost_binary.py` | `machine_learning/test_ml_training.py` | Model training, early stopping, and validation metrics |
| Time-based data splitting | `machine_learning/chronological_splitting.py` | `machine_learning/test_ml_splitting.py` | Train/validation/test boundaries, purging, and embargoes |
| Feature engineering | `feature_engineering/detection.py` | `feature_engineering/test_support_resistance_detection.py` | Confirmed support and resistance pivots without using future information early |
| Optuna optimisation | `optimisation/study.py` | `optimisation/test_optimisation.py` | Study setup, trial selection, and out-of-sample evaluation |
| Backtesting | `backtesting/engine.py`, `backtesting/metrics.py` | `backtesting/test_backtest_metrics.py` | Trade simulation and daily portfolio returns |

These are selected files for code review, not a standalone runnable package. Some imports refer to modules in the larger project that are not included here.
