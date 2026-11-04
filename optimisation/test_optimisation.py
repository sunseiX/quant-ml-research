from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import optuna
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


from quant.config import CAPITAL_EPICS
from quant.optimisation import analyser
from quant.optimisation import experiment_builder
from quant.optimisation import experiment_executor
from quant.optimisation import repository
from quant.optimisation import search_space as optimisation_search_space
from quant.optimisation import study as optimisation_study


class SearchSpaceTemplateTests(unittest.TestCase):
    def test_loads_sr_bounce_sl_tp_template(self) -> None:
        search_space = optimisation_search_space.load_search_space(
            "sr_bounce_sl_tp"
        )

        self.assertEqual(len(search_space), 2 * len(CAPITAL_EPICS))
        for epic in CAPITAL_EPICS:
            self.assertIn(
                "POSITIONING_METHOD_SETTINGS.fixed_rr.epic_overrides."
                f"{epic}.base_stop_loss_pct",
                search_space,
            )

    def test_loads_sr_bounce_rsi_template(self) -> None:
        search_space = optimisation_search_space.load_search_space(
            "sr_bounce_rsi"
        )

        self.assertEqual(len(search_space), 4)
        self.assertIn(
            "STRATEGY_CONFIRMATION_SETTINGS.rsi_momentum.minimum_buy_rsi",
            search_space,
        )

    def test_rejects_condition_on_later_parameter(self) -> None:
        search_space = {
            "dependent": {
                "type": "int",
                "low": 1,
                "high": 2,
                "condition": {"parameter": "controller", "equals": True},
            },
            "controller": {
                "type": "categorical",
                "values": [True, False],
            },
        }

        with self.assertRaisesRegex(ValueError, "earlier parameter"):
            optimisation_search_space.validate_search_space_conditions(
                search_space
            )

    def test_rejects_unknown_template(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown optimisation"):
            optimisation_search_space.load_search_space("missing_template")

    def test_rejects_invalid_template_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "lowercase letters"):
            optimisation_search_space.load_search_space("../unsafe")

    def test_loaded_template_is_an_independent_copy(self) -> None:
        parameter_name = (
            "POSITIONING_METHOD_SETTINGS.fixed_rr.epic_overrides."
            f"{CAPITAL_EPICS[0]}.base_stop_loss_pct"
        )
        first_load = optimisation_search_space.load_search_space(
            "sr_bounce_sl_tp"
        )
        first_load[parameter_name]["low"] = 99

        second_load = optimisation_search_space.load_search_space(
            "sr_bounce_sl_tp"
        )

        self.assertEqual(second_load[parameter_name]["low"], 0.0015)


class ExperimentBuilderTests(unittest.TestCase):
    def test_builds_one_experiment_from_an_optuna_trial(self) -> None:
        search_space = {
            "POSITIONING_METHOD_SETTINGS.fixed_rr.epic_overrides.GOLD.base_stop_loss_pct": {
                "type": "float",
                "low": 0.003,
                "high": 0.004,
                "step": 0.0001,
                "affects": "positioning",
            },
            "POSITIONING_METHOD_SETTINGS.fixed_rr.epic_overrides.GOLD.base_take_profit_pct": {
                "type": "float",
                "low": 0.006,
                "high": 0.007,
                "step": 0.0001,
                "affects": "positioning",
            },
        }

        class FakeTrial:
            number = 2

            @staticmethod
            def suggest_float(parameter_name, low, high, step=None):
                return high

        with patch.object(experiment_builder, "SEARCH_SPACE", search_space):
            experiment = experiment_builder.build_experiment(FakeTrial())

        self.assertEqual(experiment["experiment_id"], "trial_003")
        self.assertEqual(
            experiment["combo_values"],
            {
                parameter_name: parameter_config["high"]
                for parameter_name, parameter_config in search_space.items()
            },
        )
        gold_settings = experiment["overrides"]["POSITIONING_METHOD_SETTINGS"][
            "fixed_rr"
        ]["epic_overrides"]["GOLD"]
        self.assertEqual(gold_settings["base_stop_loss_pct"], 0.004)
        self.assertEqual(gold_settings["base_take_profit_pct"], 0.007)


class ExperimentExecutorTests(unittest.TestCase):
    def test_runtime_settings_do_not_mutate_source_experiment(self) -> None:
        experiment = {
            "experiment_id": "exp_001",
            "label": "test",
            "overrides": {"ACTIVE_STRATEGY_NAME": "barrier_reaction"},
            "combo_values": {},
        }

        runtime_experiment = experiment_executor.build_runtime_experiment(experiment)

        self.assertEqual(experiment["overrides"], {"ACTIVE_STRATEGY_NAME": "barrier_reaction"})
        self.assertEqual(runtime_experiment["overrides"], experiment["overrides"])

    def test_stage_range_calls_selected_runners_with_isolated_database(self) -> None:
        experiment = {
            "experiment_id": "exp_001",
            "label": "test",
            "overrides": {},
            "combo_values": {},
        }

        runtime_experiment = experiment_executor.build_runtime_experiment(experiment)
        worker_database_path = Path("worker.db")
        with patch.object(experiment_executor.subprocess, "run") as run_process:
            experiment_executor.run_stage_range(
                runtime_experiment=runtime_experiment,
                historical_data_cutoff="2026-01-01T00:00:00",
                database_path=worker_database_path,
                start_stage="features",
                end_stage="strategies",
            )

        executed_scripts = [
            Path(call.args[0][1]).name
            for call in run_process.call_args_list
        ]
        environment = run_process.call_args_list[0].kwargs["env"]
        self.assertEqual(
            executed_scripts,
            [
                "03_run_features.py",
                "04_run_signals.py",
            ],
        )
        self.assertEqual(
            environment["QUANT_HISTORICAL_DATA_CUTOFF"],
            "2026-01-01T00:00:00",
        )
        self.assertEqual(environment["QUANT_OPTIMISATION_RUN"], "1")
        self.assertEqual(environment["QUANT_DATABASE_PATH"], str(worker_database_path))

    def test_oos_stage_range_disables_optimisation_scoring_periods(self) -> None:
        experiment = {
            "experiment_id": "out_of_sample",
            "label": "selected",
            "overrides": {
                "BACKTEST_START_TIME": "2026-01-01T00:00:00",
                "BACKTEST_END_TIME": "2026-04-01T00:00:00",
            },
            "combo_values": {},
        }
        with (
            patch.dict(os.environ, {"QUANT_OPTIMISATION_RUN": "1"}),
            patch.object(experiment_executor.subprocess, "run") as run_process,
        ):
            experiment_executor.run_stage_range(
                runtime_experiment=experiment,
                historical_data_cutoff="2026-04-01T00:00:00",
                database_path=Path("out_of_sample.db"),
                start_stage="strategies",
                end_stage="backtesting",
                optimisation_scoring_enabled=False,
            )

        environment = run_process.call_args.kwargs["env"]
        self.assertNotIn("QUANT_OPTIMISATION_RUN", environment)
        overrides = json.loads(environment["QUANT_CONFIG_OVERRIDES_JSON"])
        self.assertNotIn("BACKTEST_START_TIME", overrides)
        self.assertNotIn("BACKTEST_END_TIME", overrides)
        self.assertEqual(
            environment["QUANT_BACKTEST_START_TIME"],
            "2026-01-01T00:00:00",
        )
        self.assertEqual(
            environment["QUANT_BACKTEST_END_TIME"],
            "2026-04-01T00:00:00",
        )

    def test_search_dependencies_control_template_boundary(self) -> None:
        search_space = {
            "feature_parameter": {
                "type": "categorical",
                "values": [1, 2],
                "affects": "features",
            },
            "strategy_parameter": {
                "type": "categorical",
                "values": [1, 2],
                "affects": "strategies",
            },
            "positioning_parameter": {
                "type": "categorical",
                "values": [1, 2],
                "affects": "positioning",
            },
        }
        with patch.object(experiment_executor, "SEARCH_SPACE", search_space):
            start_stage = experiment_executor.get_search_start_stage()
            preparation_end_stage = experiment_executor.get_preparation_end_stage()

        self.assertEqual(start_stage, "features")
        self.assertEqual(preparation_end_stage, "processing")

    def test_template_is_a_prepared_copy_of_the_central_database(self) -> None:
        search_space = {
            "FEATURE_SETTINGS.period": {
                "type": "int",
                "low": 10,
                "high": 20,
                "affects": "features",
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            central_database_path = temporary_path / "central.db"
            work_directory = temporary_path / "work"
            connection = sqlite3.connect(central_database_path)
            connection.execute("CREATE TABLE marker (value TEXT)")
            connection.execute("INSERT INTO marker VALUES ('copied')")
            connection.commit()
            connection.close()

            with (
                patch.object(
                    experiment_executor,
                    "DATABASE_PATH",
                    central_database_path,
                ),
                patch.object(
                    experiment_executor,
                    "OPTIMISATION_WORK_DIR",
                    work_directory,
                ),
                patch.object(
                    experiment_executor,
                    "SEARCH_SPACE",
                    search_space,
                ),
                patch.object(
                    experiment_executor,
                    "run_stage_range",
                ) as run_stage_range,
            ):
                template_path = experiment_executor.prepare_template_database(
                    batch_id="batch_test",
                    historical_data_cutoff="2026-01-01T00:00:00",
                )

            template_connection = sqlite3.connect(template_path)
            copied_value = template_connection.execute(
                "SELECT value FROM marker"
            ).fetchone()[0]
            template_connection.close()
            self.assertEqual(copied_value, "copied")
            self.assertEqual(
                run_stage_range.call_args.kwargs["end_stage"],
                "processing",
            )

    def test_template_can_be_cloned_by_parallel_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            template_path = temporary_path / "template.db"
            connection = sqlite3.connect(template_path)
            connection.execute("CREATE TABLE marker (value TEXT)")
            connection.execute("INSERT INTO marker VALUES ('parallel')")
            connection.commit()
            connection.close()
            worker_paths = [
                temporary_path / f"worker_{index}.db"
                for index in range(3)
            ]

            with ProcessPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(
                        experiment_executor.clone_database,
                        template_path,
                        worker_path,
                    )
                    for worker_path in worker_paths
                ]
                for future in futures:
                    future.result()

            for worker_path in worker_paths:
                worker_connection = sqlite3.connect(worker_path)
                value = worker_connection.execute(
                    "SELECT value FROM marker"
                ).fetchone()[0]
                worker_connection.close()
                self.assertEqual(value, "parallel")

    def test_existing_worker_database_is_reused_without_another_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            template_path = temporary_path / "template.db"
            worker_path = temporary_path / "worker.db"
            template_path.touch()
            worker_path.touch()
            experiment = {
                "experiment_id": "trial_001",
                "label": "test",
                "overrides": {},
                "combo_values": {},
            }

            with (
                patch.object(experiment_executor, "clone_database") as clone,
                patch.object(experiment_executor, "run_stage_range") as run,
                patch.object(
                    experiment_executor,
                    "get_search_start_stage",
                    return_value="positioning",
                ),
            ):
                experiment_executor.execute_isolated_experiment(
                    experiment=experiment,
                    historical_data_cutoff="2026-01-01T00:00:00",
                    template_database_path=template_path,
                    worker_database_path=worker_path,
                )

            clone.assert_not_called()
            run.assert_called_once()

    def test_stage_range_executes_only_the_required_stage_suffix(self) -> None:
        with patch.object(
            experiment_executor.subprocess,
            "run",
        ) as run_process:
            experiment_executor.run_stage_range(
                runtime_experiment={
                    "experiment_id": "trial_001",
                    "overrides": {},
                },
                historical_data_cutoff="2026-01-01T00:00:00",
                database_path=Path("worker_01.db"),
                start_stage="strategies",
                end_stage="positioning",
            )

        executed_scripts = [
            Path(call.args[0][1]).name
            for call in run_process.call_args_list
        ]
        self.assertEqual(
            executed_scripts,
            [
                "04_run_signals.py",
                "05_run_positioning.py",
            ],
        )

    def test_stage_range_reports_worker_trial_and_stage_progress(self) -> None:
        with (
            patch.object(experiment_executor.subprocess, "run"),
            patch("builtins.print") as progress_print,
        ):
            experiment_executor.run_stage_range(
                runtime_experiment={
                    "experiment_id": "trial_007",
                    "overrides": {},
                },
                historical_data_cutoff="2026-01-01T00:00:00",
                database_path=Path("worker_02.db"),
                start_stage="positioning",
                end_stage="positioning",
            )

        messages = [call.args[0] for call in progress_print.call_args_list]
        self.assertEqual(
            messages[0],
            "[Worker 2 | Trial 007] positioning started",
        )
        self.assertIn(
            "[Worker 2 | Trial 007] positioning completed in ",
            messages[1],
        )


class OptunaStudyTests(unittest.TestCase):
    def test_best_training_parameters_build_the_frozen_oos_experiment(self):
        parameter_name = (
            "STRATEGY_CONFIRMATION_SETTINGS.rsi_momentum.minimum_buy_rsi"
        )
        trial = SimpleNamespace(number=6, params={parameter_name: 42})
        experiment = optimisation_study.build_out_of_sample_experiment(
            trial,
            1,
            {
                "out_of_sample_start": "2026-01-01T00:00:00",
                "out_of_sample_end": "2026-04-01T00:00:00",
            },
        )

        self.assertEqual(experiment["training_rank"], 1)
        self.assertEqual(experiment["selected_trial_number"], 7)
        self.assertEqual(experiment["combo_values"], {parameter_name: 42})
        self.assertEqual(
            experiment["overrides"]["STRATEGY_CONFIRMATION_SETTINGS"]
            ["rsi_momentum"]["minimum_buy_rsi"],
            42,
        )
        self.assertEqual(
            experiment["overrides"]["BACKTEST_START_TIME"],
            "2026-01-01T00:00:00",
        )
        self.assertEqual(
            experiment["overrides"]["BACKTEST_END_TIME"],
            "2026-04-01T00:00:00",
        )

    def test_oos_selection_uses_top_unique_training_combinations(self):
        trials = [
            SimpleNamespace(
                number=0,
                state=optuna.trial.TrialState.COMPLETE,
                value=1.5,
                params={"x": 1},
            ),
            SimpleNamespace(
                number=1,
                state=optuna.trial.TrialState.COMPLETE,
                value=1.4,
                params={"x": 1},
            ),
            SimpleNamespace(
                number=2,
                state=optuna.trial.TrialState.COMPLETE,
                value=1.3,
                params={"x": 2},
            ),
            SimpleNamespace(
                number=3,
                state=optuna.trial.TrialState.PRUNED,
                value=None,
                params={"x": 3},
            ),
        ]
        fake_study = SimpleNamespace(
            get_trials=lambda deepcopy=False: trials,
        )

        selected = optimisation_study.select_top_unique_trials(fake_study, 2)

        self.assertEqual([trial.number for trial in selected], [0, 2])

    def test_unscorable_trial_is_skipped_and_following_trial_completes(self):
        study = optuna.create_study()
        with (
            patch.object(optimisation_study, "OPTUNA_PARALLEL_WORKERS", 1),
            patch.object(optimisation_study, "ProcessPoolExecutor", ThreadPoolExecutor),
            patch.object(optimisation_study, "build_worker_database_path", return_value=Path("unused.db")),
            patch.object(optimisation_study, "build_experiment", side_effect=lambda trial: {
                "experiment_id": f"trial_{trial.number}", "combo_values": {"x": trial.number}, "overrides": {},
            }),
            patch.object(optimisation_study, "find_archived_score_for_parameters", return_value=None),
            patch.object(optimisation_study, "execute_isolated_experiment", side_effect=lambda **kw: kw["experiment"]),
            patch.object(optimisation_study, "archive_latest_backtest_summary",
                         side_effect=[repository.UnscorableTrial("Undefined Sharpe"), 1.2]),
        ):
            optimisation_study.run_parallel_trials(
                study=study, batch_id="test", historical_data_cutoff="2026-01-01T00:00:00",
                template_database_path=Path("unused.db"), trial_count=2,
            )
        self.assertEqual([trial.state for trial in study.trials],
                         [optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.COMPLETE])
        self.assertEqual(study.trials[0].user_attrs["unscorable_reason"], "Undefined Sharpe")
        self.assertEqual(study.best_value, 1.2)

    def test_sampler_models_parameter_relationships_and_parallel_trials(self) -> None:
        with patch.object(optuna.samplers, "TPESampler") as sampler_class:
            optimisation_study.build_sampler()

        sampler_class.assert_called_once_with(
            n_startup_trials=(
                optimisation_study.calculate_random_startup_trial_count(
                    optimisation_study.OPTUNA_TRIAL_COUNT,
                    optimisation_study.OPTUNA_RANDOM_STARTUP_TRIAL_FRACTION,
                )
            ),
            seed=optimisation_study.OPTUNA_SEED,
            multivariate=optimisation_study.OPTUNA_TPE_MULTIVARIATE,
            constant_liar=optimisation_study.OPTUNA_TPE_CONSTANT_LIAR,
        )

    def test_calculates_random_startup_trials_from_fraction(self) -> None:
        self.assertEqual(
            optimisation_study.calculate_random_startup_trial_count(300, 0.20),
            60,
        )

    def test_formats_total_optimisation_duration(self) -> None:
        self.assertEqual(optimisation_study.format_elapsed_time(45.4), "45s")
        self.assertEqual(optimisation_study.format_elapsed_time(754.6), "12m 35s")
        self.assertEqual(optimisation_study.format_elapsed_time(7384.2), "2h 3m 4s")

    def test_estimates_remaining_optimisation_time(self) -> None:
        self.assertEqual(
            optimisation_study.estimate_remaining_time(25, 100, 50.0),
            150.0,
        )
        self.assertEqual(
            optimisation_study.estimate_remaining_time(100, 100, 50.0),
            0.0,
        )
        self.assertIsNone(
            optimisation_study.estimate_remaining_time(0, 100, 0.0)
        )

    def test_each_optimisation_creates_a_fresh_study(self) -> None:
        storage = optuna.storages.InMemoryStorage()
        with patch.object(
            optimisation_study,
            "build_batch_id",
            side_effect=["batch_first", "batch_second"],
        ):
            first_study = optimisation_study.create_study(
                storage,
                "2026-01-01T00:00:00",
            )
            second_study = optimisation_study.create_study(
                storage,
                "2026-02-01T00:00:00",
            )

        self.assertNotEqual(first_study.study_name, second_study.study_name)
        self.assertEqual(first_study.user_attrs["batch_id"], "batch_first")
        self.assertEqual(second_study.user_attrs["batch_id"], "batch_second")

    def test_pending_duplicate_uses_one_parallel_worker(self) -> None:
        first_experiment = {
            "experiment_id": "trial_001",
            "label": "duplicate",
            "overrides": {},
            "combo_values": {"stop": 0.003, "target": 0.007},
        }
        second_experiment = {
            **first_experiment,
            "experiment_id": "trial_002",
        }
        parallel_study = optuna.create_study(direction="maximize")

        with (
            patch.object(
                optimisation_study,
                "build_experiment",
                side_effect=[first_experiment, second_experiment],
            ),
            patch.object(
                optimisation_study,
                "find_archived_score_for_parameters",
                return_value=None,
            ),
            patch.object(
                optimisation_study,
                "execute_isolated_experiment",
                return_value=first_experiment,
            ) as execute,
            patch.object(
                optimisation_study,
                "archive_latest_backtest_summary",
                return_value=1.25,
            ) as archive,
            patch.object(
                optimisation_study,
                "ProcessPoolExecutor",
                ThreadPoolExecutor,
            ),
            patch.object(optimisation_study, "OPTUNA_PARALLEL_WORKERS", 2),
        ):
            optimisation_study.run_parallel_trials(
                study=parallel_study,
                batch_id="batch_test",
                historical_data_cutoff="2026-01-01T00:00:00",
                template_database_path=Path("template.db"),
                trial_count=2,
            )

        self.assertEqual(optimisation_study.completed_trial_count(parallel_study), 2)
        execute.assert_called_once()
        archive.assert_called_once()


class OptimisationArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "optimisation.db"
        self.original_database_path = repository.DATABASE_PATH
        repository.DATABASE_PATH = self.database_path
        self._create_backtest_source_tables()

    def tearDown(self) -> None:
        repository.DATABASE_PATH = self.original_database_path
        self.temporary_directory.cleanup()

    def _create_backtest_source_tables(self) -> None:
        connection = sqlite3.connect(self.database_path)
        connection.executescript(
            """
            CREATE TABLE bt_runs (
                run_at TEXT,
                strategy_name TEXT,
                positioning_name TEXT,
                resolution TEXT,
                stop_loss_pct REAL,
                take_profit_pct REAL,
                trade_count INTEGER,
                break_even_count INTEGER,
                win_count INTEGER,
                loss_count INTEGER,
                total_pnl_amount REAL,
                average_pnl_amount REAL,
                total_return_pct REAL,
                win_rate REAL,
                daily_sharpe_ratio REAL,
                average_period_sharpe_ratio REAL,
                capital_base_amount REAL,
                total_spread_cost_amount REAL
            );

            CREATE TABLE bt_runs_by_epic (
                run_at TEXT,
                strategy_name TEXT,
                positioning_name TEXT,
                epic TEXT,
                resolution TEXT,
                stop_loss_pct REAL,
                take_profit_pct REAL,
                trade_count INTEGER,
                break_even_count INTEGER,
                win_count INTEGER,
                loss_count INTEGER,
                total_pnl_amount REAL,
                average_pnl_amount REAL,
                total_return_pct REAL,
                win_rate REAL,
                daily_sharpe_ratio REAL,
                capital_base_amount REAL,
                total_spread_cost_amount REAL
            );

            CREATE TABLE bt_runs_by_period (
                run_at TEXT,
                strategy_name TEXT,
                positioning_name TEXT,
                resolution TEXT,
                period_index INTEGER,
                period_start TEXT,
                period_end TEXT,
                completed_day_count INTEGER,
                trade_count INTEGER,
                daily_sharpe_ratio REAL
            );

            CREATE TABLE proc_feature_candles (
                epic TEXT,
                resolution TEXT,
                snapshot_time TEXT
            );
            """
        )
        summary_values = (
            "2026-01-01T00:00:00",
            "barrier_reaction",
            "fixed_rr",
            "MINUTE_5",
            0.003,
            0.007,
            20,
            2,
            10,
            8,
            500.0,
            25.0,
            0.05,
            0.5,
            1.2,
            1.1,
            10000.0,
            30.0,
        )
        connection.execute(
            "INSERT INTO bt_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            summary_values,
        )
        connection.execute(
            "INSERT INTO bt_runs_by_epic VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                *summary_values[:3],
                "GOLD",
                *summary_values[3:15],
                *summary_values[16:],
            ),
        )
        for period_index in range(1, repository.OPTIMISATION_PERIOD_COUNT + 1):
            connection.execute(
                "INSERT INTO bt_runs_by_period VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "2026-01-01T00:00:00",
                    "barrier_reaction",
                    "fixed_rr",
                    "MINUTE_5",
                    period_index,
                    f"2025-{period_index:02d}-01T00:00:00",
                    f"2025-{period_index:02d}-28T00:00:00",
                    90,
                    2,
                    1.0 + period_index / 10,
                ),
            )
        connection.execute(
            "INSERT INTO proc_feature_candles VALUES (?, ?, ?)",
            ("GOLD", "MINUTE_5", "2026-01-01T00:00:00"),
        )
        connection.commit()
        connection.close()

    def test_archive_uses_normalized_parameter_rows_and_analyser_reads_them(self) -> None:
        batch_id = "batch_test"
        worker_database_path = (
            Path(self.temporary_directory.name) / "worker.db"
        )
        central_connection = sqlite3.connect(self.database_path)
        worker_connection = sqlite3.connect(worker_database_path)
        central_connection.backup(worker_connection)
        worker_connection.close()
        central_connection.executescript(
            """
            DROP TABLE bt_runs;
            DROP TABLE bt_runs_by_epic;
            DROP TABLE bt_runs_by_period;
            DROP TABLE proc_feature_candles;
            """
        )
        central_connection.commit()
        central_connection.close()

        parameter_name = "POSITIONING_METHOD_SETTINGS.fixed_rr.epic_overrides.GOLD.base_stop_loss_pct"
        experiment = {
            "experiment_id": "exp_001",
            "label": "stop=0.003",
            "overrides": {
                "ACTIVE_STRATEGY_NAME": "barrier_reaction",
                "CAPITAL_EPICS": ["GOLD"],
                "HISTORICAL_PRICES_RESOLUTION": "MINUTE_5",
            },
            "combo_values": {parameter_name: 0.003},
        }

        with patch.object(
            repository,
            "SEARCH_SPACE",
            {
                parameter_name: {
                    "type": "categorical",
                    "values": [0.003, 0.004, 0.005],
                    "affects": "positioning",
                }
            },
        ):
            repository.initialize_experiment_batch(
                batch_id=batch_id,
                historical_data_cutoff="2026-01-01T00:00:00",
                requested_trial_count=1,
                random_startup_trial_count=1,
            )
            score = repository.archive_latest_backtest_summary(
                experiment,
                batch_id,
                source_database_path=worker_database_path,
            )
            duplicate_score = repository.find_archived_score_for_parameters(
                batch_id,
                experiment["combo_values"],
            )
            repository.finalize_experiment_batch(
                batch_id,
                optimisation_duration_seconds=123.5,
            )

        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        parameter_row = connection.execute(
            "SELECT parameter_name, parameter_value_json FROM bt_experiment_parameters"
        ).fetchone()
        run_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(bt_experiment_runs)")
        }
        analysis = analyser.build_latest_batch_analysis(connection)
        batch_counts = connection.execute(
            """
            SELECT requested_trial_count, unique_run_count
            FROM bt_experiment_batches
            WHERE batch_id = ?
            """,
            (batch_id,),
        ).fetchone()
        batch_timing = connection.execute(
            """
            SELECT
                completed_at,
                parallel_worker_count,
                optimisation_duration_seconds
            FROM bt_experiment_batches
            WHERE batch_id = ?
            """,
            (batch_id,),
        ).fetchone()
        connection.close()

        self.assertEqual(score, 1.1)
        self.assertEqual(duplicate_score, 1.1)
        self.assertEqual(batch_counts["requested_trial_count"], 1)
        self.assertEqual(batch_counts["unique_run_count"], 1)
        self.assertIsNotNone(batch_timing["completed_at"])
        self.assertEqual(
            batch_timing["parallel_worker_count"],
            repository.OPTUNA_PARALLEL_WORKERS,
        )
        self.assertEqual(batch_timing["optimisation_duration_seconds"], 123.5)
        self.assertEqual(parameter_row["parameter_name"], parameter_name)
        self.assertEqual(json.loads(parameter_row["parameter_value_json"]), 0.003)
        self.assertNotIn(parameter_name, run_columns)
        self.assertEqual(analysis["results"][0]["parameters"][parameter_name], 0.003)
        self.assertEqual(
            len(analysis["period_results"]),
            repository.OPTIMISATION_PERIOD_COUNT,
        )

    def test_oos_summary_is_archived_separately_from_training_trials(self) -> None:
        batch_id = "batch_oos"
        data_split = {
            "training_start": "2024-01-01T00:00:00",
            "training_end": "2025-10-01T00:00:00",
            "out_of_sample_start": "2025-10-01T00:00:00",
            "out_of_sample_end": "2026-01-01T00:00:00",
        }
        repository.initialize_experiment_batch(
            batch_id=batch_id,
            historical_data_cutoff="2026-01-01T00:00:00",
            requested_trial_count=1,
            random_startup_trial_count=1,
            data_split=data_split,
        )

        worker_database_path = Path(self.temporary_directory.name) / "oos.db"
        central_connection = sqlite3.connect(self.database_path)
        worker_connection = sqlite3.connect(worker_database_path)
        central_connection.backup(worker_connection)
        worker_connection.close()
        central_connection.close()

        parameter_name = (
            "STRATEGY_CONFIRMATION_SETTINGS.rsi_momentum.minimum_buy_rsi"
        )
        experiment = {
            "experiment_id": "out_of_sample",
            "training_rank": 1,
            "selected_trial_number": 7,
            "label": "minimum_buy_rsi=42",
            "overrides": {
                "ACTIVE_STRATEGY_NAME": "barrier_reaction",
                "HISTORICAL_PRICES_RESOLUTION": "MINUTE_5",
                "BACKTEST_START_TIME": data_split["out_of_sample_start"],
                "BACKTEST_END_TIME": data_split["out_of_sample_end"],
            },
            "combo_values": {parameter_name: 42},
        }
        repository.archive_out_of_sample_summary(
            experiment=experiment,
            batch_id=batch_id,
            source_database_path=worker_database_path,
        )

        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        oos_row = connection.execute(
            "SELECT * FROM bt_experiment_oos_runs WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        training_count = connection.execute(
            "SELECT COUNT(*) FROM bt_experiment_runs WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()[0]
        connection.close()

        self.assertEqual(training_count, 0)
        self.assertEqual(oos_row["training_rank"], 1)
        self.assertEqual(oos_row["selected_trial_number"], 7)
        self.assertEqual(oos_row["period_start"], data_split["out_of_sample_start"])
        self.assertEqual(oos_row["period_end"], data_split["out_of_sample_end"])
        self.assertEqual(json.loads(oos_row["parameters_json"]), {parameter_name: 42})
        self.assertEqual(oos_row["daily_sharpe_ratio"], 1.2)


if __name__ == "__main__":
    unittest.main()
