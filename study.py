"""Create and coordinate one fresh parallel Optuna study."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path
import sqlite3
import subprocess
from time import perf_counter

import optuna
from optuna.trial import TrialState

from quant.config import (
    CAPITAL_EPICS,
    FINAL_HOLDOUT_FRACTION,
    HISTORICAL_PRICES_RESOLUTION,
    OPTIMISATION_ETA_ENABLED,
    OPTIMISATION_PERIOD_COUNT,
    OUT_OF_SAMPLE_TOP_TRIAL_COUNT,
    OPTUNA_PARALLEL_WORKERS,
    OPTUNA_OBJECTIVE_DIRECTION,
    OPTUNA_RANDOM_STARTUP_TRIAL_FRACTION,
    OPTUNA_SEED,
    OPTUNA_STUDY_NAME,
    OPTUNA_TPE_CONSTANT_LIAR,
    OPTUNA_TPE_MULTIVARIATE,
    OPTUNA_TRIAL_COUNT,
)
from quant.optimisation.periods import freeze_periods, read_data_split
from quant.optimisation.experiment_builder import (
    build_experiment,
    build_overrides,
    format_parameter_value,
)
from quant.optimisation.experiment_executor import (
    build_out_of_sample_database_path,
    build_worker_database_path,
    cleanup_study_workspace,
    execute_isolated_experiment,
    execute_out_of_sample_experiment,
    prepare_template_database,
)
from quant.optimisation.preflight import build_preflight_report, build_preflight_snapshot
from quant.optimisation.repository import (
    UnscorableTrial,
    archive_latest_backtest_summary,
    archive_out_of_sample_summary,
    build_batch_id,
    finalize_experiment_batch,
    find_archived_score_for_parameters,
    get_common_historical_data_cutoff,
    initialize_experiment_batch,
)
from quant.paths import OPTUNA_DATABASE_PATH, PROJECT_ROOT


def build_out_of_sample_experiment(
    trial: optuna.trial.FrozenTrial,
    training_rank: int,
    data_split: dict,
) -> dict:
    """Apply one preselected training combination to the OOS window."""
    parameter_values = dict(trial.params)
    overrides = build_overrides(parameter_values)
    overrides["BACKTEST_START_TIME"] = data_split["out_of_sample_start"]
    overrides["BACKTEST_END_TIME"] = data_split["out_of_sample_end"]
    label = " | ".join(
        f"{name.rsplit('.', 1)[-1]}={format_parameter_value(value)}"
        for name, value in parameter_values.items()
    )
    return {
        "experiment_id": f"out_of_sample_{training_rank:02d}",
        "training_rank": training_rank,
        "selected_trial_number": trial.number + 1,
        "label": label,
        "overrides": overrides,
        "combo_values": parameter_values,
    }


def select_top_unique_trials(study: optuna.Study, count: int) -> list:
    """Rank completed trials by training objective and remove duplicates."""
    if type(count) is not int or count < 1:
        raise ValueError("OUT_OF_SAMPLE_TOP_TRIAL_COUNT must be a positive integer.")
    completed_trials = [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state == TrialState.COMPLETE and trial.value is not None
    ]
    completed_trials.sort(
        key=lambda trial: trial.value,
        reverse=OPTUNA_OBJECTIVE_DIRECTION == "maximize",
    )
    selected_trials = []
    seen_signatures = set()
    for trial in completed_trials:
        signature = build_parameter_signature(trial.params)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        selected_trials.append(trial)
        if len(selected_trials) == count:
            break
    return selected_trials


def build_parameter_signature(parameter_values: dict) -> str:
    return json.dumps(
        parameter_values,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_storage_url() -> str:
    OPTUNA_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{OPTUNA_DATABASE_PATH.resolve().as_posix()}"


def calculate_random_startup_trial_count(
    trial_count: int,
    startup_fraction: float,
) -> int:
    if type(trial_count) is not int or trial_count < 1:
        raise ValueError("OPTUNA_TRIAL_COUNT must be a positive integer.")
    if (
        isinstance(startup_fraction, bool)
        or not isinstance(startup_fraction, (int, float))
        or not 0.0 <= startup_fraction <= 1.0
    ):
        raise ValueError(
            "OPTUNA_RANDOM_STARTUP_TRIAL_FRACTION must be between 0 and 1."
        )
    return round(trial_count * startup_fraction)


def build_sampler() -> optuna.samplers.TPESampler:
    return optuna.samplers.TPESampler(
        n_startup_trials=calculate_random_startup_trial_count(
            OPTUNA_TRIAL_COUNT,
            OPTUNA_RANDOM_STARTUP_TRIAL_FRACTION,
        ),
        seed=OPTUNA_SEED,
        multivariate=OPTUNA_TPE_MULTIVARIATE,
        constant_liar=OPTUNA_TPE_CONSTANT_LIAR,
    )


def completed_trial_count(study: optuna.Study) -> int:
    return sum(
        trial.state == TrialState.COMPLETE
        for trial in study.get_trials(deepcopy=False)
    )


def estimate_remaining_time(
    finished_trial_count: int,
    trial_count: int,
    elapsed_seconds: float,
) -> float | None:
    """Estimate remaining wall time from observed parallel throughput."""
    if finished_trial_count <= 0:
        return None
    remaining_trial_count = max(trial_count - finished_trial_count, 0)
    return elapsed_seconds / finished_trial_count * remaining_trial_count


def create_study(
    storage_url: str,
    historical_data_cutoff: str,
) -> optuna.Study:
    batch_id = build_batch_id()
    study_name = f"{OPTUNA_STUDY_NAME}__{batch_id}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=build_sampler(),
        pruner=optuna.pruners.NopPruner(),
        direction=OPTUNA_OBJECTIVE_DIRECTION,
    )
    study.set_user_attr("batch_id", batch_id)
    study.set_user_attr("historical_data_cutoff", historical_data_cutoff)
    return study


def run_parallel_trials(
    *,
    study: optuna.Study,
    batch_id: str,
    historical_data_cutoff: str,
    template_database_path: Path,
    trial_count: int,
) -> None:
    """Evaluate trials in worker databases while coordinating Optuna centrally."""
    if OPTUNA_PARALLEL_WORKERS <= 0:
        raise ValueError("OPTUNA_PARALLEL_WORKERS must be greater than zero.")
    if type(OPTIMISATION_ETA_ENABLED) is not bool:
        raise ValueError("OPTIMISATION_ETA_ENABLED must be a boolean.")
    requested_trial_count = 0
    finished_trial_count = 0
    last_reported_count = 0
    trials_started_at = perf_counter()
    pending_futures = {}
    future_by_signature = {}
    duplicate_trials_by_signature: dict[str, list[optuna.Trial]] = {}

    worker_count = min(OPTUNA_PARALLEL_WORKERS, trial_count)
    available_worker_database_paths = [
        build_worker_database_path(batch_id, worker_index)
        for worker_index in range(1, worker_count + 1)
    ]
    executor = ProcessPoolExecutor(max_workers=worker_count)

    def report_progress() -> None:
        nonlocal last_reported_count
        if not OPTIMISATION_ETA_ENABLED or finished_trial_count == last_reported_count:
            return
        elapsed_seconds = perf_counter() - trials_started_at
        remaining_seconds = estimate_remaining_time(
            finished_trial_count,
            trial_count,
            elapsed_seconds,
        )
        progress_percent = 100.0 * finished_trial_count / trial_count
        remaining_text = (
            format_elapsed_time(remaining_seconds)
            if remaining_seconds is not None
            else "calculating"
        )
        print(
            f"[Optimisation] {finished_trial_count}/{trial_count} trials finished "
            f"({progress_percent:.1f}%) | elapsed "
            f"{format_elapsed_time(elapsed_seconds)} | estimated remaining "
            f"{remaining_text}",
            flush=True,
        )
        last_reported_count = finished_trial_count

    try:
        while requested_trial_count < trial_count or pending_futures:
            while (
                requested_trial_count < trial_count
                and len(pending_futures) < worker_count
            ):
                trial = study.ask()
                requested_trial_count += 1
                experiment = build_experiment(trial)
                parameter_signature = build_parameter_signature(
                    experiment["combo_values"]
                )

                cached_score = find_archived_score_for_parameters(
                    batch_id,
                    experiment["combo_values"],
                )
                if cached_score is not None:
                    trial.set_user_attr("duplicate_reused", True)
                    study.tell(trial, cached_score)
                    finished_trial_count += 1
                    continue

                if parameter_signature in future_by_signature:
                    trial.set_user_attr("duplicate_reused", True)
                    duplicate_trials_by_signature.setdefault(
                        parameter_signature,
                        [],
                    ).append(trial)
                    continue

                worker_database_path = available_worker_database_paths.pop()
                future = executor.submit(
                    execute_isolated_experiment,
                    experiment=experiment,
                    historical_data_cutoff=historical_data_cutoff,
                    template_database_path=template_database_path,
                    worker_database_path=worker_database_path,
                )
                pending_futures[future] = {
                    "trial": trial,
                    "experiment": experiment,
                    "parameter_signature": parameter_signature,
                    "worker_database_path": worker_database_path,
                }
                future_by_signature[parameter_signature] = future

            if not pending_futures:
                report_progress()
                continue

            completed_futures, _ = wait(
                pending_futures,
                return_when=FIRST_COMPLETED,
            )
            for future in completed_futures:
                pending_trial = pending_futures.pop(future)
                trial = pending_trial["trial"]
                parameter_signature = pending_trial["parameter_signature"]
                worker_database_path = pending_trial["worker_database_path"]
                waiting_duplicates = duplicate_trials_by_signature.pop(
                    parameter_signature,
                    [],
                )
                future_by_signature.pop(parameter_signature, None)

                try:
                    runtime_experiment = future.result()
                    score = archive_latest_backtest_summary(
                        experiment=runtime_experiment,
                        batch_id=batch_id,
                        source_database_path=worker_database_path,
                    )
                except UnscorableTrial as exc:
                    for skipped_trial in [trial, *waiting_duplicates]:
                        skipped_trial.set_user_attr("unscorable_reason", str(exc))
                        study.tell(skipped_trial, state=TrialState.PRUNED)
                    print(f"Trial {trial.number + 1:03d} skipped: {exc}", flush=True)
                except BaseException:
                    study.tell(trial, state=TrialState.FAIL)
                    for duplicate_trial in waiting_duplicates:
                        study.tell(duplicate_trial, state=TrialState.FAIL)
                    raise
                else:
                    trial.set_user_attr("duplicate_reused", False)
                    study.tell(trial, score)
                    for duplicate_trial in waiting_duplicates:
                        study.tell(duplicate_trial, score)
                finally:
                    available_worker_database_paths.append(worker_database_path)
                finished_trial_count += 1 + len(waiting_duplicates)
            report_progress()
    except BaseException:
        for pending_trial in pending_futures.values():
            study.tell(
                pending_trial["trial"],
                state=TrialState.FAIL,
                skip_if_finished=True,
            )
        for duplicate_trials in duplicate_trials_by_signature.values():
            for duplicate_trial in duplicate_trials:
                study.tell(
                    duplicate_trial,
                    state=TrialState.FAIL,
                    skip_if_finished=True,
                )
        if hasattr(executor, "terminate_workers"):
            executor.terminate_workers()
        else:
            executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def run_optimisation() -> None:
    optimisation_started_at = perf_counter()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    latest_historical_data_cutoff = get_common_historical_data_cutoff()
    study = create_study(
        build_storage_url(),
        latest_historical_data_cutoff,
    )
    batch_id = study.user_attrs["batch_id"]
    historical_data_cutoff = study.user_attrs["historical_data_cutoff"]

    try:
        template_database_path = prepare_template_database(
            batch_id=batch_id,
            historical_data_cutoff=historical_data_cutoff,
        )
        periods = freeze_periods(
            template_database_path, CAPITAL_EPICS, HISTORICAL_PRICES_RESOLUTION,
            OPTIMISATION_PERIOD_COUNT, FINAL_HOLDOUT_FRACTION,
        )
        split_connection = sqlite3.connect(template_database_path)
        try:
            data_split = read_data_split(split_connection.cursor())
        finally:
            split_connection.close()
        print(f"Scoring {len(periods)} periods from {periods[-1][1]} to {periods[0][2]} (exclusive)", flush=True)
        print(
            f"Reserved OOS window from {data_split['out_of_sample_start']} "
            f"to {data_split['out_of_sample_end']} (exclusive)",
            flush=True,
        )
        initialize_experiment_batch(
            batch_id=batch_id, historical_data_cutoff=historical_data_cutoff,
            requested_trial_count=OPTUNA_TRIAL_COUNT, periods=periods,
            data_split=data_split,
            random_startup_trial_count=calculate_random_startup_trial_count(
                OPTUNA_TRIAL_COUNT,
                OPTUNA_RANDOM_STARTUP_TRIAL_FRACTION,
            ),
            configuration=build_preflight_snapshot(),
            preflight_text=build_preflight_report(),
            git_commit=read_git_commit(),
        )
        run_parallel_trials(
            study=study,
            batch_id=batch_id,
            historical_data_cutoff=historical_data_cutoff,
            template_database_path=template_database_path,
            trial_count=OPTUNA_TRIAL_COUNT,
        )
        if completed_trial_count(study) > 0:
            selected_trials = select_top_unique_trials(
                study,
                OUT_OF_SAMPLE_TOP_TRIAL_COUNT,
            )
            print(
                f"Evaluating {len(selected_trials)} top training combinations on OOS",
                flush=True,
            )
            for training_rank, trial in enumerate(selected_trials, start=1):
                oos_experiment = build_out_of_sample_experiment(
                    trial,
                    training_rank,
                    data_split,
                )
                oos_database_path = build_out_of_sample_database_path(
                    batch_id,
                    training_rank,
                )
                runtime_oos_experiment = execute_out_of_sample_experiment(
                    experiment=oos_experiment,
                    historical_data_cutoff=historical_data_cutoff,
                    template_database_path=template_database_path,
                    database_path=oos_database_path,
                )
                archive_out_of_sample_summary(
                    experiment=runtime_oos_experiment,
                    batch_id=batch_id,
                    source_database_path=oos_database_path,
                )
    finally:
        try:
            cleanup_study_workspace(batch_id)
        except OSError as exc:
            # Temporary-file cleanup must not invalidate completed results.
            print(f"Warning: optimisation workspace cleanup failed: {exc}", flush=True)

    if completed_trial_count(study) > 0:
        finalize_experiment_batch(
            batch_id,
            optimisation_duration_seconds=(
                perf_counter() - optimisation_started_at
            ),
        )

    else:
        print("No trials produced a valid score. Review activity in each scoring period.", flush=True)


def format_elapsed_time(elapsed_seconds: float) -> str:
    total_seconds = round(elapsed_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def read_git_commit() -> str | None:
    """Return the checked-out commit without preventing uncommitted runs."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    optimisation_started_at = perf_counter()
    try:
        run_optimisation()
    except BaseException:
        print(
            "Optimisation failed after "
            f"{format_elapsed_time(perf_counter() - optimisation_started_at)}",
            flush=True,
        )
        raise
    print(
        "Optimisation completed in "
        f"{format_elapsed_time(perf_counter() - optimisation_started_at)}",
        flush=True,
    )
