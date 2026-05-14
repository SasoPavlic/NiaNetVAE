import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from log import Log
from nianetvae.experiments.rnn_vae_experiment import RNNVAExperiment
from nianetvae.models.rnn_vae import RNNVAE
from nianetvae.search.cycle_warmstart import (
    _find_latest_trained_cycle_artifacts_before,
    _resolve_finetune_policy,
    _resolve_warm_start_sampling,
    export_skipped_non_trainable_cycle as _export_skipped_non_trainable_cycle,
)
from nianetvae.search.objective_engine import (
    DEFAULT_PENALTY,
    calculate_objective_bundle_from_cached_row,
    calculate_objective_bundle_from_experiment,
    _resolve_objective_contract,
)
from nianetvae.search.runtime_artifacts import (
    _as_jsonable,
    _cleanup_candidate_runtime,
    _export_cycle_artifacts,
    _resolve_export_dir,
    _run_final_training,
    _run_training_with_model,
    _short_exception_reason,
)
from nianetvae.search.winner_selection import (
    _resolve_winner_selection_contract,
    _select_deterministic_pareto_winner,
)


class SearchStorageConnectorProtocol(Protocol):
    def get_entries(self, hash_id: str, dataset_name: str): ...

    def save_model_and_entry(self, **kwargs): ...

    def get_cycle_candidates(self, dataset_name: str, algorithm_name: str = "NSGA3"): ...


@dataclass(frozen=True)
class SearchRuntimeContext:
    run_uuid: str
    config: dict[str, Any]
    conn: SearchStorageConnectorProtocol
    datamodule: Any
    dataset_name: str
    penalty: int = DEFAULT_PENALTY

    def __post_init__(self):
        if not isinstance(self.run_uuid, str) or not self.run_uuid.strip():
            raise ValueError("SearchRuntimeContext.run_uuid must be a non-empty string.")
        if not isinstance(self.dataset_name, str) or not self.dataset_name.strip():
            raise ValueError("SearchRuntimeContext.dataset_name must be a non-empty string.")
        if not isinstance(self.config, dict):
            raise ValueError("SearchRuntimeContext.config must be a dict.")
        if not isinstance(self.penalty, int) or self.penalty <= 0:
            raise ValueError("SearchRuntimeContext.penalty must be a positive integer.")


class RNNVAEArchitectureMultiObj(Problem):
    """
    This class defines the multiobjective problem for RNN-VAE architecture search.
    The three objectives are:
      1. The error (from validation/test metrics)
      2. The efficiency objective (params|macs|latency_ms)
      3. The PdM objective (1 - clipped fixed-theta quality proxy)
    All are minimized.
    """

    def __init__(self, dimension: int, runner: "SearchRunner"):
        self.runner = runner
        self.config = runner.ctx.config
        self.conn = runner.ctx.conn
        self.datamodule = runner.ctx.datamodule
        self.dataset_name = runner.ctx.dataset_name
        self.penalty = runner.ctx.penalty
        self.objective_contract = _resolve_objective_contract(self.config)
        self.iteration = 0
        self.stats = {
            "trained": 0,
            "cached": 0,
            "cached_db": 0,
            "cached_memory": 0,
            "cache_miss": 0,
            "invalid": 0,
            "failed": 0,
        }
        self.objective_cache_by_hash: dict[str, dict[str, Any]] = {}
        self.local_candidate_rows: list[dict[str, Any]] = []
        super().__init__(n_var=dimension, n_obj=3, n_constr=0, xl=0, xu=1)

    def _to_int(self, value, default: int | None = None):
        if default is None:
            default = self.penalty
        try:
            return int(round(float(value)))
        except Exception:
            return int(default)

    @staticmethod
    def _format_metric(value):
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    def _objectives_are_reusable(self, obj_error, obj_efficiency, obj_pdm) -> bool:
        try:
            values = [float(obj_error), float(obj_efficiency), float(obj_pdm)]
        except Exception:
            return False
        return all(math.isfinite(value) and value < float(self.penalty) for value in values)

    def _remember_objectives(
        self,
        model_hash: str,
        obj_error,
        obj_efficiency,
        obj_pdm,
        pdm_signal_quality=None,
    ) -> None:
        if not self._objectives_are_reusable(obj_error, obj_efficiency, obj_pdm):
            return
        self.objective_cache_by_hash[str(model_hash)] = {
            "obj_error": float(obj_error),
            "obj_efficiency": float(obj_efficiency),
            "obj_pdm": float(obj_pdm),
            "pdm_signal_quality": (
                None if pdm_signal_quality is None else float(pdm_signal_quality)
            ),
        }

    def _selected_metrics(self, experiment):
        selected = self.config.get("nia_search", {}).get("metrics", [])
        if isinstance(selected, str):
            selected = [selected]

        metrics = []
        anomaly_metrics = getattr(experiment, "anomaly_metrics", {}) if experiment else {}

        for metric_name in selected:
            metric_key = str(metric_name).strip()
            key_upper = metric_key.upper()
            value = getattr(experiment.metrics, key_upper, None)
            if (value is None or value == self.penalty) and isinstance(anomaly_metrics, dict):
                if metric_key in anomaly_metrics:
                    value = anomaly_metrics.get(metric_key)
                elif metric_key.lower() in anomaly_metrics:
                    value = anomaly_metrics.get(metric_key.lower())

            if value is None or value == self.penalty:
                continue
            if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                continue
            metrics.append(f"{metric_key}={self._format_metric(value)}")

        return metrics

    def _log_iteration_result(
            self,
            iteration,
            hash_id,
            status,
            obj_error,
            obj_efficiency,
            obj_pdm,
            pdm_signal_quality=None,
            duration_s=None,
            reason=None,
            metric_parts=None
    ):
        parts = [
            "ITER_RESULT",
            f"iter={iteration}",
            f"hash={hash_id}",
            f"status={status}",
            f"obj_error={self._format_metric(obj_error)}",
            f"obj_efficiency={self._format_metric(obj_efficiency)}",
            f"obj_pdm={self._format_metric(obj_pdm)}",
        ]
        if pdm_signal_quality is not None:
            parts.append(f"pdm_signal_quality={self._format_metric(pdm_signal_quality)}")
        if duration_s is not None:
            parts.append(f"duration_s={float(duration_s):.1f}")
        if reason:
            parts.append(f"reason={reason}")
        if metric_parts:
            parts.extend(metric_parts)
        Log.info(" ".join(parts))

    def _store_local_candidate(
        self,
        iteration: int,
        model_hash: str,
        solution,
        obj_error: float,
        obj_efficiency: float,
        obj_pdm: float,
        status: str,
    ) -> None:
        """
        Keep a local, in-memory candidate snapshot so winner selection can still complete
        when DB connectivity is temporarily unavailable.
        """
        if solution is None:
            return
        try:
            solution_arr = np.asarray(solution, dtype=float).reshape(-1)
        except Exception:
            return
        if solution_arr.size != int(self.n_var):
            return
        if not np.isfinite(solution_arr).all():
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        safe_obj_error = float(obj_error) if math.isfinite(float(obj_error)) else float(self.penalty)
        safe_obj_eff = float(obj_efficiency) if math.isfinite(float(obj_efficiency)) else float(self.penalty)
        safe_obj_pdm = float(obj_pdm) if math.isfinite(float(obj_pdm)) else float(self.penalty)

        self.local_candidate_rows.append(
            {
                "id": int(iteration),
                "hash_id": str(model_hash),
                "solution_array": json.dumps(solution_arr.tolist()),
                "obj_error": safe_obj_error,
                "obj_efficiency": safe_obj_eff,
                "obj_pdm": safe_obj_pdm,
                "algorithm_name": "NSGA3",
                "timestamp": timestamp,
                "source": "local_runtime_buffer",
                "status": status,
            }
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for solution in X:
            self.iteration += 1
            obj_error = self.penalty
            obj_efficiency = self.penalty
            obj_pdm = self.penalty
            pdm_signal_quality = None
            status = "failed"
            reason = None
            duration = None
            metric_parts = []

            model = RNNVAE(solution, **self.config)
            model_hash = model.get_hash()

            path = self.config['logging_params']['save_dir']
            Path(path).mkdir(parents=True, exist_ok=True)

            existing_entry = None
            cached_objectives = self.objective_cache_by_hash.get(str(model_hash))
            if cached_objectives is not None:
                status = "cached_memory"
                reason = None
                self.stats["cached"] += 1
                self.stats["cached_memory"] += 1
                obj_error = cached_objectives["obj_error"]
                obj_efficiency = cached_objectives["obj_efficiency"]
                obj_pdm = cached_objectives["obj_pdm"]
                pdm_signal_quality = cached_objectives.get("pdm_signal_quality")
            else:
                existing_entry = self.conn.get_entries(hash_id=model_hash, dataset_name=self.dataset_name)
                if existing_entry.shape[0] > 0:
                    cached_row = existing_entry.iloc[0].to_dict()
                    cached_bundle = calculate_objective_bundle_from_cached_row(
                        model=model,
                        cached_row=cached_row,
                        seq_len=self.config['data_params']['seq_len'],
                        n_features=self.config['data_params']['n_features'],
                        cfg=self.config,
                        penalty=self.penalty,
                    )
                    cached_reusable = (
                        cached_bundle.get("valid")
                        and self._objectives_are_reusable(
                            cached_bundle.get("obj_error"),
                            cached_bundle.get("obj_efficiency"),
                            cached_bundle.get("obj_pdm"),
                        )
                    )
                    if cached_reusable:
                        status = "cached_db"
                        reason = None
                        self.stats["cached"] += 1
                        self.stats["cached_db"] += 1
                        obj_error = cached_bundle["obj_error"]
                        obj_efficiency = cached_bundle["obj_efficiency"]
                        obj_pdm = cached_bundle["obj_pdm"]
                        pdm_signal_quality = cached_bundle.get("pdm_signal_quality")
                        self._remember_objectives(
                            model_hash,
                            obj_error,
                            obj_efficiency,
                            obj_pdm,
                            pdm_signal_quality,
                        )
                    else:
                        reason = f"cached_objective_miss:{cached_bundle.get('reason') or 'penalty_objectives'}"
                        self.stats["cache_miss"] += 1
                        existing_entry = existing_entry.iloc[0:0]

            if status not in {"cached_memory", "cached_db"}:
                if not model.is_valid:
                    status = "invalid"
                    reason = reason or "invalid_architecture"
                    self.stats["invalid"] += 1
                    obj_error = self.penalty
                    obj_efficiency = self.penalty
                    obj_pdm = self.penalty
                    self.conn.save_model_and_entry(
                        dataset_name=self.dataset_name,
                        alg_name="NSGA3",
                        iteration=self.iteration,
                        model=model,
                        solution=solution,
                        obj_error=obj_error,
                        obj_efficiency=obj_efficiency,
                        obj_pdm=obj_pdm,
                        objective_contract=self.objective_contract,
                    )
                else:
                    trainer = None
                    experiment = None
                    start_time = None
                    end_time = None
                    experiment = RNNVAExperiment(model, self.dataset_name, "NSGA3", **self.config)
                    trainer = Trainer(
                        enable_progress_bar=False,
                        accelerator="gpu" if torch.cuda.is_available() else "cpu",
                        devices=1,
                        default_root_dir=path,
                        log_every_n_steps=50,
                        logger=False,
                        enable_checkpointing=False,
                        **self.config['trainer_params']
                    )

                    try:
                        start_time = datetime.now()
                        trainer.fit(experiment, datamodule=self.datamodule)
                        experiment.collect_calibration_scores(self.datamodule.train_dataloader())
                        trainer.test(experiment, datamodule=self.datamodule)
                        end_time = datetime.now()
                        duration = (end_time - start_time).total_seconds()

                        objective_bundle = calculate_objective_bundle_from_experiment(
                            model=model,
                            experiment=experiment,
                            seq_len=self.config['data_params']['seq_len'],
                            n_features=self.config['data_params']['n_features'],
                            cfg=self.config,
                            penalty=self.penalty,
                        )
                        obj_error = objective_bundle["obj_error"]
                        obj_efficiency = objective_bundle["obj_efficiency"]
                        obj_pdm = objective_bundle["obj_pdm"]
                        pdm_signal_quality = objective_bundle.get("pdm_signal_quality")
                        metric_parts = self._selected_metrics(experiment)

                        if not objective_bundle.get("valid"):
                            status = "failed"
                            reason = objective_bundle.get("reason") or "objective_bundle_invalid"
                            self.stats["failed"] += 1
                        elif (
                            obj_error >= self.penalty
                            or obj_efficiency >= self.penalty
                            or obj_pdm >= self.penalty
                        ):
                            status = "failed"
                            reason = "penalty_objectives"
                            self.stats["failed"] += 1
                        else:
                            status = "trained"
                            reason = None
                            self.stats["trained"] += 1

                        self.conn.save_model_and_entry(
                            dataset_name=self.dataset_name,
                            alg_name="NSGA3",
                            iteration=self.iteration,
                            solution=solution,
                            model=model,
                            experiment=experiment,
                            obj_error=obj_error,
                            obj_efficiency=obj_efficiency,
                            obj_pdm=obj_pdm,
                            objective_contract=objective_bundle.get("objective_contract"),
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration
                        )
                        if status == "trained":
                            self._remember_objectives(
                                model_hash,
                                obj_error,
                                obj_efficiency,
                                obj_pdm,
                                pdm_signal_quality,
                            )
                    except Exception as e:
                        end_time = datetime.now()
                        if start_time is not None:
                            duration = (end_time - start_time).total_seconds()
                        status = "failed"
                        reason = _short_exception_reason(e)
                        self.stats["failed"] += 1
                        obj_error = self.penalty
                        obj_efficiency = self.penalty
                        obj_pdm = self.penalty
                        pdm_signal_quality = None
                        metric_parts = []
                        Log.error(
                            f"CANDIDATE_EVAL_FAILED iter={self.iteration} hash={model_hash} "
                            f"cycle_id={self.config.get('data_params', {}).get('cycle_id')} "
                            f"reason={reason}"
                        )
                        self.conn.save_model_and_entry(
                            dataset_name=self.dataset_name,
                            alg_name="NSGA3",
                            iteration=self.iteration,
                            solution=solution,
                            model=model,
                            experiment=None,
                            obj_error=obj_error,
                            obj_efficiency=obj_efficiency,
                            obj_pdm=obj_pdm,
                            objective_contract=self.objective_contract,
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration
                        )
                    finally:
                        _cleanup_candidate_runtime(trainer=trainer, experiment=experiment, model=model)

            if np.isnan(obj_error) or np.isnan(obj_efficiency) or np.isnan(obj_pdm):
                obj_error = self.penalty
                obj_efficiency = self.penalty
                obj_pdm = self.penalty
                pdm_signal_quality = None
                status = "failed"
                reason = "nan_objective"
                self.stats["failed"] += 1

            self._log_iteration_result(
                iteration=self.iteration,
                hash_id=model_hash,
                status=status,
                obj_error=obj_error,
                obj_efficiency=obj_efficiency,
                obj_pdm=obj_pdm,
                pdm_signal_quality=pdm_signal_quality,
                duration_s=duration,
                reason=reason,
                metric_parts=metric_parts
            )

            self._store_local_candidate(
                iteration=self.iteration,
                model_hash=model_hash,
                solution=solution,
                obj_error=obj_error,
                obj_efficiency=obj_efficiency,
                obj_pdm=obj_pdm,
                status=status,
            )

            F.append([obj_error, obj_efficiency, obj_pdm])
        out["F"] = np.array(F)


class SearchRunner:
    def __init__(self, context: SearchRuntimeContext):
        self.ctx = context

    def _db_wait_budget_seconds(self) -> int:
        raw_env = os.getenv("NIANETVAE_DB_WAIT_SECONDS")
        if raw_env is not None and str(raw_env).strip():
            try:
                value = int(float(raw_env))
                if value >= 0:
                    return value
            except Exception:
                pass
        logging_cfg = dict(self.ctx.config.get("logging_params") or {})
        raw_cfg = logging_cfg.get("db_wait_seconds", 1800)
        try:
            value = int(float(raw_cfg))
        except Exception:
            value = 1800
        return max(0, value)

    def _db_wait_poll_seconds(self) -> int:
        logging_cfg = dict(self.ctx.config.get("logging_params") or {})
        raw_cfg = logging_cfg.get("db_wait_poll_seconds", 30)
        try:
            value = int(float(raw_cfg))
        except Exception:
            value = 30
        return max(1, value)

    def _fetch_cycle_candidates_with_wait(self, *, algorithm_name: str = "NSGA3") -> pd.DataFrame:
        """
        Fetch DB cycle candidates with a bounded wait window to tolerate transient DB outages.
        """
        wait_budget = self._db_wait_budget_seconds()
        poll_seconds = self._db_wait_poll_seconds()
        started = time.monotonic()
        attempt = 0

        while True:
            attempt += 1
            candidates_df = self.ctx.conn.get_cycle_candidates(
                dataset_name=self.ctx.dataset_name,
                algorithm_name=algorithm_name,
            )
            if candidates_df is not None and not candidates_df.empty:
                if attempt > 1:
                    elapsed = time.monotonic() - started
                    Log.info(
                        f"DB_RECOVERY_SUCCESS dataset={self.ctx.dataset_name} "
                        f"attempt={attempt} elapsed_s={elapsed:.1f} "
                        f"candidate_count={len(candidates_df)}"
                    )
                return candidates_df

            elapsed = time.monotonic() - started
            remaining = wait_budget - elapsed
            if remaining <= 0:
                Log.error(
                    f"DB_RECOVERY_TIMEOUT dataset={self.ctx.dataset_name} "
                    f"wait_budget_s={wait_budget} attempts={attempt}"
                )
                return candidates_df if candidates_df is not None else pd.DataFrame()

            sleep_s = min(float(poll_seconds), max(1.0, remaining))
            Log.warning(
                f"DB_RECOVERY_WAIT dataset={self.ctx.dataset_name} "
                f"attempt={attempt} elapsed_s={elapsed:.1f} remaining_s={remaining:.1f} "
                f"sleep_s={sleep_s:.1f}"
            )
            time.sleep(sleep_s)

    def export_skipped_non_trainable_cycle(self, reason: str, detail: str = "", source: str = "runtime"):
        return _export_skipped_non_trainable_cycle(
            reason=reason,
            detail=detail,
            source=source,
            config=self.ctx.config,
            run_uuid=self.ctx.run_uuid,
        )

    def run_per_maint_finetune_cycle(self):
        data_params = self.ctx.config.get("data_params", {})
        cycle_id = data_params.get("cycle_id")
        if cycle_id is None:
            raise ValueError("per_maint_finetune requires data_params.cycle_id.")
        cycle_id = int(cycle_id)

        if cycle_id == 0:
            Log.info("FINETUNE_MODE cycle_id=0 uses baseline_search for initial architecture.")
            self.solve_architecture_problem()
            return

        previous_source = _find_latest_trained_cycle_artifacts_before(
            cycle_id,
            config=self.ctx.config,
            run_uuid=self.ctx.run_uuid,
        )
        if previous_source is None:
            raise FileNotFoundError(
                "per_maint_finetune requires previous cycle artifacts. "
                f"No trained cycle artifacts found before cycle {cycle_id:02d}."
            )
        previous_cycle_id, _, previous_weights, previous_meta = previous_source
        current_cycle_dir = _resolve_export_dir(self.ctx.config, run_uuid=self.ctx.run_uuid)

        previous_metadata = json.loads(previous_meta.read_text(encoding="utf-8"))
        previous_solution = previous_metadata.get("solution")
        if previous_solution is None:
            raise ValueError(
                f"Previous cycle metadata is missing solution array: {previous_meta}"
            )

        seed_everything(self.ctx.config['exp_params']['manual_seed'], True)
        model = RNNVAE(previous_solution, **self.ctx.config)
        if not model.is_valid:
            raise ValueError(
                "Previous cycle solution produced an invalid architecture during finetune setup."
            )

        state_dict = torch.load(previous_weights, map_location="cpu")
        model.load_state_dict(state_dict)
        finetune_policy = _resolve_finetune_policy(self.ctx.config)

        Log.info(
            f"FINETUNE_START cycle_id={cycle_id:02d} source_cycle={previous_cycle_id:02d} "
            f"source_model={previous_weights}"
        )
        Log.info(
            "FINETUNE_POLICY "
            f"base_learning_rate={finetune_policy['base_learning_rate']} "
            f"learning_rate_scale={finetune_policy['learning_rate_scale']} "
            f"finetune_learning_rate={finetune_policy['finetune_learning_rate']} "
            f"min_epochs={finetune_policy['trainer_params_override']['min_epochs']} "
            f"max_epochs={finetune_policy['trainer_params_override']['max_epochs']} "
            "scheduler=none"
        )
        final_result = _run_training_with_model(
            model,
            "PER_MAINT_FINETUNE",
            learning_rate=finetune_policy["finetune_learning_rate"],
            trainer_params_override=finetune_policy["trainer_params_override"],
            config=self.ctx.config,
            dataset_name=self.ctx.dataset_name,
            datamodule=self.ctx.datamodule,
            penalty=self.ctx.penalty,
        )
        Log.info(
            f"FINETUNE_DONE cycle_id={cycle_id:02d} source_cycle={previous_cycle_id:02d} "
            f"obj_error={final_result['obj_error']} obj_efficiency={final_result['obj_efficiency']} "
            f"obj_pdm={final_result['obj_pdm']}"
        )

        export_enabled = bool(self.ctx.config.get("logging_params", {}).get("export_enabled", False))
        if not export_enabled:
            return

        search_result = {
            "mode": "per_maint_finetune",
            "search_performed": False,
            "source_cycle_id": previous_cycle_id,
            "source_cycle_key": f"{previous_cycle_id:02d}",
            "source_weights_file": str(previous_weights),
        }
        model_path, meta_path, summary_path = _export_cycle_artifacts(
            export_dir=current_cycle_dir,
            model=final_result["model"],
            best_solution=previous_solution,
            best_algorithm="PER_MAINT_FINETUNE",
            search_result=search_result,
            final_result=final_result,
            config=self.ctx.config,
            dataset_name=self.ctx.dataset_name,
            run_uuid=self.ctx.run_uuid,
        )
        Log.info(
            f"MODEL_EXPORT_READY dir={current_cycle_dir} "
            f"weights={model_path.name} meta={meta_path.name} summary={summary_path.name}"
        )

    def run_per_maint_warmstart_cycle(self):
        data_params = self.ctx.config.get("data_params", {})
        cycle_id = data_params.get("cycle_id")
        if cycle_id is None:
            raise ValueError("per_maint_warmstart_search requires data_params.cycle_id.")
        cycle_id = int(cycle_id)

        nia_search_cfg = dict(self.ctx.config.get("nia_search") or {})
        warm_cfg = dict(nia_search_cfg.get("warm_start") or {})
        if not bool(warm_cfg.get("enabled", False)):
            Log.warning(
                "WARMSTART_MODE warm_start.enabled=false in config; "
                "enabling warm-start automatically for this run."
            )
        warm_cfg["enabled"] = True
        nia_search_cfg["warm_start"] = warm_cfg
        self.ctx.config["nia_search"] = nia_search_cfg

        if cycle_id == 0:
            Log.info(
                "WARMSTART_MODE cycle_id=00 uses random initialization by design."
            )
        else:
            Log.info(
                f"WARMSTART_MODE cycle_id={cycle_id:02d} "
                "uses warm-start search with previous-cycle anchor when available."
            )
        self.solve_architecture_problem()

    def solve_architecture_problem(self):
        """
        Uses pymoo's NSGA-III to perform a three-objective search:
          1) obj_error
          2) obj_efficiency
          3) obj_pdm
        Objective contract and fixed training policy are taken from the configuration file.
        """
        dimensionality = RNNVAE.GENE_DIMENSION
        problem = RNNVAEArchitectureMultiObj(dimension=dimensionality, runner=self)

        time_str = self.ctx.config['nia_search']['time']
        try:
            hours, minutes, seconds = map(int, time_str.split(":"))
            max_time = hours * 3600 + minutes * 60 + seconds
        except Exception as e:
            Log.error(f"Error parsing time limit from config: {time_str}. Ensure it is in HH:MM:SS format.")
            raise e
        termination = get_termination("time", max_time=max_time)

        nsga3_cfg = dict((self.ctx.config.get("nia_search") or {}).get("nsga3") or {})
        n_partitions = int(nsga3_cfg["n_partitions"])
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
        effective_population = int(ref_dirs.shape[0])

        warm_start = _resolve_warm_start_sampling(
            dimensionality,
            effective_population=effective_population,
            config=self.ctx.config,
            run_uuid=self.ctx.run_uuid,
        )
        search_init_mode = warm_start["init_mode"]
        seed_source = "random"
        if warm_start["enabled"]:
            algorithm = NSGA3(
                ref_dirs=ref_dirs,
                sampling=warm_start["sampling"],
            )
            seed_source = f"cycle_{warm_start['source_cycle_id']:02d}"
            details = warm_start["details"]
            Log.info(
                "WARMSTART_INIT "
                f"cycle_id={int((self.ctx.config.get('data_params') or {}).get('cycle_id', -1)):02d} "
                f"source_cycle={warm_start['source_cycle_id']:02d} "
                f"carry_over={details['carry_over_count']} "
                f"perturb={details['perturb_count']} "
                f"random={details['random_count']} "
                f"perturbation_strength={details['perturbation_strength']} "
                f"effective_population={details.get('population_size')}"
            )
        else:
            algorithm = NSGA3(ref_dirs=ref_dirs)
            if warm_start["reason"]:
                Log.info(f"WARMSTART_FALLBACK reason={warm_start['reason']} init_mode=random")

        n_jobs = torch.cuda.device_count() if torch.cuda.is_available() else 1
        objective_contract = _resolve_objective_contract(self.ctx.config)
        selection_contract = _resolve_winner_selection_contract(self.ctx.config)

        Log.info(
            f"SEARCH_START algorithm=NSGA3 n_partitions={n_partitions} "
            f"ref_dirs={effective_population} effective_population={effective_population} "
            f"time_limit={time_str} time_limit_seconds={max_time} n_jobs={n_jobs} "
            f"metrics={self.ctx.config['nia_search'].get('metrics')} "
            f"search_init_mode={search_init_mode} seed_source={seed_source} "
            f"fixed_optimizer={self.ctx.config['exp_params'].get('optimizer')} "
            f"obj_error={objective_contract['error_metric']} "
            f"obj_efficiency={objective_contract['efficiency_metric']} "
            "obj_pdm=clip(0.5*(1-pdm_risk_gap),0,1) "
            f"pdm_metric={objective_contract['pdm_metric']} "
            f"winner_selection={selection_contract['method']} "
            f"winner_weights="
            f"{selection_contract['weights_normalized']['error']:.4f}/"
            f"{selection_contract['weights_normalized']['efficiency']:.4f}/"
            f"{selection_contract['weights_normalized']['pdm']:.4f}"
        )
        minimize(
            problem,
            algorithm,
            termination,
            seed=self.ctx.config['exp_params']['manual_seed'],
            verbose=True,
            n_jobs=n_jobs,
        )

        candidate_rows = self._fetch_cycle_candidates_with_wait(algorithm_name="NSGA3")
        candidate_source = "db"
        if candidate_rows is None or candidate_rows.empty:
            local_rows = list(problem.local_candidate_rows)
            if local_rows:
                candidate_rows = pd.DataFrame(local_rows)
                candidate_source = "local_runtime_buffer"
                Log.warning(
                    f"WINNER_SELECTION_FALLBACK dataset={self.ctx.dataset_name} "
                    f"source={candidate_source} row_count={len(candidate_rows)} "
                    "reason=db_unavailable_or_empty_after_wait"
                )
            else:
                raise ValueError(
                    f"Winner selection failed for {self.ctx.dataset_name}: "
                    "no DB candidates found after wait and local runtime buffer is empty."
                )

        winner_selection = _select_deterministic_pareto_winner(
            candidates_df=candidate_rows,
            selection_contract=selection_contract,
            dataset_name=self.ctx.dataset_name,
            penalty=self.ctx.penalty,
            expected_solution_dim=RNNVAE.GENE_DIMENSION,
        )
        best_solution = np.asarray(winner_selection["selected_solution"], dtype=float)
        best_algorithm = winner_selection["selected_algorithm"]

        Log.info(
            "WINNER_SELECTION "
            f"dataset={self.ctx.dataset_name} method={winner_selection['method']} "
            f"source={candidate_source} "
            f"candidate_count={winner_selection['candidate_count']} "
            f"valid_count={winner_selection['valid_candidate_count']} "
            f"dedup_count={winner_selection['deduplicated_candidate_count']} "
            f"pareto_count={winner_selection['pareto_candidate_count']} "
            f"weights={winner_selection['weights_normalized']['error']:.4f}/"
            f"{winner_selection['weights_normalized']['efficiency']:.4f}/"
            f"{winner_selection['weights_normalized']['pdm']:.4f} "
            f"selected_hash={winner_selection['selected_hash']} "
            f"selected_obj_error={winner_selection['selected_objectives']['obj_error']:.4f} "
            f"selected_obj_efficiency={winner_selection['selected_objectives']['obj_efficiency']:.4f} "
            f"selected_obj_pdm={winner_selection['selected_objectives']['obj_pdm']:.4f} "
            f"selected_distance={winner_selection['selected_distance']:.6f}"
        )

        search_result = {
            "iterations": problem.iteration,
            "trained": problem.stats["trained"],
            "cached": problem.stats["cached"],
            "cached_db": problem.stats["cached_db"],
            "cached_memory": problem.stats["cached_memory"],
            "cache_miss": problem.stats["cache_miss"],
            "invalid": problem.stats["invalid"],
            "failed": problem.stats["failed"],
            "best_hash": winner_selection["selected_hash"],
            "selected_distance": winner_selection["selected_distance"],
            "best_solution": _as_jsonable(best_solution),
            "time_limit": time_str,
            "time_limit_seconds": max_time,
            "algorithm": "NSGA3",
            "n_partitions": n_partitions,
            "ref_dirs": effective_population,
            "effective_population": effective_population,
            "init_mode": search_init_mode,
            "seed_source": seed_source,
            "warm_start": _as_jsonable(warm_start["details"]),
            "source_cycle_id": warm_start.get("source_cycle_id"),
            "source_cycle_key": (
                f"{int(warm_start['source_cycle_id']):02d}"
                if warm_start.get("source_cycle_id") is not None
                else None
            ),
            "winner_selection": _as_jsonable(
                {k: v for k, v in winner_selection.items() if k != "selected_solution"}
            ),
            "winner_selection_source": candidate_source,
        }

        final_result = _run_final_training(
            best_solution,
            config=self.ctx.config,
            dataset_name=self.ctx.dataset_name,
            datamodule=self.ctx.datamodule,
            penalty=self.ctx.penalty,
        )
        best_model = final_result["model"]
        model_file = (
            Path(self.ctx.config['logging_params']['save_dir'])
            / f"{self.ctx.dataset_name}_NSGA3_{best_model.hash_id}.pt"
        )
        torch.save(best_model.state_dict(), model_file)
        Log.info(
            f"SEARCH_DONE iterations={problem.iteration} trained={problem.stats['trained']} "
            f"cached={problem.stats['cached']} cached_db={problem.stats['cached_db']} "
            f"cached_memory={problem.stats['cached_memory']} cache_miss={problem.stats['cache_miss']} "
            f"invalid={problem.stats['invalid']} failed={problem.stats['failed']} "
            f"best_hash={best_model.hash_id} selected_distance={winner_selection['selected_distance']}"
        )
        Log.info(f"BEST_MODEL_SAVED path={model_file}")

        export_enabled = bool(self.ctx.config.get("logging_params", {}).get("export_enabled", False))
        if export_enabled:
            export_dir = _resolve_export_dir(self.ctx.config, run_uuid=self.ctx.run_uuid)
            model_path, meta_path, summary_path = _export_cycle_artifacts(
                export_dir=export_dir,
                model=best_model,
                best_solution=best_solution,
                best_algorithm=best_algorithm,
                search_result=search_result,
                final_result=final_result,
                config=self.ctx.config,
                dataset_name=self.ctx.dataset_name,
                run_uuid=self.ctx.run_uuid,
            )
            Log.info(
                f"MODEL_EXPORT_READY dir={export_dir} "
                f"weights={model_path.name} meta={meta_path.name} summary={summary_path.name}"
            )


__all__ = [
    "DEFAULT_PENALTY",
    "SearchStorageConnectorProtocol",
    "SearchRuntimeContext",
    "SearchRunner",
    "RNNVAEArchitectureMultiObj",
]
