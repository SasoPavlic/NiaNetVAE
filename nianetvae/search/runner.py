import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
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
      3. The PdM objective (1 - AUPRC_premaint)
    All are minimized.
    """

    def __init__(self, dimension: int, runner: "SearchRunner"):
        self.runner = runner
        self.config = runner.ctx.config
        self.conn = runner.ctx.conn
        self.datamodule = runner.ctx.datamodule
        self.dataset_name = runner.ctx.dataset_name
        self.penalty = runner.ctx.penalty
        self.iteration = 0
        self.stats = {"trained": 0, "cached": 0, "invalid": 0, "failed": 0}
        self.best_fitness = None
        self.best_hash = None
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
            fitness,
            error,
            complexity,
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
            f"fitness={self._to_int(fitness)}",
            f"error={self._to_int(error)}",
            f"complexity={self._to_int(complexity)}",
            f"obj_error={self._format_metric(error)}",
            f"obj_efficiency={self._format_metric(complexity)}",
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

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for solution in X:
            self.iteration += 1
            fitness = self.penalty
            error = self.penalty
            complexity = self.penalty
            obj_pdm = self.penalty
            pdm_signal_quality = None
            status = "failed"
            reason = None
            duration = None
            metric_parts = []

            model = RNNVAE(solution, **self.config)
            model_hash = model.get_hash()
            existing_entry = self.conn.get_entries(hash_id=model_hash, dataset_name=self.dataset_name)

            path = self.config['logging_params']['save_dir']
            Path(path).mkdir(parents=True, exist_ok=True)

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
                if cached_bundle.get("valid"):
                    status = "cached"
                    reason = None
                    self.stats["cached"] += 1
                    error = cached_bundle["obj_error"]
                    complexity = cached_bundle["obj_efficiency"]
                    obj_pdm = cached_bundle["obj_pdm"]
                    pdm_signal_quality = cached_bundle.get("pdm_signal_quality")
                    fitness = cached_bundle["fitness"]
                else:
                    reason = f"cached_objective_miss:{cached_bundle.get('reason') or 'invalid'}"
                    existing_entry = existing_entry.iloc[0:0]

            if existing_entry.shape[0] == 0:
                if not model.is_valid:
                    status = "invalid"
                    reason = reason or "invalid_architecture"
                    self.stats["invalid"] += 1
                    error = self.penalty
                    complexity = self.penalty
                    obj_pdm = self.penalty
                    fitness = self.penalty
                    self.conn.save_model_and_entry(
                        dataset_name=self.dataset_name,
                        alg_name="NSGA3",
                        iteration=self.iteration,
                        model=model,
                        fitness=self.penalty,
                        solution=solution,
                        error=error,
                        complexity=complexity
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
                        error = objective_bundle["obj_error"]
                        complexity = objective_bundle["obj_efficiency"]
                        obj_pdm = objective_bundle["obj_pdm"]
                        pdm_signal_quality = objective_bundle.get("pdm_signal_quality")
                        fitness = objective_bundle["fitness"]
                        metric_parts = self._selected_metrics(experiment)

                        if not objective_bundle.get("valid"):
                            status = "failed"
                            reason = objective_bundle.get("reason") or "objective_bundle_invalid"
                            self.stats["failed"] += 1
                        elif fitness >= self.penalty or error >= self.penalty or complexity >= self.penalty or obj_pdm >= self.penalty:
                            status = "failed"
                            reason = "penalty_fitness"
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
                            error=error,
                            model=model,
                            experiment=experiment,
                            fitness=fitness,
                            complexity=complexity,
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration
                        )
                    except Exception as e:
                        end_time = datetime.now()
                        if start_time is not None:
                            duration = (end_time - start_time).total_seconds()
                        status = "failed"
                        reason = _short_exception_reason(e)
                        self.stats["failed"] += 1
                        fitness = self.penalty
                        error = self.penalty
                        complexity = self.penalty
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
                            error=error,
                            model=model,
                            experiment=None,
                            fitness=fitness,
                            complexity=complexity,
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration
                        )
                    finally:
                        _cleanup_candidate_runtime(trainer=trainer, experiment=experiment, model=model)

            if np.isnan(error) or np.isnan(complexity) or np.isnan(obj_pdm):
                fitness = self.penalty
                error = self.penalty
                complexity = self.penalty
                obj_pdm = self.penalty
                pdm_signal_quality = None
                status = "failed"
                reason = "nan_objective"
                self.stats["failed"] += 1

            if self.best_fitness is None or fitness < self.best_fitness:
                self.best_fitness = self._to_int(fitness)
                self.best_hash = model_hash

            self._log_iteration_result(
                iteration=self.iteration,
                hash_id=model_hash,
                status=status,
                fitness=fitness,
                error=error,
                complexity=complexity,
                obj_pdm=obj_pdm,
                pdm_signal_quality=pdm_signal_quality,
                duration_s=duration,
                reason=reason,
                metric_parts=metric_parts
            )

            F.append([error, complexity, obj_pdm])
        out["F"] = np.array(F)


class SearchRunner:
    def __init__(self, context: SearchRuntimeContext):
        self.ctx = context

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
            f"fitness={final_result['fitness']} error={final_result['error']} "
            f"complexity={final_result['complexity']}"
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
        Objective contract and optimizer settings are taken from the configuration file.
        """
        dimensionality = 7
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
            f"obj_error={objective_contract['error_metric']} "
            f"obj_efficiency={objective_contract['efficiency_metric']} "
            f"obj_pdm=1-{objective_contract['pdm_metric']} "
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

        candidate_rows = self.ctx.conn.get_cycle_candidates(dataset_name=self.ctx.dataset_name, algorithm_name="NSGA3")
        winner_selection = _select_deterministic_pareto_winner(
            candidates_df=candidate_rows,
            selection_contract=selection_contract,
            dataset_name=self.ctx.dataset_name,
            penalty=self.ctx.penalty,
        )
        best_solution = np.asarray(winner_selection["selected_solution"], dtype=float)
        best_algorithm = winner_selection["selected_algorithm"]

        Log.info(
            "WINNER_SELECTION "
            f"dataset={self.ctx.dataset_name} method={winner_selection['method']} "
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
            "invalid": problem.stats["invalid"],
            "failed": problem.stats["failed"],
            "best_hash": winner_selection["selected_hash"],
            "best_fitness": winner_selection["selected_distance"],
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
            f"cached={problem.stats['cached']} invalid={problem.stats['invalid']} failed={problem.stats['failed']} "
            f"best_hash={best_model.hash_id} best_fitness={winner_selection['selected_distance']}"
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
