import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch import Trainer
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from log import Log
from nianetvae.experiments.rnn_vae_experiment import RNNVAExperiment
from nianetvae.models.rnn_vae import RNNVAE

# Global variables that are set by main.py before calling solve_architecture_problem
RUN_UUID = None
config = None
conn = None
datamodule = None
dataset_name = None
PENALTY = int(9e10)


def _as_jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    return value


def _get_git_ref() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _resolve_export_dir(cfg: dict) -> Path:
    logging_params = cfg.get("logging_params", {})
    export_root = logging_params.get("model_export_dir", "logs/per_maint_models")
    dataset = str(cfg.get("data_params", {}).get("dataset_name", "dataset")).strip() or "dataset"
    regime = str(cfg.get("data_params", {}).get("regime", "")).strip().lower()
    cycle_id = cfg.get("data_params", {}).get("cycle_id")

    if regime == "per_maint" and cycle_id is not None:
        try:
            cycle_dir = f"cycle_{int(cycle_id):02d}"
        except Exception:
            cycle_dir = f"cycle_{cycle_id}"
        return Path(export_root) / dataset / cycle_dir

    run_label = RUN_UUID or datetime.now().strftime("%Y%m%d%H%M%S")
    return Path(export_root) / dataset / f"run_{run_label}"


def _build_final_trainer(default_root_dir: str, trainer_params_override: dict | None = None):
    trainer_params = dict(config.get('trainer_params', {}))
    if trainer_params_override:
        trainer_params.update(trainer_params_override)
    return Trainer(
        enable_progress_bar=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        default_root_dir=default_root_dir,
        log_every_n_steps=50,
        logger=False,
        enable_checkpointing=False,
        **trainer_params
    )


def _run_training_with_model(
        model: RNNVAE,
        algorithm_name: str,
        learning_rate: float | None = None,
        trainer_params_override: dict | None = None,
):
    final_root = config['logging_params']['save_dir']
    experiment = RNNVAExperiment(model, dataset_name, algorithm_name, **config)
    if learning_rate is not None:
        experiment.learning_rate = float(learning_rate)
    effective_trainer_params = dict(config.get("trainer_params", {}))
    if trainer_params_override:
        effective_trainer_params.update(trainer_params_override)
    Log.info(
        "TRAINING_POLICY "
        f"alg={algorithm_name} optimizer={model.optimizer_name} "
        f"learning_rate={experiment.learning_rate} scheduler=none "
        f"min_epochs={effective_trainer_params.get('min_epochs')} "
        f"max_epochs={effective_trainer_params.get('max_epochs')}"
    )
    trainer = _build_final_trainer(
        default_root_dir=final_root,
        trainer_params_override=trainer_params_override,
    )

    started_at = datetime.now()
    trainer.fit(experiment, datamodule=datamodule)
    trainer.test(experiment, datamodule=datamodule)
    ended_at = datetime.now()
    duration_s = (ended_at - started_at).total_seconds()

    final_metrics = {}
    try:
        final_metrics = experiment.metrics.compute()
    except Exception:
        final_metrics = {}
    anomaly_metrics = getattr(experiment, "anomaly_metrics", {}) or {}
    fitness, error, complexity = calculate_fitness(
        model,
        experiment,
        config['data_params']['seq_len']
    )
    return {
        "model": model,
        "experiment": experiment,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_s": duration_s,
        "fitness": fitness,
        "error": error,
        "complexity": complexity,
        "metrics": final_metrics,
        "anomaly_metrics": anomaly_metrics,
    }


def _run_final_training(best_solution):
    seed_everything(config['exp_params']['manual_seed'], True)
    model = RNNVAE(best_solution, **config)
    if not model.is_valid:
        raise ValueError("Best solution produced an invalid model during final training.")
    return _run_training_with_model(model, "NSGA2")


def _resolve_finetune_policy() -> dict:
    workflow = config.get("workflow") or {}
    finetune_cfg = workflow.get("finetune") or {}
    exp_params = config.get("exp_params") or {}
    trainer_params = config.get("trainer_params") or {}

    base_lr = float(exp_params.get("learning_rate", 0.01))
    lr_scale = float(finetune_cfg.get("learning_rate_scale", 0.1))
    if base_lr <= 0:
        raise ValueError(f"Invalid exp_params.learning_rate={base_lr}. Must be > 0.")
    if lr_scale <= 0:
        raise ValueError(
            f"Invalid workflow.finetune.learning_rate_scale={lr_scale}. Must be > 0."
        )
    finetune_lr = base_lr * lr_scale

    max_epochs = int(finetune_cfg.get("max_epochs", 3))
    if max_epochs < 1:
        raise ValueError(
            f"Invalid workflow.finetune.max_epochs={max_epochs}. Must be >= 1."
        )

    default_min_epochs = int(trainer_params.get("min_epochs", 1))
    min_epochs = int(finetune_cfg.get("min_epochs", min(default_min_epochs, max_epochs)))
    if min_epochs < 1:
        raise ValueError(
            f"Invalid workflow.finetune.min_epochs={min_epochs}. Must be >= 1."
        )
    if min_epochs > max_epochs:
        raise ValueError(
            "Invalid fine-tune epoch policy: "
            f"workflow.finetune.min_epochs={min_epochs} > max_epochs={max_epochs}."
        )

    return {
        "base_learning_rate": base_lr,
        "learning_rate_scale": lr_scale,
        "finetune_learning_rate": finetune_lr,
        "trainer_params_override": {
            "min_epochs": min_epochs,
            "max_epochs": max_epochs,
        },
    }


def _resolve_cycle_export_dir(cycle_id: int) -> Path:
    cfg = {
        "logging_params": dict(config.get("logging_params", {})),
        "data_params": dict(config.get("data_params", {})),
    }
    cfg["data_params"]["regime"] = "per_maint"
    cfg["data_params"]["cycle_id"] = int(cycle_id)
    return _resolve_export_dir(cfg)


def _find_latest_trained_cycle_artifacts_before(cycle_id: int):
    for source_cycle_id in range(int(cycle_id) - 1, -1, -1):
        source_cycle_dir = _resolve_cycle_export_dir(source_cycle_id)
        source_weights = source_cycle_dir / "model.pt"
        source_meta = source_cycle_dir / "model_meta.json"
        if source_weights.exists() and source_meta.exists():
            return source_cycle_id, source_cycle_dir, source_weights, source_meta
    return None


def export_skipped_non_trainable_cycle(reason: str, detail: str = "", source: str = "runtime"):
    data_params = config.get("data_params", {})
    cycle_id = data_params.get("cycle_id")
    if cycle_id is None:
        return
    cycle_id = int(cycle_id)
    if cycle_id <= 0:
        return

    export_enabled = bool(config.get("logging_params", {}).get("export_enabled", False))
    if not export_enabled:
        return

    export_dir = _resolve_export_dir(config)
    export_dir.mkdir(parents=True, exist_ok=True)
    status_path = export_dir / "cycle_status.json"
    payload = {
        "schema_version": "1.0",
        "status": "skipped_non_trainable",
        "cycle_id": cycle_id,
        "dataset_name": data_params.get("dataset_name"),
        "regime": data_params.get("regime"),
        "reason": reason,
        "detail": detail,
        "source": source,
        "run_uuid": RUN_UUID,
        "created_at": datetime.now().isoformat(),
    }
    status_path.write_text(
        json.dumps(_as_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    Log.info(
        f"FINETUNE_SKIP_MARKER_WRITTEN cycle_id={cycle_id:02d} path={status_path} "
        f"reason={reason}"
    )


def run_per_maint_finetune_cycle():
    data_params = config.get("data_params", {})
    cycle_id = data_params.get("cycle_id")
    if cycle_id is None:
        raise ValueError("per_maint_finetune requires data_params.cycle_id.")
    cycle_id = int(cycle_id)

    if cycle_id == 0:
        Log.info("FINETUNE_MODE cycle_id=0 uses baseline_search for initial architecture.")
        solve_architecture_problem()
        return

    previous_source = _find_latest_trained_cycle_artifacts_before(cycle_id)
    if previous_source is None:
        raise FileNotFoundError(
            "per_maint_finetune requires previous cycle artifacts. "
            f"No trained cycle artifacts found before cycle {cycle_id:02d}."
        )
    previous_cycle_id, previous_cycle_dir, previous_weights, previous_meta = previous_source
    current_cycle_dir = _resolve_export_dir(config)

    previous_metadata = json.loads(previous_meta.read_text(encoding="utf-8"))
    previous_solution = previous_metadata.get("solution")
    if previous_solution is None:
        raise ValueError(
            f"Previous cycle metadata is missing solution array: {previous_meta}"
        )

    seed_everything(config['exp_params']['manual_seed'], True)
    model = RNNVAE(previous_solution, **config)
    if not model.is_valid:
        raise ValueError(
            "Previous cycle solution produced an invalid architecture during finetune setup."
        )

    state_dict = torch.load(previous_weights, map_location="cpu")
    model.load_state_dict(state_dict)
    finetune_policy = _resolve_finetune_policy()

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
    )
    Log.info(
        f"FINETUNE_DONE cycle_id={cycle_id:02d} source_cycle={previous_cycle_id:02d} "
        f"fitness={final_result['fitness']} error={final_result['error']} "
        f"complexity={final_result['complexity']}"
    )

    export_enabled = bool(config.get("logging_params", {}).get("export_enabled", False))
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
    )
    Log.info(
        f"MODEL_EXPORT_READY dir={current_cycle_dir} "
        f"weights={model_path.name} meta={meta_path.name} summary={summary_path.name}"
    )


def _export_cycle_artifacts(
        export_dir: Path,
        model: RNNVAE,
        best_solution,
        best_algorithm,
        search_result: dict,
        final_result: dict,
):
    export_dir.mkdir(parents=True, exist_ok=True)
    model_path = export_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    data_params = config.get("data_params", {})
    metadata = {
        "schema_version": "1.0",
        "dataset_name": data_params.get("dataset_name"),
        "db_dataset_name": dataset_name,
        "regime": data_params.get("regime"),
        "cycle_id": data_params.get("cycle_id"),
        "model_class": "nianetvae.models.rnn_vae.RNNVAE",
        "mapping_context": _as_jsonable(getattr(model, "mapping_context", {})),
        "solution": _as_jsonable(best_solution),
        "hash_id": str(model.hash_id),
        "n_features": data_params.get("n_features"),
        "seq_len": data_params.get("seq_len"),
        "stride": data_params.get("stride"),
        "rolling_window": data_params.get("rolling_window"),
        "train_minutes": data_params.get("train_minutes"),
        "post_train_minutes": data_params.get("post_train_minutes"),
        "pre_maint_minutes": data_params.get("pre_maint_minutes"),
        "train_phases": data_params.get("train_phases"),
        "test_phases": data_params.get("test_phases"),
        "created_at": datetime.now().isoformat(),
        "run_uuid": RUN_UUID,
        "git_ref": _get_git_ref(),
        "weights_file": "model.pt",
    }
    meta_path = export_dir / "model_meta.json"
    meta_path.write_text(json.dumps(_as_jsonable(metadata), indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "run_uuid": RUN_UUID,
        "git_ref": _get_git_ref(),
        "algorithm": best_algorithm,
        "dataset_name": data_params.get("dataset_name"),
        "db_dataset_name": dataset_name,
        "regime": data_params.get("regime"),
        "cycle_id": data_params.get("cycle_id"),
        "search": search_result,
        "final_training": {
            "started_at": final_result["started_at"],
            "ended_at": final_result["ended_at"],
            "duration_s": final_result["duration_s"],
            "fitness": final_result["fitness"],
            "error": final_result["error"],
            "complexity": final_result["complexity"],
            "metrics": final_result["metrics"],
            "anomaly_metrics": final_result["anomaly_metrics"],
        },
        "artifacts": {
            "weights_file": "model.pt",
            "meta_file": "model_meta.json",
        }
    }
    summary_path = export_dir / "search_summary.json"
    summary_path.write_text(json.dumps(_as_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
    return model_path, meta_path, summary_path


def calculate_fitness(model, experiment, seq_len):
    """
    Calculate fitness from raw SMAPE and model complexity.

    Args:
        model: The model being evaluated.
        experiment: The experiment object containing metrics.
        seq_len: Sequence length of the dataset.

    Returns:
        fitness: The computed combined fitness value (error + complexity).
        error: The computed error term.
        complexity: The computed complexity term.
    """
    # Check if metrics are complete
    if not experiment.metrics.are_metrics_complete():
        Log.error("Some metric values are still None. Fitness function is waiting for metrics data.")
        return int(9e10), int(9e10), int(9e10)

    # Fetch raw metrics.
    try:
        raw_metrics = experiment.metrics.compute()
        Log.debug(f"Raw metrics: {raw_metrics}")
    except Exception as e:
        Log.error(f"Error computing metrics: {e}")
        return int(9e10), int(9e10), int(9e10)

    smape = raw_metrics.get("SMAPE")
    if smape is None:
        Log.error("SMAPE metric is missing in computed metrics; cannot evaluate fitness.")
        return int(9e10), int(9e10), int(9e10)

    try:
        smape_value = float(smape)
    except Exception:
        Log.error(f"SMAPE value is not numeric: {smape}")
        return int(9e10), int(9e10), int(9e10)

    if not math.isfinite(smape_value):
        Log.error(f"Invalid non-finite SMAPE value: {smape_value}")
        return int(9e10), int(9e10), int(9e10)

    # Complexity calculation
    def normalize_complexity(value, max_bound):
        return value / max_bound

    encoding_complexity = normalize_complexity(len(model.encoding_layers), seq_len)
    decoding_complexity = normalize_complexity(len(model.decoding_layers), seq_len)
    bottleneck_complexity = normalize_complexity(model.bottleneck_size, seq_len)

    max_possible_complexity = 3.0  # Sum of all normalized components
    complexity = int(
        round((encoding_complexity + decoding_complexity + bottleneck_complexity) / max_possible_complexity, 6)
        * 1000000
    )

    # Total fitness calculation
    try:
        error = int(round(smape_value, 6) * 1000000)
        fitness = error + complexity
        Log.debug(f"Calculated fitness: {fitness}, error: {error}, complexity: {complexity}")

        if not math.isfinite(float(fitness)) or not math.isfinite(float(error)) or not math.isfinite(float(complexity)):
            Log.error("Invalid fitness, error, or complexity value detected. Returning worst possible value.")
            return int(9e10), int(9e10), int(9e10)

    except Exception as e:
        Log.error(f"Error during fitness calculation: {e}")
        return int(9e10), int(9e10), int(9e10)

    return fitness, error, complexity


class RNNVAEArchitectureMultiObj(Problem):
    """
    This class defines the multiobjective problem for RNN-VAE architecture search.
    The two objectives are:
      1. The error (from validation/test metrics)
      2. The model complexity (based on the number of layers and bottleneck size)
    Both are to be minimized.
    """

    def __init__(self, dimension, config, conn, datamodule, dataset_name):
        self.config = config
        self.conn = conn
        self.datamodule = datamodule
        self.dataset_name = dataset_name
        self.iteration = 0
        self.stats = {"trained": 0, "cached": 0, "invalid": 0, "failed": 0}
        self.best_fitness = None
        self.best_hash = None
        super().__init__(n_var=dimension, n_obj=2, n_constr=0, xl=0, xu=1)

    @staticmethod
    def _to_int(value, default=PENALTY):
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
            if (value is None or value == PENALTY) and isinstance(anomaly_metrics, dict):
                if metric_key in anomaly_metrics:
                    value = anomaly_metrics.get(metric_key)
                elif metric_key.lower() in anomaly_metrics:
                    value = anomaly_metrics.get(metric_key.lower())

            if value is None or value == PENALTY:
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
        ]
        if duration_s is not None:
            parts.append(f"duration_s={float(duration_s):.1f}")
        if reason:
            parts.append(f"reason={reason}")
        if metric_parts:
            parts.extend(metric_parts)
        Log.info(" ".join(parts))

    def _evaluate(self, X, out, *args, **kwargs):
        # X is an array of candidate solutions, shape (n_individuals, dimension)
        F = []
        for solution in X:
            self.iteration += 1
            fitness = PENALTY
            error = PENALTY
            complexity = PENALTY
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
                status = "cached"
                self.stats["cached"] += 1
                error = self._to_int(existing_entry['error'][0])
                complexity = self._to_int(existing_entry['complexity'][0])
                if 'fitness' in existing_entry.columns:
                    fitness = self._to_int(existing_entry['fitness'][0])
                else:
                    fitness = self._to_int(error + complexity)
            else:
                # If the model configuration is invalid, assign worst values.
                if not model.is_valid:
                    status = "invalid"
                    reason = "invalid_architecture"
                    self.stats["invalid"] += 1
                    error = PENALTY
                    complexity = PENALTY
                    fitness = PENALTY
                    self.conn.save_model_and_entry(
                        dataset_name=self.dataset_name,
                        alg_name="NSGA2",
                        iteration=self.iteration,
                        model=model,
                        fitness=PENALTY,
                        solution=solution,
                        error=error,
                        complexity=complexity
                    )
                else:
                    status = "trained"
                    self.stats["trained"] += 1
                    experiment = RNNVAExperiment(model, self.dataset_name, "NSGA2", **self.config)
                    trainer = Trainer(
                        enable_progress_bar=True,
                        accelerator="gpu" if torch.cuda.is_available() else "cpu",
                        devices=1,
                        default_root_dir=path,
                        log_every_n_steps=50,
                        logger=False,
                        enable_checkpointing=False,
                        **self.config['trainer_params']
                    )

                    start_time = datetime.now()
                    trainer.fit(experiment, datamodule=self.datamodule)
                    trainer.test(experiment, datamodule=self.datamodule)
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()

                    fitness, error, complexity = calculate_fitness(
                        model,
                        experiment,
                        self.config['data_params']['seq_len']
                    )
                    metric_parts = self._selected_metrics(experiment)

                    self.conn.save_model_and_entry(
                        dataset_name=self.dataset_name,
                        alg_name="NSGA2",
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

            # Ensure that if any value is nan, we use the worst-case penalty.
            if np.isnan(error) or np.isnan(complexity):
                fitness = PENALTY
                error = PENALTY
                complexity = PENALTY
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
                duration_s=duration,
                reason=reason,
                metric_parts=metric_parts
            )

            F.append([error, complexity])
        out["F"] = np.array(F)


def solve_architecture_problem():
    """
    Uses pymoo's NSGA-II to perform a multiobjective search for the optimal balance between
    error and complexity. Optimizer settings (e.g., population size, termination criteria) are
    taken from the configuration file.
    """
    DIMENSIONALITY = 7

    # Create the multiobjective problem instance.
    # RNNVAEArchitectureMultiObj is assumed to be defined elsewhere.
    problem = RNNVAEArchitectureMultiObj(
        dimension=DIMENSIONALITY,
        config=config,
        conn=conn,
        datamodule=datamodule,
        dataset_name=dataset_name
    )

    # Determine termination criteria.
    # Use time-based termination if a time limit is provided in the config

    # Expected format: "HH:MM:SS" (e.g., "95:00:00")
    time_str = config['nia_search']['time']
    try:
        hours, minutes, seconds = map(int, time_str.split(":"))
        max_time = hours * 3600 + minutes * 60 + seconds
    except Exception as e:
        Log.error(f"Error parsing time limit from config: {time_str}. Ensure it is in HH:MM:SS format.")
        raise e
    termination = get_termination("time", max_time=max_time)

    # Set up the NSGA-II algorithm with the population size from config.
    algorithm = NSGA2(pop_size=config['nia_search']['population_size'])

    # Determine the number of parallel jobs (using CUDA if available).
    n_jobs = torch.cuda.device_count() if torch.cuda.is_available() else 1

    Log.info(
        f"SEARCH_START algorithm=NSGA2 population_size={config['nia_search']['population_size']} "
        f"time_limit={time_str} time_limit_seconds={max_time} n_jobs={n_jobs} "
        f"metrics={config['nia_search'].get('metrics')}"
    )
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=config['exp_params']['manual_seed'],
        verbose=True,
        n_jobs=n_jobs
    )

    # Retrieve the best solution from the database (using your existing criteria).
    best_solution, best_algorithm = conn.best_results(dataset_name)
    if best_solution is None:
        Log.error(
            f"SEARCH_DONE iterations={problem.iteration} trained={problem.stats['trained']} "
            f"cached={problem.stats['cached']} invalid={problem.stats['invalid']} failed={problem.stats['failed']} "
            "best_hash=None best_fitness=None no_solution=true"
        )
        return

    search_result = {
        "iterations": problem.iteration,
        "trained": problem.stats["trained"],
        "cached": problem.stats["cached"],
        "invalid": problem.stats["invalid"],
        "failed": problem.stats["failed"],
        "best_hash": problem.best_hash,
        "best_fitness": problem.best_fitness,
        "best_solution": _as_jsonable(best_solution),
        "time_limit": time_str,
        "time_limit_seconds": max_time,
    }

    final_result = _run_final_training(best_solution)
    best_model = final_result["model"]
    model_file = (
        Path(config['logging_params']['save_dir'])
        / f"{dataset_name}_NSGA2_{best_model.hash_id}.pt"
    )
    torch.save(best_model.state_dict(), model_file)
    Log.info(
        f"SEARCH_DONE iterations={problem.iteration} trained={problem.stats['trained']} "
        f"cached={problem.stats['cached']} invalid={problem.stats['invalid']} failed={problem.stats['failed']} "
        f"best_hash={best_model.hash_id} best_fitness={problem.best_fitness}"
    )
    Log.info(f"BEST_MODEL_SAVED path={model_file}")

    export_enabled = bool(config.get("logging_params", {}).get("export_enabled", False))
    if export_enabled:
        export_dir = _resolve_export_dir(config)
        model_path, meta_path, summary_path = _export_cycle_artifacts(
            export_dir=export_dir,
            model=best_model,
            best_solution=best_solution,
            best_algorithm=best_algorithm,
            search_result=search_result,
            final_result=final_result,
        )
        Log.info(
            f"MODEL_EXPORT_READY dir={export_dir} "
            f"weights={model_path.name} meta={meta_path.name} summary={summary_path.name}"
        )
