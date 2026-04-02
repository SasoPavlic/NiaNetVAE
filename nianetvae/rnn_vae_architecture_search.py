import json
import gc
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch import Trainer
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

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


def _cleanup_candidate_runtime(trainer=None, experiment=None, model=None):
    for obj in (trainer, experiment, model):
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _short_exception_reason(exc: Exception) -> str:
    text = str(exc).strip().replace("\n", " ")
    text = " ".join(text.split())
    if len(text) > 180:
        text = text[:177] + "..."
    return f"{exc.__class__.__name__}:{text}" if text else exc.__class__.__name__


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
    objective_bundle = calculate_objective_bundle(
        model,
        metrics_payload=final_metrics,
        anomaly_metrics=anomaly_metrics,
        seq_len=config['data_params']['seq_len'],
        n_features=config['data_params']['n_features'],
        cfg=config,
    )
    return {
        "model": model,
        "experiment": experiment,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_s": duration_s,
        "fitness": objective_bundle["fitness"],
        "error": objective_bundle["obj_error"],
        "complexity": objective_bundle["obj_efficiency"],
        "obj_pdm": objective_bundle["obj_pdm"],
        "pdm_signal_quality": objective_bundle["pdm_signal_quality"],
        "objective_reason": objective_bundle.get("reason"),
        "objective_contract": objective_bundle.get("objective_contract"),
        "metrics": final_metrics,
        "anomaly_metrics": anomaly_metrics,
    }


def _run_final_training(best_solution):
    seed_everything(config['exp_params']['manual_seed'], True)
    model = RNNVAE(best_solution, **config)
    if not model.is_valid:
        raise ValueError("Best solution produced an invalid model during final training.")
    return _run_training_with_model(model, "NSGA3")


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


def _resolve_warm_start_sampling(dimensionality: int, effective_population: int):
    nia_search = config.get("nia_search") or {}
    warm_cfg = nia_search.get("warm_start") or {}
    pop_size = int(effective_population)
    data_params = config.get("data_params") or {}
    regime = str(data_params.get("regime", "")).strip().lower()
    cycle_id = data_params.get("cycle_id")
    base_seed = int((config.get("exp_params") or {}).get("manual_seed", 42))

    result = {
        "enabled": False,
        "sampling": None,
        "init_mode": "random",
        "source_cycle_id": None,
        "reason": None,
        "details": {
            "enabled": False,
            "population_size": pop_size,
            "carry_over_count": 0,
            "perturb_count": 0,
            "random_count": pop_size,
            "perturbation_strength": None,
        },
    }

    if not bool(warm_cfg.get("enabled", False)):
        result["reason"] = "warm_start_disabled"
        return result

    if regime != "per_maint":
        result["reason"] = f"unsupported_regime:{regime or 'none'}"
        return result

    if cycle_id is None:
        result["reason"] = "missing_cycle_id"
        return result

    cycle_id = int(cycle_id)
    if cycle_id <= 0:
        result["reason"] = f"cycle_{cycle_id:02d}_random_init"
        return result

    previous_source = _find_latest_trained_cycle_artifacts_before(cycle_id)
    if previous_source is None:
        result["reason"] = f"no_previous_trained_cycle_before_{cycle_id:02d}"
        return result

    source_cycle_id, _, _, previous_meta = previous_source
    previous_metadata = json.loads(previous_meta.read_text(encoding="utf-8"))
    anchor_solution = previous_metadata.get("solution")
    if anchor_solution is None:
        result["reason"] = f"missing_solution_in_{previous_meta.name}"
        return result

    try:
        anchor = np.asarray(anchor_solution, dtype=float).reshape(-1)
    except Exception:
        result["reason"] = "invalid_anchor_solution_format"
        return result

    if anchor.size != dimensionality:
        result["reason"] = f"invalid_anchor_dim:{anchor.size}"
        return result

    anchor = np.clip(anchor, 0.0, 1.0)

    carry_ratio = float(warm_cfg.get("carry_over_ratio", 0.10))
    perturb_ratio = float(warm_cfg.get("perturb_ratio", 0.40))
    perturbation_strength = float(warm_cfg.get("perturbation_strength", 0.08))

    if carry_ratio < 0 or perturb_ratio < 0:
        raise ValueError(
            "Invalid warm_start ratios: carry_over_ratio and perturb_ratio must be >= 0."
        )
    if perturbation_strength < 0:
        raise ValueError(
            "Invalid warm_start perturbation_strength: must be >= 0."
        )

    carry_count = int(round(pop_size * carry_ratio))
    perturb_count = int(round(pop_size * perturb_ratio))
    if carry_ratio > 0 and carry_count == 0 and pop_size > 0:
        carry_count = 1
    if perturb_ratio > 0 and perturb_count == 0 and pop_size - carry_count > 0:
        perturb_count = 1

    if carry_count + perturb_count > pop_size:
        overflow = (carry_count + perturb_count) - pop_size
        reduce_perturb = min(perturb_count, overflow)
        perturb_count -= reduce_perturb
        overflow -= reduce_perturb
        if overflow > 0:
            carry_count = max(0, carry_count - overflow)

    random_count = pop_size - carry_count - perturb_count
    rng = np.random.default_rng(base_seed + cycle_id)

    parts = []
    if carry_count > 0:
        parts.append(np.tile(anchor, (carry_count, 1)))
    if perturb_count > 0:
        noise = rng.uniform(-perturbation_strength, perturbation_strength, size=(perturb_count, dimensionality))
        parts.append(np.clip(anchor + noise, 0.0, 1.0))
    if random_count > 0:
        parts.append(rng.uniform(0.0, 1.0, size=(random_count, dimensionality)))

    if not parts:
        result["reason"] = "empty_population_after_warm_start_counts"
        return result

    sampling = np.vstack(parts).astype(float, copy=False)
    rng.shuffle(sampling, axis=0)

    result.update({
        "enabled": True,
        "sampling": sampling,
        "init_mode": "warm_start",
        "source_cycle_id": int(source_cycle_id),
        "reason": None,
        "details": {
            "enabled": True,
            "source_cycle_id": int(source_cycle_id),
            "population_size": pop_size,
            "carry_over_count": int(carry_count),
            "perturb_count": int(perturb_count),
            "random_count": int(random_count),
            "perturbation_strength": float(perturbation_strength),
            "base_seed": int(base_seed),
        },
    })
    return result


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
    workflow_mode = str((config.get("workflow") or {}).get("mode", "")).strip().lower() or None
    seed_source = (config.get("exp_params") or {}).get("manual_seed")
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
        "provenance": {
            "experiment_mode": workflow_mode,
            "source_cycle": None,
            "seed_source": seed_source,
        },
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


def run_per_maint_warmstart_cycle():
    data_params = config.get("data_params", {})
    cycle_id = data_params.get("cycle_id")
    if cycle_id is None:
        raise ValueError("per_maint_warmstart_search requires data_params.cycle_id.")
    cycle_id = int(cycle_id)

    nia_search_cfg = dict(config.get("nia_search") or {})
    warm_cfg = dict(nia_search_cfg.get("warm_start") or {})
    if not bool(warm_cfg.get("enabled", False)):
        Log.warning(
            "WARMSTART_MODE warm_start.enabled=false in config; "
            "enabling warm-start automatically for this run."
        )
    warm_cfg["enabled"] = True
    nia_search_cfg["warm_start"] = warm_cfg
    config["nia_search"] = nia_search_cfg

    if cycle_id == 0:
        Log.info(
            "WARMSTART_MODE cycle_id=00 uses random initialization by design."
        )
    else:
        Log.info(
            f"WARMSTART_MODE cycle_id={cycle_id:02d} "
            "uses warm-start search with previous-cycle anchor when available."
        )
    solve_architecture_problem()


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
    workflow_mode = str((config.get("workflow") or {}).get("mode", "")).strip().lower() or None
    seed_source = (config.get("exp_params") or {}).get("manual_seed")
    source_cycle = search_result.get("source_cycle_id")
    search_init_mode = search_result.get("init_mode")
    warm_start_payload = search_result.get("warm_start")
    provenance = {
        "experiment_mode": workflow_mode,
        "source_cycle": source_cycle,
        "seed_source": seed_source,
        "search_init_mode": search_init_mode,
    }
    if isinstance(warm_start_payload, dict):
        provenance["warm_start"] = _as_jsonable(warm_start_payload)
    metadata = {
        "schema_version": "1.0",
        "dataset_name": data_params.get("dataset_name"),
        "db_dataset_name": dataset_name,
        "regime": data_params.get("regime"),
        "cycle_id": data_params.get("cycle_id"),
        "workflow_mode": workflow_mode,
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
        "provenance": provenance,
    }
    meta_path = export_dir / "model_meta.json"
    meta_path.write_text(json.dumps(_as_jsonable(metadata), indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "run_uuid": RUN_UUID,
        "git_ref": _get_git_ref(),
        "algorithm": best_algorithm,
        "workflow_mode": workflow_mode,
        "dataset_name": data_params.get("dataset_name"),
        "db_dataset_name": dataset_name,
        "regime": data_params.get("regime"),
        "cycle_id": data_params.get("cycle_id"),
        "provenance": provenance,
        "winner_selection": search_result.get("winner_selection"),
        "search": search_result,
        "final_training": {
            "started_at": final_result["started_at"],
            "ended_at": final_result["ended_at"],
            "duration_s": final_result["duration_s"],
            "fitness": final_result["fitness"],
            "error": final_result["error"],
            "complexity": final_result["complexity"],
            "obj_pdm": final_result.get("obj_pdm"),
            "pdm_signal_quality": final_result.get("pdm_signal_quality"),
            "objective_reason": final_result.get("objective_reason"),
            "objective_contract": final_result.get("objective_contract"),
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


def _resolve_objective_contract(cfg: dict | None = None) -> dict:
    cfg = cfg or config or {}
    objectives = dict(cfg.get("objectives") or {})
    error_metric = str(((objectives.get("error") or {}).get("metric") or "SMAPE")).strip().upper()
    efficiency_metric = str(((objectives.get("efficiency") or {}).get("metric") or "macs")).strip().lower()
    pdm_metric = str(((objectives.get("pdm") or {}).get("metric") or "auprc_premaint")).strip().lower()
    return {
        "error_metric": error_metric,
        "efficiency_metric": efficiency_metric,
        "pdm_metric": pdm_metric,
    }


def _resolve_winner_selection_contract(cfg: dict | None = None) -> dict:
    cfg = cfg or config or {}
    objectives = dict(cfg.get("objectives") or {})
    selection_cfg = dict(objectives.get("selection") or {})

    method = str(selection_cfg.get("method", "weighted_ideal_distance")).strip().lower()
    if method != "weighted_ideal_distance":
        raise ValueError(
            f"Unsupported objectives.selection.method={method!r}. "
            "Allowed value: weighted_ideal_distance."
        )

    default_weights = {"error": 0.30, "efficiency": 0.20, "pdm": 0.50}
    raw_weights_cfg = dict(selection_cfg.get("weights") or {})
    resolved_weights = {}
    for key in ("error", "efficiency", "pdm"):
        raw_value = raw_weights_cfg.get(key, default_weights[key])
        try:
            value = float(raw_value)
        except Exception:
            raise ValueError(
                f"Invalid objectives.selection.weights.{key}={raw_value!r}. Expected finite float >= 0."
            ) from None
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"Invalid objectives.selection.weights.{key}={raw_value!r}. Expected finite float >= 0."
            )
        resolved_weights[key] = value

    total = resolved_weights["error"] + resolved_weights["efficiency"] + resolved_weights["pdm"]
    if total <= 0:
        raise ValueError(
            "Invalid objectives.selection.weights: sum(error, efficiency, pdm) must be > 0."
        )
    weights_normalized = {
        "error": resolved_weights["error"] / total,
        "efficiency": resolved_weights["efficiency"] / total,
        "pdm": resolved_weights["pdm"] / total,
    }
    return {
        "method": method,
        "weights": resolved_weights,
        "weights_normalized": weights_normalized,
    }


def _parse_solution_array(raw_value):
    if raw_value is None:
        return None
    parsed = None
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except Exception:
            return None
    else:
        parsed = raw_value

    try:
        arr = np.asarray(parsed, dtype=float)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _parse_timestamp_sort_key(raw_value):
    if raw_value is None:
        return float("inf")
    if isinstance(raw_value, datetime):
        dt = raw_value
    else:
        text = str(raw_value).strip()
        if not text:
            return float("inf")
        dt = None
        try:
            dt = datetime.fromisoformat(text)
        except Exception:
            dt = None
        if dt is None:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except Exception:
                    dt = None
    if dt is None:
        return float("inf")
    try:
        return float(dt.timestamp())
    except Exception:
        return float("inf")


def _pareto_mask_minimize(objectives: np.ndarray) -> np.ndarray:
    count = int(objectives.shape[0])
    keep = np.ones(count, dtype=bool)
    for i in range(count):
        if not keep[i]:
            continue
        for j in range(count):
            if i == j:
                continue
            dominates = np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i])
            if dominates:
                keep[i] = False
                break
    return keep


def _select_deterministic_pareto_winner(candidates_df, selection_contract: dict):
    if candidates_df is None:
        raise ValueError("Winner selection failed: no candidate rows were returned from DB.")

    try:
        records = candidates_df.to_dict(orient="records")
    except Exception:
        records = []
    candidate_count = len(records)
    if candidate_count == 0:
        raise ValueError(
            f"Winner selection failed for {dataset_name}: no DB candidates found in cycle-scoped pool."
        )

    valid = []
    for row in records:
        obj_error = _safe_float(row.get("error"))
        obj_efficiency = _safe_float(row.get("complexity"))
        pr_auc_mean = _safe_float(row.get("pr_auc_mean"))
        obj_pdm = _safe_float(1.0 - pr_auc_mean) if pr_auc_mean is not None else None
        solution = _parse_solution_array(row.get("solution_array"))
        if obj_error is None or obj_efficiency is None or obj_pdm is None or solution is None:
            continue
        if obj_error >= PENALTY or obj_efficiency >= PENALTY or obj_pdm >= PENALTY:
            continue
        valid.append(
            {
                "id": int(row.get("id")) if row.get("id") is not None else int(9e18),
                "hash_id": str(row.get("hash_id", "")),
                "algorithm_name": str(row.get("algorithm_name", "NSGA3")),
                "timestamp_sort_key": _parse_timestamp_sort_key(row.get("timestamp")),
                "obj_error": float(obj_error),
                "obj_efficiency": float(obj_efficiency),
                "obj_pdm": float(obj_pdm),
                "pdm_signal_quality": float(pr_auc_mean),
                "solution": solution,
            }
        )

    if not valid:
        raise ValueError(
            f"Winner selection failed for {dataset_name}: no valid objective candidates after filtering."
        )

    def _tie_break_key(item):
        return (
            float(item["obj_pdm"]),
            float(item["obj_error"]),
            float(item["obj_efficiency"]),
            float(item["timestamp_sort_key"]),
            int(item["id"]),
        )

    dedup = {}
    for item in valid:
        key = item["hash_id"]
        if key not in dedup or _tie_break_key(item) < _tie_break_key(dedup[key]):
            dedup[key] = item
    deduped = list(dedup.values())
    if not deduped:
        raise ValueError(
            f"Winner selection failed for {dataset_name}: no candidates after hash de-duplication."
        )

    objective_matrix = np.array(
        [[row["obj_error"], row["obj_efficiency"], row["obj_pdm"]] for row in deduped],
        dtype=float,
    )
    mask = _pareto_mask_minimize(objective_matrix)
    pareto_rows = [deduped[i] for i in range(len(deduped)) if bool(mask[i])]
    if not pareto_rows:
        raise ValueError(
            f"Winner selection failed for {dataset_name}: Pareto set is empty after filtering."
        )

    pareto_matrix = np.array(
        [[row["obj_error"], row["obj_efficiency"], row["obj_pdm"]] for row in pareto_rows],
        dtype=float,
    )
    mins = np.min(pareto_matrix, axis=0)
    maxs = np.max(pareto_matrix, axis=0)
    spans = maxs - mins
    normalized = np.zeros_like(pareto_matrix, dtype=float)
    positive_span = spans > 0
    if np.any(positive_span):
        normalized[:, positive_span] = (
            (pareto_matrix[:, positive_span] - mins[positive_span]) / spans[positive_span]
        )

    weights = selection_contract["weights_normalized"]
    distances = np.sqrt(
        normalized[:, 0] ** 2 * float(weights["error"])
        + normalized[:, 1] ** 2 * float(weights["efficiency"])
        + normalized[:, 2] ** 2 * float(weights["pdm"])
    )
    best_distance = float(np.min(distances))
    tie_indices = [i for i, d in enumerate(distances) if abs(float(d) - best_distance) <= 1e-12]
    tied_rows = [pareto_rows[i] for i in tie_indices]
    tied_rows.sort(key=_tie_break_key)
    selected = tied_rows[0]

    return {
        "method": selection_contract["method"],
        "weights": _as_jsonable(selection_contract["weights"]),
        "weights_normalized": _as_jsonable(selection_contract["weights_normalized"]),
        "candidate_count": int(candidate_count),
        "valid_candidate_count": int(len(valid)),
        "deduplicated_candidate_count": int(len(deduped)),
        "pareto_candidate_count": int(len(pareto_rows)),
        "selected_hash": selected["hash_id"],
        "selected_id": int(selected["id"]),
        "selected_algorithm": selected["algorithm_name"],
        "selected_objectives": {
            "obj_error": float(selected["obj_error"]),
            "obj_efficiency": float(selected["obj_efficiency"]),
            "obj_pdm": float(selected["obj_pdm"]),
        },
        "selected_pdm_signal_quality": float(selected["pdm_signal_quality"]),
        "selected_distance": float(best_distance),
        "selected_solution": _as_jsonable(selected["solution"]),
    }


def _safe_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _penalty_objective_bundle(reason: str, objective_contract: dict | None = None) -> dict:
    return {
        "valid": False,
        "reason": reason,
        "objective_contract": objective_contract or _resolve_objective_contract(),
        "obj_error": float(PENALTY),
        "obj_efficiency": float(PENALTY),
        "obj_pdm": float(PENALTY),
        "pdm_signal_quality": None,
        "fitness": float(PENALTY),
    }


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def _model_forward(model, signal_batch: torch.Tensor):
    try:
        return model({"signal": signal_batch})
    except Exception:
        return model(signal_batch)


def _estimate_model_macs(model, seq_len: int, n_features: int) -> tuple[float | None, str | None]:
    was_training = bool(model.training) if hasattr(model, "training") else False
    device = _model_device(model)
    dummy_signal = torch.zeros((1, int(seq_len), int(n_features)), dtype=torch.float32, device=device)
    try:
        if hasattr(model, "eval"):
            model.eval()
        try:
            from thop import profile as thop_profile

            macs, _ = thop_profile(model, inputs=({"signal": dummy_signal},), verbose=False)
            macs_value = _safe_float(macs)
            if macs_value is not None and macs_value > 0:
                return macs_value, None
        except Exception:
            pass

        if not hasattr(torch, "profiler") or not hasattr(torch.profiler, "profile"):
            return None, "macs_profiler_unavailable"
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            torch.cuda.synchronize()
        with torch.inference_mode():
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                _model_forward(model, dummy_signal)
        if device.type == "cuda":
            torch.cuda.synchronize()

        total_flops = 0.0
        for event in prof.key_averages():
            flops_value = _safe_float(getattr(event, "flops", None))
            if flops_value is not None and flops_value > 0:
                total_flops += flops_value
        if total_flops <= 0:
            return None, "macs_flops_not_reported"
        return float(total_flops / 2.0), None
    except Exception as exc:
        return None, f"macs_estimation_failed:{exc.__class__.__name__}"
    finally:
        try:
            if hasattr(model, "train"):
                model.train(was_training)
        except Exception:
            pass


def _estimate_model_latency_ms(
    model,
    seq_len: int,
    n_features: int,
    warmup_steps: int = 3,
    measure_steps: int = 7,
) -> tuple[float | None, str | None]:
    was_training = bool(model.training) if hasattr(model, "training") else False
    device = _model_device(model)
    dummy_signal = torch.zeros((1, int(seq_len), int(n_features)), dtype=torch.float32, device=device)
    try:
        if hasattr(model, "eval"):
            model.eval()
        with torch.inference_mode():
            for _ in range(max(1, int(warmup_steps))):
                _model_forward(model, dummy_signal)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            durations_ms = []
            for _ in range(max(1, int(measure_steps))):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                _model_forward(model, dummy_signal)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ended = time.perf_counter()
                durations_ms.append((ended - started) * 1000.0)

        if not durations_ms:
            return None, "latency_no_samples"
        latency_ms = _safe_float(float(np.median(np.asarray(durations_ms, dtype=np.float64))))
        if latency_ms is None or latency_ms <= 0:
            return None, "latency_non_finite"
        return latency_ms, None
    except Exception as exc:
        return None, f"latency_estimation_failed:{exc.__class__.__name__}"
    finally:
        try:
            if hasattr(model, "train"):
                model.train(was_training)
        except Exception:
            pass


def _compute_efficiency_objective(model, metric_name: str, seq_len: int, n_features: int) -> tuple[float | None, str | None]:
    metric = str(metric_name).strip().lower()
    if metric == "params":
        try:
            value = float(sum(int(p.numel()) for p in model.parameters()))
        except Exception as exc:
            return None, f"params_count_failed:{exc.__class__.__name__}"
        if value <= 0 or not math.isfinite(value):
            return None, "params_non_finite"
        return value, None

    if metric == "macs":
        return _estimate_model_macs(model, seq_len=seq_len, n_features=n_features)

    if metric == "latency_ms":
        return _estimate_model_latency_ms(model, seq_len=seq_len, n_features=n_features)

    return None, f"unsupported_efficiency_metric:{metric}"


def _metrics_payload_from_cached_entry(entry: dict | None) -> dict:
    entry = entry or {}
    metric_keys = ("MAE", "MSE", "RMSE", "MAPE", "RMAPE", "SMAPE")
    return {metric_key: entry.get(metric_key) for metric_key in metric_keys}


def _anomaly_payload_from_cached_entry(entry: dict | None) -> dict:
    entry = entry or {}
    keys = (
        "precision",
        "recall",
        "f1_score",
        "pr_auc_mean",
        "pr_auc_std",
        "roc_auc_mean",
        "roc_auc_std",
    )
    return {key: entry.get(key) for key in keys}


def calculate_objective_bundle(
    model,
    metrics_payload: dict | None,
    anomaly_metrics: dict | None,
    seq_len: int,
    n_features: int,
    cfg: dict | None = None,
) -> dict:
    objective_contract = _resolve_objective_contract(cfg)
    metrics_payload = metrics_payload or {}
    anomaly_metrics = anomaly_metrics or {}

    error_metric = objective_contract["error_metric"]
    obj_error = _safe_float(metrics_payload.get(error_metric))
    if obj_error is None:
        return _penalty_objective_bundle(
            reason=f"missing_or_invalid_error_metric:{error_metric}",
            objective_contract=objective_contract,
        )

    obj_efficiency, eff_reason = _compute_efficiency_objective(
        model=model,
        metric_name=objective_contract["efficiency_metric"],
        seq_len=int(seq_len),
        n_features=int(n_features),
    )
    if obj_efficiency is None:
        Log.warning(
            "OBJECTIVE_EFFICIENCY_FALLBACK "
            f"metric={objective_contract['efficiency_metric']} "
            f"reason={eff_reason or 'invalid_efficiency_objective'} penalty=true"
        )
        return _penalty_objective_bundle(
            reason=eff_reason or "invalid_efficiency_objective",
            objective_contract=objective_contract,
        )

    pdm_signal_quality = _safe_float(anomaly_metrics.get("pr_auc_mean"))
    if pdm_signal_quality is None:
        Log.warning(
            "OBJECTIVE_PDM_FALLBACK "
            "metric=auprc_premaint reason=missing_or_invalid_pdm_signal_quality penalty=true"
        )
        return _penalty_objective_bundle(
            reason="missing_or_invalid_pdm_signal_quality",
            objective_contract=objective_contract,
        )

    obj_pdm = _safe_float(1.0 - pdm_signal_quality)
    if obj_pdm is None:
        return _penalty_objective_bundle(
            reason="invalid_obj_pdm",
            objective_contract=objective_contract,
        )

    fitness = _safe_float(obj_error + obj_efficiency)
    if fitness is None:
        return _penalty_objective_bundle(
            reason="invalid_compatibility_fitness",
            objective_contract=objective_contract,
        )

    return {
        "valid": True,
        "reason": None,
        "objective_contract": objective_contract,
        "obj_error": float(obj_error),
        "obj_efficiency": float(obj_efficiency),
        "obj_pdm": float(obj_pdm),
        "pdm_signal_quality": float(pdm_signal_quality),
        # Compatibility fields retained until C13.5.
        "fitness": float(fitness),
    }


def calculate_objective_bundle_from_experiment(model, experiment, seq_len: int, n_features: int, cfg: dict | None = None):
    if experiment is None or getattr(experiment, "metrics", None) is None:
        return _penalty_objective_bundle("missing_experiment_metrics", objective_contract=_resolve_objective_contract(cfg))

    try:
        if not experiment.metrics.are_metrics_complete():
            return _penalty_objective_bundle(
                reason="incomplete_metrics",
                objective_contract=_resolve_objective_contract(cfg),
            )
        metrics_payload = experiment.metrics.compute()
    except Exception as exc:
        return _penalty_objective_bundle(
            reason=f"metrics_compute_failed:{exc.__class__.__name__}",
            objective_contract=_resolve_objective_contract(cfg),
        )

    anomaly_metrics = getattr(experiment, "anomaly_metrics", {}) or {}
    return calculate_objective_bundle(
        model=model,
        metrics_payload=metrics_payload,
        anomaly_metrics=anomaly_metrics,
        seq_len=seq_len,
        n_features=n_features,
        cfg=cfg,
    )


def calculate_objective_bundle_from_cached_row(model, cached_row, seq_len: int, n_features: int, cfg: dict | None = None):
    return calculate_objective_bundle(
        model=model,
        metrics_payload=_metrics_payload_from_cached_entry(cached_row),
        anomaly_metrics=_anomaly_payload_from_cached_entry(cached_row),
        seq_len=seq_len,
        n_features=n_features,
        cfg=cfg,
    )


def calculate_fitness(model, experiment, seq_len, cfg: dict | None = None):
    """
    Compatibility wrapper retained for DB/export flow until C13.5.
    Returns scalar compatibility fitness plus the first two objectives.
    """
    n_features = int(((cfg or config or {}).get("data_params") or {}).get("n_features", 1))
    bundle = calculate_objective_bundle_from_experiment(
        model=model,
        experiment=experiment,
        seq_len=int(seq_len),
        n_features=n_features,
        cfg=cfg or config,
    )
    return bundle["fitness"], bundle["obj_error"], bundle["obj_efficiency"]


class RNNVAEArchitectureMultiObj(Problem):
    """
    This class defines the multiobjective problem for RNN-VAE architecture search.
    The three objectives are:
      1. The error (from validation/test metrics)
      2. The efficiency objective (params|macs|latency_ms)
      3. The PdM objective (1 - AUPRC_premaint)
    All are minimized.
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
        super().__init__(n_var=dimension, n_obj=3, n_constr=0, xl=0, xu=1)

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
        # X is an array of candidate solutions, shape (n_individuals, dimension)
        F = []
        for solution in X:
            self.iteration += 1
            fitness = PENALTY
            error = PENALTY
            complexity = PENALTY
            obj_pdm = PENALTY
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
                )
                if cached_bundle.get("valid"):
                    status = "cached"
                    reason = None
                    self.stats["cached"] += 1
                    error = cached_bundle["obj_error"]
                    complexity = cached_bundle["obj_efficiency"]
                    obj_pdm = cached_bundle["obj_pdm"]
                    pdm_signal_quality = cached_bundle.get("pdm_signal_quality")
                    # Compatibility scalar remains in use until C13.5.
                    fitness = cached_bundle["fitness"]
                else:
                    # Cached entry does not satisfy the current objective contract; re-evaluate.
                    reason = f"cached_objective_miss:{cached_bundle.get('reason') or 'invalid'}"
                    existing_entry = existing_entry.iloc[0:0]

            if existing_entry.shape[0] == 0:
                # If the model configuration is invalid, assign worst values.
                if not model.is_valid:
                    status = "invalid"
                    reason = reason or "invalid_architecture"
                    self.stats["invalid"] += 1
                    error = PENALTY
                    complexity = PENALTY
                    obj_pdm = PENALTY
                    fitness = PENALTY
                    self.conn.save_model_and_entry(
                        dataset_name=self.dataset_name,
                        alg_name="NSGA3",
                        iteration=self.iteration,
                        model=model,
                        fitness=PENALTY,
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
                        )
                        error = objective_bundle["obj_error"]
                        complexity = objective_bundle["obj_efficiency"]
                        obj_pdm = objective_bundle["obj_pdm"]
                        pdm_signal_quality = objective_bundle.get("pdm_signal_quality")
                        # Compatibility scalar remains in use until C13.5.
                        fitness = objective_bundle["fitness"]
                        metric_parts = self._selected_metrics(experiment)

                        if not objective_bundle.get("valid"):
                            status = "failed"
                            reason = objective_bundle.get("reason") or "objective_bundle_invalid"
                            self.stats["failed"] += 1
                        elif fitness >= PENALTY or error >= PENALTY or complexity >= PENALTY or obj_pdm >= PENALTY:
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
                        fitness = PENALTY
                        error = PENALTY
                        complexity = PENALTY
                        obj_pdm = PENALTY
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

            # Ensure that if any value is nan, we use the worst-case penalty.
            if np.isnan(error) or np.isnan(complexity) or np.isnan(obj_pdm):
                fitness = PENALTY
                error = PENALTY
                complexity = PENALTY
                obj_pdm = PENALTY
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


def solve_architecture_problem():
    """
    Uses pymoo's NSGA-III to perform a three-objective search:
      1) obj_error
      2) obj_efficiency
      3) obj_pdm
    Objective contract and optimizer settings are taken from the configuration file.
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

    nsga3_cfg = dict((config.get("nia_search") or {}).get("nsga3") or {})
    n_partitions = int(nsga3_cfg["n_partitions"])
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
    effective_population = int(ref_dirs.shape[0])

    warm_start = _resolve_warm_start_sampling(
        DIMENSIONALITY,
        effective_population=effective_population,
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
            f"cycle_id={int((config.get('data_params') or {}).get('cycle_id', -1)):02d} "
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

    # Determine the number of parallel jobs (using CUDA if available).
    n_jobs = torch.cuda.device_count() if torch.cuda.is_available() else 1
    objective_contract = _resolve_objective_contract(config)
    selection_contract = _resolve_winner_selection_contract(config)

    Log.info(
        f"SEARCH_START algorithm=NSGA3 n_partitions={n_partitions} "
        f"ref_dirs={effective_population} effective_population={effective_population} "
        f"time_limit={time_str} time_limit_seconds={max_time} n_jobs={n_jobs} "
        f"metrics={config['nia_search'].get('metrics')} "
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
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=config['exp_params']['manual_seed'],
        verbose=True,
        n_jobs=n_jobs
    )

    candidate_rows = conn.get_cycle_candidates(dataset_name=dataset_name, algorithm_name="NSGA3")
    winner_selection = _select_deterministic_pareto_winner(
        candidates_df=candidate_rows,
        selection_contract=selection_contract,
    )
    best_solution = np.asarray(winner_selection["selected_solution"], dtype=float)
    best_algorithm = winner_selection["selected_algorithm"]

    Log.info(
        "WINNER_SELECTION "
        f"dataset={dataset_name} method={winner_selection['method']} "
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

    final_result = _run_final_training(best_solution)
    best_model = final_result["model"]
    model_file = (
        Path(config['logging_params']['save_dir'])
        / f"{dataset_name}_NSGA3_{best_model.hash_id}.pt"
    )
    torch.save(best_model.state_dict(), model_file)
    Log.info(
        f"SEARCH_DONE iterations={problem.iteration} trained={problem.stats['trained']} "
        f"cached={problem.stats['cached']} invalid={problem.stats['invalid']} failed={problem.stats['failed']} "
        f"best_hash={best_model.hash_id} best_fitness={winner_selection['selected_distance']}"
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
