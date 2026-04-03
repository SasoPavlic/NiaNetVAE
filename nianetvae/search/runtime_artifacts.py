import gc
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything

from log import Log
from nianetvae.experiments.rnn_vae_experiment import RNNVAExperiment
from nianetvae.models.rnn_vae import RNNVAE


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


def _resolve_export_dir(cfg: dict, run_uuid: str | None = None) -> Path:
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

    run_label = run_uuid or datetime.now().strftime("%Y%m%d%H%M%S")
    return Path(export_root) / dataset / f"run_{run_label}"


def _build_final_trainer(
    config: dict,
    default_root_dir: str,
    trainer_params_override: dict | None = None,
):
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
    *,
    config: dict,
    dataset_name: str,
    datamodule,
    penalty: int | float,
    learning_rate: float | None = None,
    trainer_params_override: dict | None = None,
):
    from .objective_engine import calculate_objective_bundle

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
        config=config,
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
        penalty=penalty,
    )
    return {
        "model": model,
        "experiment": experiment,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_s": duration_s,
        "obj_error": objective_bundle["obj_error"],
        "obj_efficiency": objective_bundle["obj_efficiency"],
        "obj_pdm": objective_bundle["obj_pdm"],
        "pdm_signal_quality": objective_bundle["pdm_signal_quality"],
        "objective_reason": objective_bundle.get("reason"),
        "objective_contract": objective_bundle.get("objective_contract"),
        "metrics": final_metrics,
        "anomaly_metrics": anomaly_metrics,
    }


def _run_final_training(
    best_solution,
    *,
    config: dict,
    dataset_name: str,
    datamodule,
    penalty: int | float,
):
    seed_everything(config['exp_params']['manual_seed'], True)
    model = RNNVAE(best_solution, **config)
    if not model.is_valid:
        raise ValueError("Best solution produced an invalid model during final training.")
    return _run_training_with_model(
        model,
        "NSGA3",
        config=config,
        dataset_name=dataset_name,
        datamodule=datamodule,
        penalty=penalty,
    )


def _export_cycle_artifacts(
        export_dir: Path,
        model: RNNVAE,
        best_solution,
        best_algorithm,
        search_result: dict,
        final_result: dict,
        *,
        config: dict,
        dataset_name: str,
        run_uuid: str,
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
    winner_selection = search_result.get("winner_selection")
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
        "run_uuid": run_uuid,
        "git_ref": _get_git_ref(),
        "weights_file": "model.pt",
        "provenance": provenance,
        "winner_selection": {
            "method": winner_selection.get("method"),
            "weights_normalized": winner_selection.get("weights_normalized"),
            "selected_hash": winner_selection.get("selected_hash"),
            "selected_objectives": winner_selection.get("selected_objectives"),
            "selected_distance": winner_selection.get("selected_distance"),
        } if isinstance(winner_selection, dict) else None,
    }
    meta_path = export_dir / "model_meta.json"
    meta_path.write_text(json.dumps(_as_jsonable(metadata), indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "run_uuid": run_uuid,
        "git_ref": _get_git_ref(),
        "algorithm": best_algorithm,
        "workflow_mode": workflow_mode,
        "dataset_name": data_params.get("dataset_name"),
        "db_dataset_name": dataset_name,
        "regime": data_params.get("regime"),
        "cycle_id": data_params.get("cycle_id"),
        "provenance": provenance,
        "winner_selection": winner_selection,
        "search": search_result,
        "final_training": {
            "started_at": final_result["started_at"],
            "ended_at": final_result["ended_at"],
            "duration_s": final_result["duration_s"],
            "obj_error": final_result["obj_error"],
            "obj_efficiency": final_result["obj_efficiency"],
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
