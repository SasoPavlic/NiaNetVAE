import gc
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import joblib
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from log import Log
from nianetvae.experiments.rnn_vae_experiment import RNNVAExperiment
from nianetvae.models.rnn_vae import RNNVAE


ARTIFACT_CONTRACT_VERSION = "2.0"


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


def _hash_json_payload(payload: object) -> str:
    raw = json.dumps(_as_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _feature_hash(feature_names: list[str]) -> str:
    raw = json.dumps(list(feature_names), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _scaler_state_payload(scaler) -> dict:
    payload = {
        "class": f"{type(scaler).__module__}.{type(scaler).__name__}",
    }
    for attr in ("mean_", "scale_", "var_", "n_features_in_", "n_samples_seen_"):
        if hasattr(scaler, attr):
            payload[attr] = _as_jsonable(getattr(scaler, attr))
    for attr in (
        "nianetvae_preprocessing_policy_",
        "nianetvae_preprocessing_policy_version_",
        "nianetvae_passthrough_indices_",
    ):
        if hasattr(scaler, attr):
            payload[attr] = _as_jsonable(getattr(scaler, attr))
    return payload


def _build_artifact_contracts(datamodule, data_params: dict, scaler_file: str) -> dict:
    if datamodule is None:
        raise ValueError("Artifact contract v2 export requires an initialized datamodule.")

    scaler = getattr(datamodule, "scaler", None)
    if scaler is None:
        raise ValueError("Artifact contract v2 export requires datamodule.scaler.")

    rolling_feature_names = list(getattr(datamodule, "rolling_feature_names", []) or [])
    if not rolling_feature_names:
        raise ValueError("Artifact contract v2 export requires datamodule.rolling_feature_names.")

    n_features = int(getattr(datamodule, "n_features", None) or data_params.get("n_features") or len(rolling_feature_names))
    if len(rolling_feature_names) != n_features:
        raise ValueError(
            "Feature contract mismatch during export: "
            f"rolling_feature_names={len(rolling_feature_names)} n_features={n_features}."
        )

    scaler_state = _scaler_state_payload(scaler)
    split_info = dict(getattr(datamodule, "split_info", {}) or {})
    feature_hash = getattr(datamodule, "feature_hash", None) or _feature_hash(rolling_feature_names)
    preprocessing_report = dict(
        getattr(datamodule, "preprocessing_report", {})
        or split_info.get("preprocessing_report")
        or {}
    )
    preprocessing_policy = str(
        preprocessing_report.get("policy")
        or data_params.get("preprocessing_policy")
        or "standard_scaler_v1"
    )
    preprocessing_report.setdefault("policy", preprocessing_policy)
    preprocessing_report.setdefault("policy_version", "1.0")
    standardized_feature_count = preprocessing_report.get(
        "standardized_feature_count"
    )
    if standardized_feature_count is None:
        standardized_feature_count = n_features
    preprocessing_payload = {
        "policy": preprocessing_policy,
        "policy_version": preprocessing_report.get("policy_version"),
        "behavior": preprocessing_report.get("behavior"),
        "preserves_feature_order": preprocessing_report.get(
            "preserves_feature_order", True
        ),
        "preserves_feature_count": preprocessing_report.get(
            "preserves_feature_count", True
        ),
        "configured_binary_feature_names": list(
            preprocessing_report.get("configured_binary_feature_names") or []
        ),
        "matched_binary_feature_names": list(
            preprocessing_report.get("matched_binary_feature_names") or []
        ),
        "binary_derived_feature_indices": list(
            preprocessing_report.get("binary_derived_feature_indices") or []
        ),
        "binary_derived_feature_names": list(
            preprocessing_report.get("binary_derived_feature_names") or []
        ),
        "binary_derived_feature_count": int(
            preprocessing_report.get("binary_derived_feature_count") or 0
        ),
        "applied_binary_feature_names": list(
            preprocessing_report.get("applied_binary_feature_names") or []
        ),
        "passthrough_feature_indices": list(
            preprocessing_report.get("passthrough_feature_indices") or []
        ),
        "passthrough_feature_names": list(
            preprocessing_report.get("passthrough_feature_names") or []
        ),
        "passthrough_feature_count": int(
            preprocessing_report.get("passthrough_feature_count") or 0
        ),
        "standardized_feature_indices": list(
            preprocessing_report.get("standardized_feature_indices") or []
        ),
        "standardized_feature_count": int(standardized_feature_count),
    }
    preprocessing_contract_hash = _hash_json_payload(preprocessing_payload)
    return {
        "feature_contract": {
            "base_feature_names": list(getattr(datamodule, "base_feature_names", []) or []),
            "rolling_feature_names": rolling_feature_names,
            "rolling_aggregations": list(getattr(datamodule, "rolling_aggregations", []) or []),
            "rolling_window": data_params.get("rolling_window") or getattr(datamodule, "rolling_window", None),
            "feature_hash": feature_hash,
            "n_features": n_features,
            "binary_base_feature_names": preprocessing_payload[
                "matched_binary_feature_names"
            ],
            "binary_derived_feature_names": preprocessing_payload[
                "binary_derived_feature_names"
            ],
            "binary_derived_feature_indices": preprocessing_payload[
                "binary_derived_feature_indices"
            ],
        },
        "preprocessing_contract": {
            **preprocessing_payload,
            "contract_hash": preprocessing_contract_hash,
            "scaler_type": scaler_state["class"],
            "scaler_file": scaler_file,
            "scaler_feature_count": int(getattr(scaler, "n_features_in_", n_features)),
            "scaler_hash": _hash_json_payload(scaler_state),
        },
        "sequence_contract": {
            "seq_len": data_params.get("seq_len") or getattr(datamodule, "seq_len", None),
            "stride": data_params.get("stride") or getattr(datamodule, "stride", None),
            "score_stride": data_params.get("stride") or getattr(datamodule, "stride", None),
            "window_label_policy": "end_anchor_phase",
            "cross_gap_windows_allowed": False,
        },
        "split_contract": {
            "regime": data_params.get("regime") or getattr(datamodule, "regime", None),
            "cycle_id": data_params.get("cycle_id") if data_params.get("cycle_id") is not None else getattr(datamodule, "cycle_id", None),
            "train_minutes": data_params.get("train_minutes") or getattr(datamodule, "train_minutes", None),
            "post_train_minutes": data_params.get("post_train_minutes") or getattr(datamodule, "post_train_minutes", None),
            "pre_maint_minutes": data_params.get("pre_maint_minutes") or getattr(datamodule, "pre_maint_minutes", None),
            "train_phases": data_params.get("train_phases") or getattr(datamodule, "train_phases", None),
            "test_phases": data_params.get("test_phases") or getattr(datamodule, "test_phases", None),
            "baseline_start": split_info.get("baseline_start"),
            "baseline_end": split_info.get("baseline_end"),
            "maintenance_id": split_info.get("maintenance_id"),
            "maintenance_start": split_info.get("maintenance_start"),
            "maintenance_end": split_info.get("maintenance_end"),
            "post_train_start": split_info.get("post_train_start"),
            "post_train_end": split_info.get("post_train_end"),
            "test_start": split_info.get("test_start"),
            "test_end": split_info.get("test_end"),
            "train_rows": split_info.get("train_rows"),
            "test_rows": split_info.get("test_rows"),
            "train_segments": list(getattr(datamodule, "train_segment_metadata", []) or []),
            "test_segments": list(getattr(datamodule, "test_segment_metadata", []) or []),
            "fine_tune_data_policy": split_info.get("fine_tune_data_policy"),
            "validation_split_policy": data_params.get("validation_split_policy")
            or getattr(datamodule, "validation_split_policy", "window_chronological_v1"),
            "validation_split_report": split_info.get("validation_split_report")
            or dict(getattr(datamodule, "validation_split_report", {}) or {}),
            "batch_size": data_params.get("batch_size")
            if data_params.get("batch_size") is not None
            else getattr(datamodule, "batch_size", None),
            "shuffle_train": split_info.get("shuffle_train"),
            "drop_last_train": split_info.get("drop_last_train"),
            "train_shuffle_seed": split_info.get("train_shuffle_seed"),
            "test_informative_for_pdm_objective": split_info.get("test_informative_for_pdm_objective"),
        },
    }


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
    early_stopping_policy: dict | None = None,
    deterministic: bool | None = None,
):
    trainer_params = dict(config.get('trainer_params', {}))
    if trainer_params_override:
        trainer_params.update(trainer_params_override)
    callbacks = []
    early_stopping_policy = dict(early_stopping_policy or {})
    early_stopping_enabled = bool(early_stopping_policy.get("enabled", False))
    if early_stopping_enabled:
        monitor = str(early_stopping_policy.get("monitor", "val_loss"))
        mode = str(early_stopping_policy.get("mode", "min"))
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(default_root_dir) / "best_checkpoints",
                filename="best-{epoch:03d}",
                monitor=monitor,
                mode=mode,
                save_top_k=1,
                save_last=False,
                save_weights_only=True,
                auto_insert_metric_name=False,
            )
        )
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=int(early_stopping_policy.get("patience", 2)),
                min_delta=float(early_stopping_policy.get("min_delta", 0.0)),
                strict=True,
                check_finite=True,
            )
        )
    return Trainer(
        enable_progress_bar=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        default_root_dir=default_root_dir,
        log_every_n_steps=50,
        logger=False,
        enable_checkpointing=early_stopping_enabled,
        callbacks=callbacks,
        deterministic=bool(deterministic) if deterministic is not None else None,
        **trainer_params
    )


def _restore_best_validation_weights(
    experiment: RNNVAExperiment,
    trainer: Trainer,
    early_stopping_policy: dict | None,
) -> dict:
    """Restore the best validation checkpoint before calibration and export."""
    policy = dict(early_stopping_policy or {})
    report = {
        "enabled": bool(policy.get("enabled", False)),
        "requested": bool(policy.get("restore_best_weights", True)),
        "restored_best_weights": False,
        "best_epoch": None,
        "best_validation_loss": None,
        "best_checkpoint_path": None,
    }
    if not report["enabled"] or not report["requested"]:
        return report

    callback = next(
        (
            item
            for item in getattr(trainer, "callbacks", [])
            if isinstance(item, ModelCheckpoint)
        ),
        None,
    )
    if callback is None or not str(callback.best_model_path or "").strip():
        raise RuntimeError(
            "Early stopping requested best-weight restoration, but no best checkpoint was produced."
        )

    checkpoint_path = Path(callback.best_model_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(
            f"Best validation checkpoint has no state_dict: {checkpoint_path}"
        )
    experiment.load_state_dict(state_dict, strict=True)

    best_score = getattr(callback, "best_model_score", None)
    if best_score is not None:
        try:
            best_score = float(best_score.detach().cpu().item())
        except Exception:
            best_score = float(best_score)
    best_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    report.update(
        {
            "restored_best_weights": True,
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_validation_loss": best_score,
            "best_checkpoint_path": str(checkpoint_path),
        }
    )
    Log.info(
        "BEST_WEIGHTS_RESTORED "
        f"checkpoint={checkpoint_path} best_epoch={report['best_epoch']} "
        f"best_validation_loss={report['best_validation_loss']}"
    )
    return report


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
    early_stopping_policy: dict | None = None,
    deterministic: bool | None = None,
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
        f"learning_rate={experiment.learning_rate} weight_decay={experiment.weight_decay} scheduler=none "
        f"min_epochs={effective_trainer_params.get('min_epochs')} "
        f"max_epochs={effective_trainer_params.get('max_epochs')}"
    )
    trainer = _build_final_trainer(
        config=config,
        default_root_dir=final_root,
        trainer_params_override=trainer_params_override,
        early_stopping_policy=early_stopping_policy,
        deterministic=deterministic,
    )

    started_at = datetime.now()
    trainer.fit(experiment, datamodule=datamodule)
    best_weights_report = _restore_best_validation_weights(
        experiment,
        trainer,
        early_stopping_policy,
    )
    experiment.collect_calibration_scores(datamodule.train_dataloader())
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
        "obj_pdm": objective_bundle["obj_pdm"],
        "obj_alarm_burden": objective_bundle["obj_alarm_burden"],
        "pdm_signal_quality": objective_bundle["pdm_signal_quality"],
        "diagnostic_params": objective_bundle.get("diagnostic_params"),
        "diagnostic_macs": objective_bundle.get("diagnostic_macs"),
        "diagnostic_macs_reason": objective_bundle.get("diagnostic_macs_reason"),
        "objective_reason": objective_bundle.get("reason"),
        "objective_contract": objective_bundle.get("objective_contract"),
        "metrics": final_metrics,
        "anomaly_metrics": anomaly_metrics,
        "trainer_policy": {
            "min_epochs": effective_trainer_params.get("min_epochs"),
            "max_epochs": effective_trainer_params.get("max_epochs"),
            "completed_epochs": int(getattr(trainer, "current_epoch", 0)),
            "early_stopping": dict(early_stopping_policy or {}),
            "deterministic": bool(deterministic) if deterministic is not None else None,
            **best_weights_report,
        },
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
        datamodule=None,
):
    export_dir.mkdir(parents=True, exist_ok=True)
    stale_status_path = export_dir / "cycle_status.json"
    if stale_status_path.exists():
        stale_status_path.unlink()
        Log.info(f"STALE_CYCLE_STATUS_REMOVED path={stale_status_path}")
    model_path = export_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    data_params = config.get("data_params", {})
    scaler_file = "scaler.joblib"
    contracts = _build_artifact_contracts(datamodule, data_params, scaler_file=scaler_file)
    scaler_path = export_dir / scaler_file
    joblib.dump(getattr(datamodule, "scaler"), scaler_path)

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
    for key in (
        "mode",
        "search_performed",
        "initialization",
        "source_label",
        "expected_hash_id",
        "retrain_from_scratch",
    ):
        if key in search_result:
            provenance[key] = _as_jsonable(search_result.get(key))
    if isinstance(warm_start_payload, dict):
        provenance["warm_start"] = _as_jsonable(warm_start_payload)
    winner_selection = search_result.get("winner_selection")
    metadata = {
        "schema_version": ARTIFACT_CONTRACT_VERSION,
        "contract_version": ARTIFACT_CONTRACT_VERSION,
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
        "scaler_file": scaler_file,
        "feature_contract": contracts["feature_contract"],
        "preprocessing_contract": contracts["preprocessing_contract"],
        "sequence_contract": contracts["sequence_contract"],
        "split_contract": contracts["split_contract"],
        "training_policy": {
            "optimizer": str(model.optimizer_name),
            "learning_rate": float(final_result["experiment"].learning_rate),
            "weight_decay": float(final_result["experiment"].weight_decay),
            "kld_weight": float((config.get("exp_params") or {}).get("kld_weight", 0.01)),
            "batch_size": _as_jsonable(
                (contracts.get("split_contract") or {}).get("batch_size")
            ),
            "trainer": _as_jsonable(final_result.get("trainer_policy") or {}),
            "fine_tune_data_policy": _as_jsonable(
                (contracts.get("split_contract") or {}).get("fine_tune_data_policy")
            ),
        },
        "final_training_anomaly_metrics": _as_jsonable(final_result.get("anomaly_metrics") or {}),
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
        "schema_version": ARTIFACT_CONTRACT_VERSION,
        "contract_version": ARTIFACT_CONTRACT_VERSION,
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
            "training_policy": {
                "optimizer": str(model.optimizer_name),
                "learning_rate": float(final_result["experiment"].learning_rate),
                "weight_decay": float(final_result["experiment"].weight_decay),
                "kld_weight": float((config.get("exp_params") or {}).get("kld_weight", 0.01)),
                "batch_size": _as_jsonable(
                    (contracts.get("split_contract") or {}).get("batch_size")
                ),
                "trainer": _as_jsonable(final_result.get("trainer_policy") or {}),
                "fine_tune_data_policy": _as_jsonable(
                    (contracts.get("split_contract") or {}).get("fine_tune_data_policy")
                ),
            },
            "obj_error": final_result["obj_error"],
            "obj_pdm": final_result.get("obj_pdm"),
            "obj_alarm_burden": final_result.get("obj_alarm_burden"),
            "pdm_signal_quality": final_result.get("pdm_signal_quality"),
            "diagnostic_params": final_result.get("diagnostic_params"),
            "diagnostic_macs": final_result.get("diagnostic_macs"),
            "diagnostic_macs_reason": final_result.get("diagnostic_macs_reason"),
            "objective_reason": final_result.get("objective_reason"),
            "objective_contract": final_result.get("objective_contract"),
            "metrics": final_result["metrics"],
            "anomaly_metrics": final_result["anomaly_metrics"],
        },
        "artifacts": {
            "weights_file": "model.pt",
            "meta_file": "model_meta.json",
            "scaler_file": scaler_file,
        },
        "feature_contract": contracts["feature_contract"],
        "preprocessing_contract": contracts["preprocessing_contract"],
        "sequence_contract": contracts["sequence_contract"],
        "split_contract": contracts["split_contract"],
        "training_policy": {
            "optimizer": str(model.optimizer_name),
            "learning_rate": float(final_result["experiment"].learning_rate),
            "weight_decay": float(final_result["experiment"].weight_decay),
            "kld_weight": float((config.get("exp_params") or {}).get("kld_weight", 0.01)),
            "batch_size": _as_jsonable(
                (contracts.get("split_contract") or {}).get("batch_size")
            ),
            "trainer": _as_jsonable(final_result.get("trainer_policy") or {}),
            "fine_tune_data_policy": _as_jsonable(
                (contracts.get("split_contract") or {}).get("fine_tune_data_policy")
            ),
        },
    }
    summary_path = export_dir / "search_summary.json"
    summary_path.write_text(json.dumps(_as_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
    return model_path, meta_path, summary_path
