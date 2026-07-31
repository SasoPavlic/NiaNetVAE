import json
import math
from datetime import datetime

import numpy as np

from log import Log
from .runtime_artifacts import _as_jsonable, _resolve_export_dir


def _resolve_early_stopping_policy(raw_policy: dict | None) -> dict:
    raw_policy = dict(raw_policy or {})
    policy = {
        "enabled": bool(raw_policy.get("enabled", False)),
        "monitor": str(raw_policy.get("monitor", "val_loss")),
        "mode": str(raw_policy.get("mode", "min")).strip().lower(),
        "patience": int(raw_policy.get("patience", 2)),
        "min_delta": float(raw_policy.get("min_delta", 0.0)),
        "restore_best_weights": bool(raw_policy.get("restore_best_weights", True)),
    }
    if policy["mode"] not in {"min", "max"}:
        raise ValueError("Early-stopping mode must be 'min' or 'max'.")
    if policy["patience"] < 0:
        raise ValueError("Early-stopping patience must be >= 0.")
    if not math.isfinite(policy["min_delta"]) or policy["min_delta"] < 0:
        raise ValueError("Early-stopping min_delta must be a finite value >= 0.")
    if policy["enabled"] and not policy["monitor"].strip():
        raise ValueError("Early-stopping monitor must not be empty when enabled.")
    return policy


def _resolve_cycle0_training_policy(config: dict) -> dict:
    """Resolve the optional fixed-architecture cycle-0 retraining contract."""
    workflow = dict(config.get("workflow") or {})
    finetune_cfg = dict(workflow.get("finetune") or {})
    cycle0_cfg = dict(finetune_cfg.get("cycle0") or {})
    raw_mode = cycle0_cfg.get("mode", "architecture_search")
    mode = str(raw_mode).strip().lower()
    allowed_modes = {"architecture_search", "fixed_architecture_retrain"}
    if mode not in allowed_modes:
        raise ValueError(
            f"Invalid workflow.finetune.cycle0.mode={raw_mode!r}. "
            f"Allowed values: {', '.join(sorted(allowed_modes))}."
        )

    if mode == "architecture_search":
        return {
            "mode": mode,
            "search_performed": True,
        }

    raw_solution = cycle0_cfg.get("solution")
    try:
        solution = np.asarray(raw_solution, dtype=float).reshape(-1)
    except Exception:
        raise ValueError(
            "workflow.finetune.cycle0.solution must be a finite six-value architecture vector."
        ) from None
    if solution.size != 6 or not np.isfinite(solution).all():
        raise ValueError(
            "workflow.finetune.cycle0.solution must be a finite six-value architecture vector."
        )
    if np.any(solution < 0.0) or np.any(solution > 1.0):
        raise ValueError(
            "workflow.finetune.cycle0.solution values must all be within [0, 1]."
        )

    expected_hash_id = str(cycle0_cfg.get("expected_hash_id") or "").strip()
    if not expected_hash_id:
        raise ValueError(
            "workflow.finetune.cycle0.expected_hash_id is required for fixed retraining."
        )
    if cycle0_cfg.get("retrain_from_scratch") is not True:
        raise ValueError(
            "workflow.finetune.cycle0.retrain_from_scratch must be true; "
            "Stage-3 retraining must not reuse the old weights."
        )

    trainer_params = dict(config.get("trainer_params") or {})
    default_min_epochs = int(trainer_params.get("min_epochs", 1))
    min_epochs = int(cycle0_cfg.get("min_epochs", default_min_epochs))
    max_epochs = int(cycle0_cfg.get("max_epochs", 30))
    if min_epochs < 1 or max_epochs < 1 or min_epochs > max_epochs:
        raise ValueError(
            "Invalid workflow.finetune.cycle0 epoch policy: require "
            "1 <= min_epochs <= max_epochs."
        )

    early_stopping_cfg = dict(finetune_cfg.get("early_stopping") or {})
    early_stopping_cfg.update(dict(cycle0_cfg.get("early_stopping") or {}))
    early_stopping = _resolve_early_stopping_policy(early_stopping_cfg)
    if not early_stopping["enabled"]:
        raise ValueError(
            "workflow.finetune.cycle0 fixed retraining requires early_stopping.enabled=true."
        )

    return {
        "mode": mode,
        "search_performed": False,
        "solution": solution.tolist(),
        "expected_hash_id": expected_hash_id,
        "source_label": str(cycle0_cfg.get("source_label") or "").strip() or None,
        "retrain_from_scratch": True,
        "initialization": "fresh_seeded",
        "deterministic": bool(cycle0_cfg.get("deterministic", True)),
        "trainer_params_override": {
            "min_epochs": min_epochs,
            "max_epochs": max_epochs,
        },
        "early_stopping": early_stopping,
    }


def _resolve_finetune_policy(config: dict) -> dict:
    workflow = config.get("workflow") or {}
    finetune_cfg = workflow.get("finetune") or {}
    exp_params = config.get("exp_params") or {}
    trainer_params = config.get("trainer_params") or {}

    base_lr = float(exp_params.get("learning_rate", 0.003))
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

    early_stopping = _resolve_early_stopping_policy(
        finetune_cfg.get("early_stopping")
    )

    return {
        "base_learning_rate": base_lr,
        "learning_rate_scale": lr_scale,
        "finetune_learning_rate": finetune_lr,
        "deterministic": bool(finetune_cfg.get("deterministic", False)),
        "trainer_params_override": {
            "min_epochs": min_epochs,
            "max_epochs": max_epochs,
        },
        "early_stopping": early_stopping,
        "short_local_fallback_applied": False,
    }


def _apply_finetune_data_constraints(policy: dict, split_info: dict | None) -> dict:
    """Adapt the trainer only when leakage-free local validation is impossible."""
    resolved = {
        **dict(policy),
        "trainer_params_override": dict(policy.get("trainer_params_override") or {}),
        "early_stopping": dict(policy.get("early_stopping") or {}),
    }
    report = dict((split_info or {}).get("fine_tune_data_policy") or {})
    if report.get("early_stopping_eligible") is not False:
        return resolved

    min_epochs = int(resolved["trainer_params_override"]["min_epochs"])
    resolved["trainer_params_override"]["max_epochs"] = min_epochs
    resolved["early_stopping"]["enabled"] = False
    resolved["early_stopping"]["disabled_reason"] = (
        "short_local_segment_without_leakage_free_validation"
    )
    resolved["short_local_fallback_applied"] = True
    resolved["short_local_fallback_reason"] = report.get(
        "short_local_fallback_reason"
    )
    return resolved


def _resolve_cycle_export_dir(cycle_id: int, config: dict, run_uuid: str | None = None):
    cfg = {
        "logging_params": dict(config.get("logging_params", {})),
        "data_params": dict(config.get("data_params", {})),
    }
    cfg["data_params"]["regime"] = "per_maint"
    cfg["data_params"]["cycle_id"] = int(cycle_id)
    return _resolve_export_dir(cfg, run_uuid=run_uuid)


def _find_latest_trained_cycle_artifacts_before(cycle_id: int, config: dict, run_uuid: str | None = None):
    for source_cycle_id in range(int(cycle_id) - 1, -1, -1):
        source_cycle_dir = _resolve_cycle_export_dir(source_cycle_id, config=config, run_uuid=run_uuid)
        source_weights = source_cycle_dir / "model.pt"
        source_meta = source_cycle_dir / "model_meta.json"
        if source_weights.exists() and source_meta.exists():
            return source_cycle_id, source_cycle_dir, source_weights, source_meta
    return None


def _resolve_warm_start_sampling(
    dimensionality: int,
    effective_population: int,
    config: dict,
    run_uuid: str | None = None,
):
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

    previous_source = _find_latest_trained_cycle_artifacts_before(cycle_id, config=config, run_uuid=run_uuid)
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
        result["reason"] = f"invalid_anchor_dim:{anchor.size}_expected:{dimensionality}"
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


def export_skipped_non_trainable_cycle(
    reason: str,
    detail: str = "",
    source: str = "runtime",
    *,
    config: dict,
    run_uuid: str,
):
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

    export_dir = _resolve_export_dir(config, run_uuid=run_uuid)
    export_dir.mkdir(parents=True, exist_ok=True)
    status_path = export_dir / "cycle_status.json"
    removed_stale_artifacts = []
    for filename in ("model.pt", "model_meta.json", "scaler.joblib", "search_summary.json"):
        stale_path = export_dir / filename
        if stale_path.exists():
            stale_path.unlink()
            removed_stale_artifacts.append(filename)
    if removed_stale_artifacts:
        Log.info(
            f"STALE_CYCLE_ARTIFACTS_REMOVED cycle_id={cycle_id:02d} "
            f"files={','.join(removed_stale_artifacts)}"
        )
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
        "run_uuid": run_uuid,
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
