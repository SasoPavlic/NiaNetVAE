import json
from datetime import datetime

import numpy as np

from log import Log
from .runtime_artifacts import _as_jsonable, _resolve_export_dir


def _resolve_finetune_policy(config: dict) -> dict:
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
