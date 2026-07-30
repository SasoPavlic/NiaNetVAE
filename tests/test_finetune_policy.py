import pytest

from nianetvae.search.cycle_warmstart import (
    _apply_finetune_data_constraints,
    _resolve_finetune_policy,
    export_skipped_non_trainable_cycle,
)


def test_finetune_policy_resolves_learning_rate_epochs_and_early_stopping() -> None:
    config = {
        "workflow": {
            "finetune": {
                "learning_rate_scale": 0.1,
                "min_epochs": 3,
                "max_epochs": 10,
                "early_stopping": {
                    "enabled": True,
                    "monitor": "val_loss",
                    "mode": "min",
                    "patience": 2,
                    "min_delta": 0.0001,
                },
            }
        },
        "trainer_params": {"min_epochs": 1, "max_epochs": 4},
        "exp_params": {"learning_rate": 0.003},
    }

    policy = _resolve_finetune_policy(config)

    assert policy["finetune_learning_rate"] == pytest.approx(0.0003)
    assert policy["trainer_params_override"] == {"min_epochs": 3, "max_epochs": 10}
    assert policy["early_stopping"] == {
        "enabled": True,
        "monitor": "val_loss",
        "mode": "min",
        "patience": 2,
        "min_delta": 0.0001,
    }


def test_finetune_policy_rejects_invalid_early_stopping_mode() -> None:
    config = {
        "workflow": {
            "finetune": {
                "early_stopping": {"enabled": True, "mode": "sideways"},
            }
        },
        "trainer_params": {"min_epochs": 1, "max_epochs": 3},
        "exp_params": {"learning_rate": 0.003},
    }

    with pytest.raises(ValueError, match="must be 'min' or 'max'"):
        _resolve_finetune_policy(config)


def test_short_local_constraint_disables_early_stopping_and_uses_min_epochs() -> None:
    policy = {
        "trainer_params_override": {"min_epochs": 3, "max_epochs": 10},
        "early_stopping": {
            "enabled": True,
            "monitor": "val_loss",
            "mode": "min",
            "patience": 2,
            "min_delta": 0.0001,
        },
        "short_local_fallback_applied": False,
    }
    split_info = {
        "fine_tune_data_policy": {
            "early_stopping_eligible": False,
            "short_local_fallback_reason": "test-short-local",
        }
    }

    resolved = _apply_finetune_data_constraints(policy, split_info)

    assert resolved["trainer_params_override"] == {"min_epochs": 3, "max_epochs": 3}
    assert resolved["early_stopping"]["enabled"] is False
    assert resolved["early_stopping"]["disabled_reason"] == (
        "short_local_segment_without_leakage_free_validation"
    )
    assert resolved["short_local_fallback_applied"] is True
    assert resolved["short_local_fallback_reason"] == "test-short-local"
    assert policy["trainer_params_override"]["max_epochs"] == 10
    assert policy["early_stopping"]["enabled"] is True


def test_skipped_cycle_removes_stale_trained_artifacts(tmp_path) -> None:
    cycle_dir = tmp_path / "MetroPT" / "cycle_10"
    cycle_dir.mkdir(parents=True)
    for filename in ("model.pt", "model_meta.json", "scaler.joblib", "search_summary.json"):
        (cycle_dir / filename).write_text("stale", encoding="utf-8")
    config = {
        "logging_params": {
            "export_enabled": True,
            "model_export_dir": str(tmp_path),
        },
        "data_params": {
            "dataset_name": "MetroPT",
            "regime": "per_maint",
            "cycle_id": 10,
        },
        "workflow": {"mode": "per_maint_finetune_search"},
        "exp_params": {"manual_seed": 42},
    }

    export_skipped_non_trainable_cycle(
        reason="non_trainable_cycle",
        detail="zero local fine-tune segments",
        config=config,
        run_uuid="test-run",
    )

    assert (cycle_dir / "cycle_status.json").exists()
    assert not any(
        (cycle_dir / filename).exists()
        for filename in ("model.pt", "model_meta.json", "scaler.joblib", "search_summary.json")
    )
