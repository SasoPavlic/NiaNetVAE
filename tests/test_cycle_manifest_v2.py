import hashlib
import json
from pathlib import Path

import pytest

from nianetvae.tools import generate_cycle_manifest


def _write_cycle_artifacts(
    cycle_dir: Path,
    *,
    with_scaler: bool = True,
    preprocessing_policy: str | None = None,
) -> None:
    cycle_dir.mkdir(parents=True, exist_ok=True)
    (cycle_dir / "model.pt").write_text("weights", encoding="utf-8")
    if with_scaler:
        (cycle_dir / "scaler.joblib").write_text("scaler", encoding="utf-8")
    meta = {
        "schema_version": "2.0",
        "contract_version": "2.0",
        "cycle_id": 0,
        "hash_id": "abc",
        "run_uuid": "run",
        "created_at": "2026-01-01T00:00:00",
        "feature_contract": {
            "feature_hash": "hash",
            "rolling_window": "60s",
        },
        "preprocessing_contract": {
            "scaler_file": "scaler.joblib",
        },
        "sequence_contract": {
            "seq_len": 200,
            "stride": 1,
        },
        "split_contract": {
            "train_minutes": 1440,
            "post_train_minutes": 1440,
            "pre_maint_minutes": 120,
            "validation_split_policy": "window_chronological_v1",
            "batch_size": 64,
            "shuffle_train": True,
            "drop_last_train": False,
            "train_shuffle_seed": 42,
        },
        "provenance": {
            "experiment_mode": "per_maint_finetune_search",
            "source_cycle": None,
            "seed_source": 42,
            "mode": "fixed_architecture_retrain",
            "search_performed": False,
            "initialization": "fresh_seeded",
            "source_label": "frozen cycle-0 winner",
            "expected_hash_id": "abc",
            "retrain_from_scratch": True,
        },
    }
    if preprocessing_policy is not None:
        meta["preprocessing_contract"] = {
            "policy": preprocessing_policy,
            "policy_version": "1.0",
            "contract_hash": "preprocessing-hash",
            "passthrough_feature_count": 48,
        }
    (cycle_dir / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_manifest_config_fingerprint_preserves_implicit_legacy_policy() -> None:
    implicit = {
        "data_params": {
            "dataset_name": "MetroPT",
            "rolling_window": "60s",
            "seq_len": 200,
        }
    }
    explicit = json.loads(json.dumps(implicit))
    explicit["data_params"]["preprocessing_policy"] = "standard_scaler_v1"
    explicit["data_params"]["binary_feature_names"] = ["COMP"]
    binary = json.loads(json.dumps(explicit))
    binary["data_params"]["preprocessing_policy"] = "binary_passthrough_v1"
    explicit_loader = json.loads(json.dumps(implicit))
    explicit_loader["data_params"].update(
        {
            "batch_size": 64,
            "shuffle_train": True,
            "drop_last_train": False,
            "train_shuffle_seed": 42,
        }
    )
    data_params = implicit["data_params"]
    legacy_payload = {
        "dataset_name": data_params.get("dataset_name"),
        "data_path": data_params.get("data_path"),
        "rolling_window": data_params.get("rolling_window"),
        "seq_len": data_params.get("seq_len"),
        "stride": data_params.get("stride"),
        "train_minutes": data_params.get("train_minutes"),
        "post_train_minutes": data_params.get("post_train_minutes"),
        "pre_maint_minutes": data_params.get("pre_maint_minutes"),
        "train_phases": data_params.get("train_phases"),
        "test_phases": data_params.get("test_phases"),
        "workflow_mode": data_params.get("workflow_mode"),
        "finetune_data_policy": data_params.get("finetune_data_policy"),
    }
    legacy_raw = json.dumps(legacy_payload, sort_keys=True, default=str)
    legacy_fingerprint = hashlib.sha1(legacy_raw.encode("utf-8")).hexdigest()

    assert generate_cycle_manifest._config_fingerprint(implicit) == legacy_fingerprint
    assert generate_cycle_manifest._config_fingerprint(implicit) == (
        generate_cycle_manifest._config_fingerprint(explicit)
    )
    assert generate_cycle_manifest._config_fingerprint(implicit) != (
        generate_cycle_manifest._config_fingerprint(binary)
    )
    assert generate_cycle_manifest._config_fingerprint(implicit) != (
        generate_cycle_manifest._config_fingerprint(explicit_loader)
    )


def test_build_manifest_emits_v2_scaler_contract(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(generate_cycle_manifest, "_cycle_trainable", lambda _config, _cycle_id: (True, None))
    export_root = tmp_path / "exports"
    cycle_dir = export_root / "MetroPT" / "cycle_00"
    _write_cycle_artifacts(cycle_dir)

    manifest = generate_cycle_manifest.build_manifest(
        config={
            "data_params": {"dataset_name": "MetroPT", "regime": "per_maint"},
            "workflow": {"mode": "per_maint_finetune_search"},
            "exp_params": {"manual_seed": 42},
        },
        export_root=export_root,
        cycles=[0],
        paths_relative_to=export_root / "MetroPT",
    )

    cycle = manifest["cycles"]["00"]
    assert manifest["schema_version"] == "2.0"
    assert manifest["contract_version"] == "2.0"
    assert cycle["status"] == "trained"
    assert cycle["contract_version"] == "2.0"
    assert cycle["scaler_path"] == "cycle_00/scaler.joblib"
    assert cycle["feature_hash"] == "hash"
    assert cycle["seq_len"] == 200
    assert cycle["rolling_window"] == "60s"
    assert cycle["preprocessing_policy"] == "standard_scaler_v1"
    assert cycle["preprocessing_policy_version"] == "1.0"
    assert cycle["binary_passthrough_feature_count"] == 0
    assert cycle["validation_split_policy"] == "window_chronological_v1"
    assert cycle["batch_size"] == 64
    assert cycle["shuffle_train"] is True
    assert cycle["drop_last_train"] is False
    assert cycle["train_shuffle_seed"] == 42
    assert cycle["mode"] == "fixed_architecture_retrain"
    assert cycle["search_performed"] is False
    assert cycle["initialization"] == "fresh_seeded"
    assert cycle["source_label"] == "frozen cycle-0 winner"
    assert cycle["expected_hash_id"] == "abc"
    assert cycle["retrain_from_scratch"] is True
    assert manifest["observed_preprocessing_policies"] == ["standard_scaler_v1"]


def test_build_manifest_propagates_binary_preprocessing_contract(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        generate_cycle_manifest,
        "_cycle_trainable",
        lambda _config, _cycle_id: (True, None),
    )
    export_root = tmp_path / "exports"
    _write_cycle_artifacts(
        export_root / "MetroPT" / "cycle_00",
        preprocessing_policy="binary_passthrough_v1",
    )

    manifest = generate_cycle_manifest.build_manifest(
        config={
            "data_params": {
                "dataset_name": "MetroPT",
                "regime": "per_maint",
                "preprocessing_policy": "binary_passthrough_v1",
            },
            "workflow": {"mode": "per_maint_finetune_search"},
            "exp_params": {"manual_seed": 42},
        },
        export_root=export_root,
        cycles=[0],
        paths_relative_to=export_root / "MetroPT",
    )

    cycle = manifest["cycles"]["00"]
    assert manifest["preprocessing_policy"] == "binary_passthrough_v1"
    assert cycle["preprocessing_policy"] == "binary_passthrough_v1"
    assert cycle["preprocessing_policy_version"] == "1.0"
    assert cycle["preprocessing_contract_hash"] == "preprocessing-hash"
    assert cycle["binary_passthrough_feature_count"] == 48
    assert manifest["observed_preprocessing_policies"] == ["binary_passthrough_v1"]


def test_build_manifest_rejects_missing_v2_scaler(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(generate_cycle_manifest, "_cycle_trainable", lambda _config, _cycle_id: (True, None))
    export_root = tmp_path / "exports"
    _write_cycle_artifacts(export_root / "MetroPT" / "cycle_00", with_scaler=False)

    with pytest.raises(FileNotFoundError, match="scaler artifact"):
        generate_cycle_manifest.build_manifest(
            config={
                "data_params": {"dataset_name": "MetroPT", "regime": "per_maint"},
                "workflow": {"mode": "per_maint_finetune_search"},
                "exp_params": {"manual_seed": 42},
            },
            export_root=export_root,
            cycles=[0],
            paths_relative_to=export_root / "MetroPT",
        )
