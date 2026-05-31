import json
from pathlib import Path

import pytest

from nianetvae.tools import generate_cycle_manifest


def _write_cycle_artifacts(cycle_dir: Path, *, with_scaler: bool = True) -> None:
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
        },
        "provenance": {
            "experiment_mode": "per_maint_finetune_search",
            "source_cycle": None,
            "seed_source": 42,
        },
    }
    (cycle_dir / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")


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
