from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
import yaml

from nianetvae.artifacts import StudyArtifactStore, read_json
from nianetvae.cli import main
from nianetvae.dataloaders.metropt import prepare_metropt
from nianetvae.dataloaders.sequences import sequence_anchor_mask

from .helpers import synthetic_config


def test_sequence_anchors_reset_across_mask_gaps() -> None:
    index = pd.date_range("2020-01-01", periods=10, freq="min")
    mask = pd.Series([True, True, True, False, True, True, True, True, False, True], index=index)
    anchors = sequence_anchor_mask(mask, sequence_length=3)
    assert anchors[anchors].index.tolist() == [index[2], index[6], index[7]]


def test_prepared_data_freezes_one_shared_population_and_cache(tmp_path) -> None:
    config = synthetic_config(tmp_path)
    prepared = prepare_metropt(config.data, config.preprocessing.policy)
    assert prepared.scaled_features.shape[1] == 90
    assert prepared.preprocessor.fitted_row_count == int(prepared.baseline_train_mask.sum())
    assert not (prepared.evaluation_mask & prepared.post_maintenance_train_mask).any()
    assert set(prepared.operation_phase.unique()) == {0, 1, 2}

    store = StudyArtifactStore.from_config(config)
    store.initialize(config, prepared, repository=tmp_path)
    store.save_prepared_cache(prepared)
    restored = store.load_prepared_cache()
    assert restored.data_contract_fingerprint == prepared.data_contract_fingerprint
    assert restored.preprocessor.fingerprint == prepared.preprocessor.fingerprint
    assert len(read_json(store.manifest_path)["source_contract_fingerprint"]) == 64


def test_prepared_study_rejects_a_different_runtime_source(tmp_path, monkeypatch) -> None:
    config = synthetic_config(tmp_path)
    prepared = prepare_metropt(config.data, config.preprocessing.policy)
    store = StudyArtifactStore.from_config(config)
    store.initialize(config, prepared, repository=tmp_path)
    monkeypatch.setattr("nianetvae.artifacts.source_contract_fingerprint", lambda: "0" * 64)
    with pytest.raises(ValueError, match="source_contract_fingerprint"):
        store.assert_initialized(config, prepared)


def test_cli_prepare_reuses_hash_verified_cache(tmp_path) -> None:
    config = synthetic_config(tmp_path)
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(config.as_dict(), sort_keys=False), encoding="utf-8")
    assert main(["--config", str(config_path), "prepare"]) == 0
    store = StudyArtifactStore.from_config(config)
    first_hash = store.prepared_cache_path.stat().st_mtime_ns
    assert main(["--config", str(config_path), "prepare"]) == 0
    assert store.prepared_cache_path.stat().st_mtime_ns == first_hash


def test_shared_manifest_updates_are_serialized(tmp_path) -> None:
    config = synthetic_config(tmp_path)
    prepared = prepare_metropt(config.data, config.preprocessing.policy)
    store = StudyArtifactStore.from_config(config)
    store.initialize(config, prepared, repository=tmp_path)

    def record(value: int) -> None:
        def update(manifest: dict) -> None:
            manifest.setdefault("concurrent_test_values", []).append(value)

        store.update_study_manifest(update)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(record, range(12)))
    manifest = read_json(store.manifest_path)
    assert sorted(manifest["concurrent_test_values"]) == list(range(12))
