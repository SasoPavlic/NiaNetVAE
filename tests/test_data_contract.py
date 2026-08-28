from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pandas as pd
import pytest
import yaml

from nianetvae.artifacts import StudyArtifactStore, read_json
from nianetvae.cli import main
from nianetvae.config import StudyConfig
from nianetvae.dataloaders.metropt import build_events, prepare_metropt
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


def test_reported_failure_rows_merge_into_physical_failures() -> None:
    """Davari et al. (2021) Table II splits multi-day failures at midnight.

    Five consecutive rows sit one minute apart at midnight because the expert
    reported each calendar day separately. Treating them as distinct failures
    inflates the event count and places the pre-failure window of the
    continuation inside the still-running failure it is meant to predict.
    """
    config = StudyConfig()
    verbatim = build_events(replace(config.data, failure_merge_gap_minutes=0))
    merged = build_events(config.data)

    assert len(verbatim) == 21, "the reported schedule must stay verbatim in configuration"
    assert len(merged) == 16, "21 reported rows describe 16 physical failures"
    assert [event.event_id for event in merged if "+" in event.event_id] == [
        "#2+#3",
        "#9+#10",
        "#12+#13",
        "#16+#17+#18",
    ]
    for earlier, later in zip(merged, merged[1:], strict=False):
        gap = (later.start - earlier.end).total_seconds() / 60.0
        assert gap > config.data.failure_merge_gap_minutes


def test_failure_merging_is_disabled_by_a_zero_tolerance() -> None:
    config = StudyConfig()
    verbatim = build_events(replace(config.data, failure_merge_gap_minutes=0))
    assert [event.event_id for event in verbatim[:3]] == ["#1", "#2", "#3"]
