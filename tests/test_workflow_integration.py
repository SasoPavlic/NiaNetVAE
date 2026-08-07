from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from nianetvae.artifacts import StudyArtifactStore, read_json
from nianetvae.dataloaders.metropt import (
    cycle_source_and_anchor_masks,
    prepare_metropt,
)
from nianetvae.experiments import WorkflowRunner, build_comparison
from nianetvae.search.engine import SearchEngine

from .helpers import synthetic_config


def test_iforest_static_and_nianetvae_share_end_to_end_contract(tmp_path) -> None:
    workflows = ("iforest_static", "nianetvae_per_maintenance")
    config = synthetic_config(tmp_path, workflows=workflows)
    prepared = prepare_metropt(config.data, config.preprocessing.policy)
    store = StudyArtifactStore.from_config(config)
    store.initialize(config, prepared, repository=tmp_path)
    store.save_prepared_cache(prepared)

    SearchEngine(config, prepared, store).run()

    runner = WorkflowRunner(config, prepared, store)
    iforest = runner.run_workflow("iforest_static")
    nianet = runner.run_workflow("nianetvae_per_maintenance")
    assert iforest["workflow_id"] == "iforest_static"
    assert nianet["workflow_id"] == "nianetvae_per_maintenance"
    assert read_json(store.workflow_manifest_path("iforest_static"))["status"] == "completed"
    assert (
        read_json(store.workflow_manifest_path("nianetvae_per_maintenance"))["status"]
        == "completed"
    )
    build_comparison(config, store)
    assert store.validate_study(workflows)["valid"] is True

    cycle = read_json(
        store.workflow_dir("iforest_static") / "cycles" / "cycle_00" / "cycle_result.json"
    )
    cycle_predictions = store.root / cycle["predictions"]
    cycle_predictions.write_text(
        cycle_predictions.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="predictions hash mismatch"):
        store.validate_study(workflows)


def test_empty_evaluation_cycle_preserves_lineage_and_validates(tmp_path) -> None:
    workflows = ("iforest_static", "iforest_per_maintenance")
    config = synthetic_config(tmp_path, workflows=workflows)
    adjacent_events = (
        ("2020-01-01 04:00:00", "2020-01-01 04:10:00", "#1", "high"),
        ("2020-01-01 04:11:00", "2020-01-01 04:20:00", "#2", "high"),
        ("2020-01-01 07:10:00", "2020-01-01 07:20:00", "#3", "high"),
    )
    config = replace(config, data=replace(config.data, maintenance_windows=adjacent_events))
    prepared = prepare_metropt(config.data, config.preprocessing.policy)
    anchor_counts = [
        int(cycle_source_and_anchor_masks(prepared, cycle, config.data.test_phases)[1].sum())
        for cycle in prepared.cycles
    ]
    assert anchor_counts[1] == 0
    assert sum(anchor_counts) == int(prepared.evaluation_mask.sum())

    store = StudyArtifactStore.from_config(config)
    store.initialize(config, prepared, repository=tmp_path)
    store.save_prepared_cache(prepared)
    runner = WorkflowRunner(config, prepared, store)
    for workflow_id in workflows:
        runner.run_workflow(workflow_id)
    build_comparison(config, store)
    assert store.validate_study(workflows)["valid"] is True

    cycle = read_json(
        store.workflow_dir("iforest_static") / "cycles" / "cycle_01" / "cycle_result.json"
    )
    assert cycle["model_status"] == "reused_static"
    assert cycle["prediction_count"] == 0
    assert cycle["evaluation_status"] == "no_evaluation_anchors"
    predictions = pd.read_csv(store.root / cycle["predictions"])
    assert predictions.empty
    assert list(predictions.columns) == [
        "timestamp",
        "cycle_id",
        "operation_phase",
        "anomaly_score",
        "risk_score",
        "maintenance_risk",
    ]

    adaptive_cycle = read_json(
        store.workflow_dir("iforest_per_maintenance") / "cycles" / "cycle_01" / "cycle_result.json"
    )
    assert adaptive_cycle["model_status"] == "alias_no_trainable_local_window"
    assert adaptive_cycle["effective_model_cycle"] == 0
    assert adaptive_cycle["prediction_count"] == 0
    assert adaptive_cycle["evaluation_status"] == "no_evaluation_anchors"

    contract = read_json(store.shared_dir / "data_contract.json")
    assert contract["cycles"][1]["evaluation_anchor_count"] == 0
    assert len(contract["cycles"][1]["evaluation_index_hash"]) == 64
