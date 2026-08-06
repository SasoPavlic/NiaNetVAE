from __future__ import annotations

import pytest

from nianetvae.artifacts import StudyArtifactStore, read_json
from nianetvae.dataloaders.metropt import prepare_metropt
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
