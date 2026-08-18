from __future__ import annotations

from dataclasses import replace

import pytest

from nianetvae.artifacts import StudyArtifactStore, read_json
from nianetvae.dataloaders.metropt import prepare_metropt
from nianetvae.experiments import WorkflowRunner, build_comparison
from nianetvae.search.engine import SearchEngine
from nianetvae.search.migration import (
    migrate_search_artifacts,
    search_runtime_fingerprint,
)

from .helpers import synthetic_config


def test_verified_search_migration_preserves_donor_evidence_and_validates(tmp_path) -> None:
    workflows = ("iforest_static", "nianetvae_per_maintenance")
    donor = synthetic_config(tmp_path, workflows=workflows)
    prepared = prepare_metropt(donor.data, donor.preprocessing.policy)
    donor_store = StudyArtifactStore.from_config(donor)
    donor_store.initialize(donor, prepared, repository=tmp_path)
    donor_store.save_prepared_cache(prepared)
    donor_selected = SearchEngine(donor, prepared, donor_store).run()

    target = replace(
        donor,
        study_name="synthetic controlled migration target",
        artifacts=replace(donor.artifacts, study_id="synthetic_v2"),
    ).validate()
    target_prepared = prepare_metropt(target.data, target.preprocessing.policy)
    target_store = StudyArtifactStore.from_config(target)
    target_store.initialize(target, target_prepared, repository=tmp_path)
    target_store.save_prepared_cache(target_prepared)

    migrated = migrate_search_artifacts(
        target,
        target_prepared,
        target_store,
        donor_study_root=donor_store.root,
        donor_search_runtime_fingerprint=search_runtime_fingerprint(),
    )

    assert migrated["execution_mode"] == "verified_search_migration"
    assert migrated["migration"]["donor_study_id"] == donor.artifacts.study_id
    assert migrated["candidate_count"] == donor_selected["candidate_count"]
    selected = read_json(target_store.search_dir / "selected_architecture.json")
    assert selected["study_id"] == target.artifacts.study_id
    assert selected["architecture"] == donor_selected["architecture"]
    assert (
        SearchEngine(target, target_prepared, target_store).run()["architecture"]
        == selected["architecture"]
    )

    runner = WorkflowRunner(target, target_prepared, target_store)
    for workflow_id in workflows:
        runner.run_workflow(workflow_id)
    build_comparison(target, target_store)
    assert target_store.validate_study(workflows)["valid"] is True


def test_search_migration_rejects_an_unverified_donor_runtime(tmp_path) -> None:
    donor = synthetic_config(tmp_path, workflows=("nianetvae_per_maintenance",))
    prepared = prepare_metropt(donor.data, donor.preprocessing.policy)
    donor_store = StudyArtifactStore.from_config(donor)
    donor_store.initialize(donor, prepared, repository=tmp_path)
    donor_store.save_prepared_cache(prepared)
    SearchEngine(donor, prepared, donor_store).run()

    target = replace(
        donor,
        artifacts=replace(donor.artifacts, study_id="synthetic_v2"),
    ).validate()
    target_store = StudyArtifactStore.from_config(target)
    target_store.initialize(target, prepared, repository=tmp_path)
    target_store.save_prepared_cache(prepared)

    with pytest.raises(ValueError, match="runtime fingerprints differ"):
        migrate_search_artifacts(
            target,
            prepared,
            target_store,
            donor_study_root=donor_store.root,
            donor_search_runtime_fingerprint="0" * 64,
        )
