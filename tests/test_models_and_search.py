from __future__ import annotations

from dataclasses import replace

import numpy as np

from nianetvae.artifacts import StudyArtifactStore
from nianetvae.contracts import handcrafted_vae_spec
from nianetvae.dataloaders.metropt import prepare_metropt
from nianetvae.models.recurrent import build_recurrent_model
from nianetvae.search.engine import SearchEngine, parse_duration_seconds, select_winner
from nianetvae.search.genome import decode_genome

from .helpers import synthetic_config


def test_six_gene_decoder_is_deterministic_and_buildable() -> None:
    genome = (0.4, 0.2, 0.8, 0.04, 0.14, 0.55)
    first = decode_genome(genome, input_dim=90, sequence_length=200)
    second = decode_genome(genome, input_dim=90, sequence_length=200)
    assert first.architecture_hash == second.architecture_hash
    assert first.genome == genome
    assert build_recurrent_model(first).architecture == first
    assert (
        build_recurrent_model(handcrafted_vae_spec(90, 200)).architecture.model_kind
        == "recurrent_vae"
    )


def test_weighted_pareto_selection_and_duration_parser() -> None:
    rows = [
        _candidate(1, (0.2, 0.6, 0.2)),
        _candidate(2, (0.4, 0.2, 0.1)),
        _candidate(3, (0.8, 0.8, 0.8)),
    ]
    selected = select_winner(rows, weights=(0.2, 0.5, 0.3), penalty=9e10)
    assert selected["candidate"]["id"] in {1, 2}
    assert selected["pareto_candidate_count"] == 2
    assert parse_duration_seconds("72:00:00") == 259_200


def test_one_generation_search_persists_candidates_checkpoint_and_winner(tmp_path) -> None:
    config = synthetic_config(tmp_path, workflows=("nianetvae_per_maintenance",))
    prepared = prepare_metropt(config.data, config.preprocessing.policy)
    artifacts = StudyArtifactStore.from_config(config)
    artifacts.initialize(config, prepared, repository=tmp_path)
    selected = SearchEngine(config, prepared, artifacts).run()
    assert selected["candidate_count"] == 3
    assert selected["valid_candidate_count"] == 3
    assert (artifacts.search_dir / "candidates.csv").is_file()
    assert (artifacts.search_dir / "checkpoints" / "nsga3.dill").is_file()
    assert (artifacts.search_dir / "selected_architecture.json").is_file()

    extended = replace(
        config,
        search=replace(config.search, max_generations=3, max_time="00:20:00"),
    ).validate()
    assert extended.fingerprint() == config.fingerprint()
    resumed = SearchEngine(extended, prepared, artifacts).run()
    assert resumed["candidate_count"] >= selected["candidate_count"]
    assert resumed["pymoo_result"]["resumed_from_generation"] >= 1
    assert resumed["execution_budget"]["max_generations"] == 3


def _candidate(identifier: int, objectives: tuple[float, float, float]) -> dict:
    architecture = decode_genome(np.full(6, identifier / 10.0), input_dim=90, sequence_length=200)
    return {
        "id": identifier,
        "status": "valid",
        "obj_error": objectives[0],
        "obj_pdm": objectives[1],
        "obj_alarm_burden": objectives[2],
        "created_at": f"2020-01-0{identifier}",
        "architecture": architecture.as_dict(),
        "genome": architecture.genome,
    }
