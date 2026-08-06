"""Fresh shared-core NSGA-III search and deterministic winner selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection
from pymoo.util.ref_dirs import get_reference_directions

from ..artifacts import (
    StudyArtifactStore,
    atomic_write_csv,
    atomic_write_json,
    read_json,
    sha256_file,
    source_contract_fingerprint,
    utc_now,
)
from ..config import StudyConfig
from ..dataloaders.metropt import PreparedMetroPTData
from .checkpointing import SearchCheckpointManager
from .genome import GENOME_DIMENSION
from .objectives import CandidateEvaluator
from .storage import CandidateStore


def parse_duration_seconds(value: str) -> int:
    pieces = [piece.strip() for piece in str(value).split(":")]
    if len(pieces) != 3:
        raise ValueError("Search max_time must use HH:MM:SS syntax.")
    hours, minutes, seconds = (int(piece) for piece in pieces)
    if hours < 0 or minutes not in range(60) or seconds not in range(60):
        raise ValueError("Invalid search max_time.")
    total = hours * 3600 + minutes * 60 + seconds
    if total < 1:
        raise ValueError("Search max_time must be positive.")
    return total


def search_contract(config: StudyConfig, prepared: PreparedMetroPTData) -> dict[str, Any]:
    immutable = {
        "schema_version": "1.0",
        "study_id": config.artifacts.study_id,
        "source_contract_fingerprint": source_contract_fingerprint(),
        "data_contract_fingerprint": prepared.data_contract_fingerprint,
        "n_partitions": config.search.n_partitions,
        "genome_dimension": GENOME_DIMENSION,
        "candidate_min_epochs": config.search.candidate_min_epochs,
        "candidate_max_epochs": config.search.candidate_max_epochs,
        "training": asdict(config.training),
        "objectives": {
            "error": config.search.reconstruction_metric,
            "pdm": config.search.pdm_metric,
            "alarm_burden": config.search.alarm_burden_metric,
            "alarm_burden_risk_threshold": config.search.alarm_burden_risk_threshold,
            "smoothing_window_minutes": config.evaluation.risk_window_minutes,
        },
        "winner_weights": list(config.search.winner_weights),
        "invalid_penalty": config.search.invalid_penalty,
    }
    encoded = json.dumps(immutable, sort_keys=True, separators=(",", ":"), default=str)
    fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return {
        **immutable,
        "search_contract_fingerprint": fingerprint,
        "replaceable_budget": {
            "max_generations": config.search.max_generations,
            "max_time": config.search.max_time,
        },
    }


class _ArchitectureProblem(ElementwiseProblem):
    def __init__(self, evaluator: CandidateEvaluator) -> None:
        super().__init__(
            n_var=GENOME_DIMENSION,
            n_obj=3,
            n_ieq_constr=0,
            xl=np.zeros(GENOME_DIMENSION),
            xu=np.ones(GENOME_DIMENSION),
        )
        self.evaluator = evaluator

    def _evaluate(self, genome, out, *args, **kwargs) -> None:
        out["F"] = np.asarray(self.evaluator.evaluate(genome), dtype=float)


class SearchEngine:
    def __init__(
        self,
        config: StudyConfig,
        prepared: PreparedMetroPTData,
        artifacts: StudyArtifactStore,
    ) -> None:
        self.config = config.validate()
        self.prepared = prepared
        self.artifacts = artifacts
        artifacts.assert_initialized(config, prepared)
        if config.search.database_backend != "sqlite":
            raise ValueError("The rewritten study supports only the local SQLite candidate ledger.")
        database_path = artifacts.root / config.search.database_path
        self.candidates = CandidateStore(database_path, config.search.database_table)
        self.contract = search_contract(config, prepared)

    def run(self) -> dict[str, Any]:
        with self.artifacts.exclusive_lock("architecture-search", timeout_seconds=1.0):
            return self._run_locked()

    def _run_locked(self) -> dict[str, Any]:
        if not self.config.search.enabled:
            raise ValueError("Architecture search is disabled in the study configuration.")
        selected_path = self.artifacts.search_dir / "selected_architecture.json"
        manifest_path = self.artifacts.search_dir / "search_manifest.json"
        if selected_path.is_file():
            selected = read_json(selected_path)
            if (
                selected.get("search_contract_fingerprint")
                != self.contract["search_contract_fingerprint"]
            ):
                raise ValueError(
                    "Existing selected architecture belongs to another search contract."
                )
            if not manifest_path.is_file():
                raise FileNotFoundError("Selected architecture has no search manifest.")
            previous_manifest = read_json(manifest_path)
            selected_relative = previous_manifest.get("outputs", {}).get("selected_architecture")
            selected_hash = previous_manifest.get("output_sha256", {}).get("selected_architecture")
            if (
                previous_manifest.get("status") != "completed"
                or previous_manifest.get("search_contract_fingerprint")
                != self.contract["search_contract_fingerprint"]
                or selected_relative != self.artifacts.relative(selected_path)
                or not selected_hash
                or sha256_file(selected_path) != selected_hash
            ):
                raise ValueError("Selected architecture failed search-manifest validation.")
            previous_budget = selected.get("execution_budget")
            if not isinstance(previous_budget, dict):
                raise ValueError("Selected architecture is missing its search execution budget.")
            extended = int(self.config.search.max_generations) > int(
                previous_budget.get("max_generations", 0)
            ) or parse_duration_seconds(self.config.search.max_time) > parse_duration_seconds(
                str(previous_budget.get("max_time", "00:00:00"))
            )
            if not extended:
                return selected
            nianet_manifest = self.artifacts.workflow_manifest_path("nianetvae_per_maintenance")
            if nianet_manifest.exists():
                raise ValueError(
                    "Search budget cannot be extended after the NiaNetVAE workflow has begun. "
                    "Use a new study_id to avoid changing architecture mid-run."
                )
            checkpoint_path = self.artifacts.search_dir / "checkpoints" / "nsga3.dill"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "Search execution budget was extended, but the NSGA-III checkpoint is missing."
                )

        atomic_write_json(
            manifest_path,
            {
                **self.contract,
                "status": "running",
                "started_at": utc_now(),
            },
        )
        evaluator = CandidateEvaluator(
            self.config,
            self.prepared,
            self.candidates,
            self.contract["search_contract_fingerprint"],
        )
        problem = _ArchitectureProblem(evaluator)
        reference_directions = get_reference_directions(
            "das-dennis",
            3,
            n_partitions=self.config.search.n_partitions,
        )
        termination = TerminationCollection(
            get_termination("n_gen", self.config.search.max_generations),
            get_termination("time", max_time=parse_duration_seconds(self.config.search.max_time)),
        )
        checkpoint = SearchCheckpointManager(
            self.artifacts.search_dir / "checkpoints",
            contract=self.contract,
            interval_generations=self.config.search.checkpoint_interval_generations,
        )
        callback = checkpoint.callback()
        algorithm = checkpoint.load(problem=problem, termination=termination, callback=callback)
        resumed_from_generation = (
            max(0, int(getattr(algorithm, "n_gen", 0) or 0) - 1) if algorithm is not None else 0
        )
        if algorithm is None:
            algorithm = NSGA3(
                pop_size=len(reference_directions),
                ref_dirs=reference_directions,
            )
        result = minimize(
            problem,
            algorithm,
            termination=termination,
            seed=self.config.training.seed,
            callback=callback,
            verbose=True,
            copy_algorithm=False,
            save_history=False,
        )
        checkpoint.save(algorithm, completed=True)
        self.candidates.checkpoint()

        candidates = self.candidates.frame(
            study_id=self.config.artifacts.study_id,
            search_contract_fingerprint=self.contract["search_contract_fingerprint"],
        )
        atomic_write_csv(self.artifacts.search_dir / "candidates.csv", candidates)
        selected = select_winner(
            self.candidates.rows(
                study_id=self.config.artifacts.study_id,
                search_contract_fingerprint=self.contract["search_contract_fingerprint"],
            ),
            weights=self.config.search.winner_weights,
            penalty=self.config.search.invalid_penalty,
        )
        next_generation = int(getattr(algorithm, "n_gen", 0) or 0)
        completed_generations = max(0, next_generation - 1)
        stop_reason = (
            "generation_budget"
            if completed_generations >= self.config.search.max_generations
            else "time_budget"
        )
        payload = {
            "schema_version": "1.0",
            "study_id": self.config.artifacts.study_id,
            "search_contract_fingerprint": self.contract["search_contract_fingerprint"],
            "selection_method": "pareto_weighted_ideal_distance",
            "weights": list(_normalized_weights(self.config.search.winner_weights)),
            "candidate_count": selected["candidate_count"],
            "valid_candidate_count": selected["valid_candidate_count"],
            "pareto_candidate_count": selected["pareto_candidate_count"],
            "selected_candidate_id": selected["candidate"]["id"],
            "selected_distance": selected["distance"],
            "selected_objectives": {
                "obj_error": selected["candidate"]["obj_error"],
                "obj_pdm": selected["candidate"]["obj_pdm"],
                "obj_alarm_burden": selected["candidate"]["obj_alarm_burden"],
            },
            "architecture": selected["candidate"]["architecture"],
            "genome": list(selected["candidate"]["genome"]),
            "execution_budget": {
                "max_generations": self.config.search.max_generations,
                "max_time": self.config.search.max_time,
            },
            "completed_at": utc_now(),
            "pymoo_result": {
                "next_generation": next_generation,
                "completed_generations": completed_generations,
                "resumed_from_generation": resumed_from_generation,
                "evaluations": int(
                    getattr(getattr(algorithm, "evaluator", None), "n_eval", 0) or 0
                ),
                "stop_reason": stop_reason,
                "termination_message": str(getattr(result, "message", "")),
            },
        }
        atomic_write_json(selected_path, payload)
        search_outputs = {
            "selected_architecture": self.artifacts.relative(selected_path),
            "candidates_csv": self.artifacts.relative(self.artifacts.search_dir / "candidates.csv"),
            "candidate_database": self.artifacts.relative(self.candidates.path),
            "algorithm_checkpoint": self.artifacts.relative(checkpoint.algorithm_path),
            "checkpoint_metadata": self.artifacts.relative(checkpoint.metadata_path),
        }
        atomic_write_json(
            manifest_path,
            {
                **self.contract,
                "status": "completed",
                "completed_at": utc_now(),
                "candidate_count": selected["candidate_count"],
                "valid_candidate_count": selected["valid_candidate_count"],
                "pareto_candidate_count": selected["pareto_candidate_count"],
                "selected_architecture": self.artifacts.relative(selected_path),
                "execution_budget": payload["execution_budget"],
                "pymoo_result": payload["pymoo_result"],
                "outputs": search_outputs,
                "output_sha256": {
                    label: sha256_file(self.artifacts.root / relative)
                    for label, relative in search_outputs.items()
                },
            },
        )

        def update(study: dict[str, Any]) -> None:
            study["search"] = {
                "status": "completed",
                "search_contract_fingerprint": self.contract["search_contract_fingerprint"],
                "selected_architecture": self.artifacts.relative(selected_path),
                "completed_generations": completed_generations,
                "stop_reason": stop_reason,
            }
            study["updated_at"] = utc_now()

        self.artifacts.update_study_manifest(update)
        return payload


def _normalized_weights(weights) -> tuple[float, float, float]:
    values = np.asarray(tuple(float(value) for value in weights), dtype=float)
    if values.shape != (3,) or np.any(values < 0.0) or values.sum() <= 0.0:
        raise ValueError("Winner weights must contain three non-negative values with positive sum.")
    values = values / values.sum()
    return tuple(float(value) for value in values)


def _pareto_mask(objectives: np.ndarray) -> np.ndarray:
    keep = np.ones(len(objectives), dtype=bool)
    for candidate in range(len(objectives)):
        if not keep[candidate]:
            continue
        for challenger in range(len(objectives)):
            if candidate == challenger:
                continue
            if np.all(objectives[challenger] <= objectives[candidate]) and np.any(
                objectives[challenger] < objectives[candidate]
            ):
                keep[candidate] = False
                break
    return keep


def select_winner(
    rows: list[dict[str, Any]],
    *,
    weights,
    penalty: float,
) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("status") == "valid"
        and all(
            np.isfinite(float(row[key])) and float(row[key]) < float(penalty)
            for key in ("obj_error", "obj_pdm", "obj_alarm_burden")
        )
    ]
    if not valid:
        raise ValueError("Search completed without a valid architecture candidate.")
    matrix = np.asarray(
        [[row["obj_error"], row["obj_pdm"], row["obj_alarm_burden"]] for row in valid],
        dtype=float,
    )
    pareto = [row for row, keep in zip(valid, _pareto_mask(matrix), strict=True) if bool(keep)]
    pareto_matrix = np.asarray(
        [[row["obj_error"], row["obj_pdm"], row["obj_alarm_burden"]] for row in pareto],
        dtype=float,
    )
    minima = pareto_matrix.min(axis=0)
    spans = pareto_matrix.max(axis=0) - minima
    normalized = np.zeros_like(pareto_matrix)
    nonzero = spans > 0.0
    normalized[:, nonzero] = (pareto_matrix[:, nonzero] - minima[nonzero]) / spans[nonzero]
    normalized_weights = np.asarray(_normalized_weights(weights), dtype=float)
    distances = np.sqrt((normalized**2 * normalized_weights).sum(axis=1))
    best = float(distances.min())
    ties = [
        (row, float(distance))
        for row, distance in zip(pareto, distances, strict=True)
        if abs(float(distance) - best) <= 1e-12
    ]
    ties.sort(
        key=lambda item: (
            float(item[0]["obj_alarm_burden"]),
            float(item[0]["obj_pdm"]),
            float(item[0]["obj_error"]),
            str(item[0]["created_at"]),
            int(item[0]["id"]),
        )
    )
    return {
        "candidate_count": len(rows),
        "valid_candidate_count": len(valid),
        "pareto_candidate_count": len(pareto),
        "candidate": ties[0][0],
        "distance": ties[0][1],
    }
