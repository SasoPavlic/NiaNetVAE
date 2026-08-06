"""Official pymoo algorithm checkpointing with immutable-state validation."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import dill

from ..artifacts import atomic_write_json, read_json


class SearchCheckpointManager:
    def __init__(
        self,
        directory: str | Path,
        *,
        contract: dict[str, Any],
        interval_generations: int,
    ) -> None:
        self.directory = Path(directory)
        self.algorithm_path = self.directory / "nsga3.dill"
        self.metadata_path = self.directory / "nsga3_checkpoint.json"
        self.contract = dict(contract)
        self.interval_generations = max(1, int(interval_generations))
        self.last_saved_generation: int | None = None

    def callback(self):
        manager = self

        class Callback:
            def __call__(self, algorithm) -> None:
                generation = int(getattr(algorithm, "n_gen", 0) or 0)
                if generation <= 0:
                    return
                if (
                    manager.last_saved_generation is not None
                    and generation - manager.last_saved_generation < manager.interval_generations
                ):
                    return
                manager.save(algorithm, completed=False)
                manager.last_saved_generation = generation

        return Callback()

    def load(self, *, problem, termination, callback):
        if not self.algorithm_path.exists() and not self.metadata_path.exists():
            return None
        if not self.algorithm_path.exists() or not self.metadata_path.exists():
            raise ValueError(
                "Incomplete NSGA-III checkpoint: algorithm and metadata must both exist."
            )
        metadata = read_json(self.metadata_path)
        observed = metadata.get("search_contract_fingerprint")
        expected = self.contract["search_contract_fingerprint"]
        if observed != expected:
            raise ValueError(
                "NSGA-III checkpoint contract mismatch. "
                "Use a new study_id instead of mixing searches."
            )
        with self.algorithm_path.open("rb") as handle:
            algorithm = dill.load(handle)
        algorithm.problem = problem
        algorithm.termination = termination
        algorithm.callback = callback
        # Time termination is a per-job execution budget. A serialized timer
        # must never make a legitimate continuation terminate immediately.
        algorithm.start_time = time.time()
        if hasattr(algorithm, "has_terminated"):
            algorithm.has_terminated = False
        return algorithm

    def save(self, algorithm, *, completed: bool) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        generation = int(getattr(algorithm, "n_gen", 0) or 0)
        evaluations = int(getattr(getattr(algorithm, "evaluator", None), "n_eval", 0) or 0)
        original_problem = getattr(algorithm, "problem", None)
        original_callback = getattr(algorithm, "callback", None)
        temporary = self.algorithm_path.with_suffix(".dill.tmp")
        try:
            algorithm.problem = None
            algorithm.callback = None
            with temporary.open("wb") as handle:
                dill.dump(algorithm, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.algorithm_path)
        finally:
            algorithm.problem = original_problem
            algorithm.callback = original_callback
        metadata = {
            **self.contract,
            "completed": bool(completed),
            "next_generation": generation,
            "completed_generations": (max(0, generation - 1) if completed else generation),
            "evaluations": evaluations,
            "process_id": os.getpid(),
            "checkpoint_file": self.algorithm_path.name,
        }
        atomic_write_json(self.metadata_path, metadata)
