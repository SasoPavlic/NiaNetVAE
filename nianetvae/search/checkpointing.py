import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection

from log import Log


CHECKPOINT_SCHEMA_VERSION = "1.0"

_RUNTIME_METADATA_KEYS = {
    "checkpoint_file",
    "completed",
    "evaluations",
    "generation",
    "process_id",
    "updated_at",
}
_MUTABLE_BUDGET_KEYS = {
    "config_fingerprint",
    "state_fingerprint",
    "termination",
}


class SearchCheckpointError(RuntimeError):
    """Raised when checkpoint resume cannot proceed safely."""


@dataclass(frozen=True)
class SearchTerminationSpec:
    termination: Any
    contract: dict[str, Any]
    time_str: str
    time_seconds: int


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_state_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Return fields that must remain unchanged while resuming optimizer state."""
    excluded = _RUNTIME_METADATA_KEYS | _MUTABLE_BUDGET_KEYS
    return {
        key: _jsonable(value)
        for key, value in payload.items()
        if key not in excluded
    }


def _checkpoint_state_fingerprint(payload: dict[str, Any]) -> str:
    return _stable_hash(_checkpoint_state_contract(payload))


def parse_time_seconds(value: str | int | float) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError("empty time value")
    days = 0
    if "-" in text:
        day_part, text = text.split("-", 1)
        days = int(day_part)
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"time must be HH:MM:SS or D-HH:MM:SS, got {value!r}")
    hours, minutes, seconds = (int(part) for part in parts)
    if minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60 or hours < 0 or days < 0:
        raise ValueError(f"invalid time value {value!r}")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def resolve_search_termination(config: dict[str, Any]) -> SearchTerminationSpec:
    nia_search = dict(config.get("nia_search") or {})
    termination_cfg = dict(nia_search.get("termination") or {})

    if not termination_cfg:
        time_str = str(nia_search.get("time") or "01:00:00").strip()
        max_time = parse_time_seconds(time_str)
        return SearchTerminationSpec(
            termination=get_termination("time", max_time=max_time),
            contract={"type": "time", "time": time_str, "time_seconds": max_time},
            time_str=time_str,
            time_seconds=max_time,
        )

    term_type = str(termination_cfg.get("type") or "hybrid").strip().lower()
    time_str = str(termination_cfg.get("time") or nia_search.get("time") or "01:00:00").strip()
    max_time = parse_time_seconds(time_str)

    if term_type == "time":
        termination = get_termination("time", max_time=max_time)
        contract = {"type": "time", "time": time_str, "time_seconds": max_time}
    elif term_type in {"n_gen", "generation", "generations"}:
        n_gen = int(termination_cfg.get("n_gen"))
        if n_gen < 1:
            raise ValueError(f"nia_search.termination.n_gen must be >= 1, got {n_gen}.")
        termination = get_termination("n_gen", n_gen)
        contract = {"type": "n_gen", "n_gen": n_gen}
    elif term_type == "hybrid":
        n_gen = int(termination_cfg.get("n_gen"))
        if n_gen < 1:
            raise ValueError(f"nia_search.termination.n_gen must be >= 1, got {n_gen}.")
        termination = TerminationCollection(
            get_termination("n_gen", n_gen),
            get_termination("time", max_time=max_time),
        )
        contract = {
            "type": "hybrid",
            "n_gen": n_gen,
            "time": time_str,
            "time_seconds": max_time,
        }
    else:
        raise ValueError(
            "nia_search.termination.type must be one of: time, n_gen, hybrid. "
            f"Received {term_type!r}."
        )

    return SearchTerminationSpec(
        termination=termination,
        contract=contract,
        time_str=time_str,
        time_seconds=max_time,
    )


def _checkpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict((config.get("nia_search") or {}).get("checkpoint") or {})


def checkpoint_enabled(config: dict[str, Any]) -> bool:
    return bool(_checkpoint_config(config).get("enabled", False))


def _safe_label(value: Any) -> str:
    text = str(value or "dataset").strip() or "dataset"
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)


def resolve_checkpoint_dir(config: dict[str, Any], dataset_name: str) -> Path:
    logging_params = config.get("logging_params") or {}
    data_params = config.get("data_params") or {}
    export_root = Path(logging_params.get("model_export_dir") or "logs/per_maint_models")
    dataset = _safe_label(data_params.get("dataset_name") or dataset_name)
    regime = str(data_params.get("regime", "")).strip().lower()
    cycle_id = data_params.get("cycle_id")

    if regime == "per_maint" and cycle_id is not None:
        try:
            cycle_dir = f"cycle_{int(cycle_id):02d}"
        except Exception:
            cycle_dir = f"cycle_{_safe_label(cycle_id)}"
        return export_root / dataset / cycle_dir / "checkpoints"

    return export_root / dataset / "checkpoints" / _safe_label(dataset_name)


def checkpoint_paths(config: dict[str, Any], dataset_name: str) -> tuple[Path, Path]:
    checkpoint_dir = resolve_checkpoint_dir(config, dataset_name)
    return checkpoint_dir / "nsga3.dill", checkpoint_dir / "nsga3_meta.json"


def build_checkpoint_contract(
    *,
    config: dict[str, Any],
    dataset_name: str,
    algorithm_name: str,
    n_partitions: int,
    effective_population: int,
    objective_contract: dict[str, Any],
    selection_contract: dict[str, Any],
    termination_contract: dict[str, Any],
) -> dict[str, Any]:
    data_params = dict(config.get("data_params") or {})
    workflow = dict(config.get("workflow") or {})
    exp_params = dict(config.get("exp_params") or {})
    trainer_params = dict(config.get("trainer_params") or {})

    contract = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "algorithm": str(algorithm_name),
        "dataset_name": str(dataset_name),
        "data": {
            key: data_params.get(key)
            for key in (
                "dataset_name",
                "regime",
                "cycle_id",
                "seq_len",
                "n_features",
                "train_minutes",
                "post_train_minutes",
                "pre_maint_minutes",
                "rolling_window",
                "stride",
                "train_phases",
                "test_phases",
            )
            if key in data_params
        },
        "workflow_mode": workflow.get("mode"),
        "nsga3": {
            "n_partitions": int(n_partitions),
            "effective_population": int(effective_population),
        },
        "objective_contract": _jsonable(objective_contract),
        "winner_selection_contract": _jsonable(selection_contract),
        "termination": _jsonable(termination_contract),
        "training_policy": {
            "trainer_params": _jsonable(trainer_params),
            "exp_params": {
                key: exp_params.get(key)
                for key in (
                    "optimizer",
                    "learning_rate",
                    "weight_decay",
                    "kld_weight",
                    "manual_seed",
                )
                if key in exp_params
            },
        },
    }
    # Termination is a replaceable execution budget when an official pymoo
    # checkpoint resumes. The remaining contract identifies optimizer state.
    contract["state_fingerprint"] = _checkpoint_state_fingerprint(contract)
    contract["config_fingerprint"] = _stable_hash(contract)
    return contract


class NSGA3CheckpointCallback:
    def __init__(self, manager: "SearchCheckpointManager"):
        self.manager = manager
        self.last_saved_generation: int | None = None

    def __call__(self, algorithm):
        generation = int(getattr(algorithm, "n_gen", getattr(algorithm, "n_iter", 0)) or 0)
        if generation <= 0:
            return
        interval = self.manager.interval_generations
        if self.last_saved_generation is not None and generation - self.last_saved_generation < interval:
            return
        self.manager.save_algorithm(algorithm, completed=False)
        self.last_saved_generation = generation


class SearchCheckpointManager:
    def __init__(self, *, config: dict[str, Any], dataset_name: str, contract: dict[str, Any]):
        self.config = config
        self.dataset_name = dataset_name
        self.contract = dict(contract)
        self.cfg = _checkpoint_config(config)
        self.enabled = checkpoint_enabled(config)
        self.resume_mode = str(self.cfg.get("resume", "auto")).strip().lower()
        self.on_mismatch = str(self.cfg.get("on_mismatch", "fail")).strip().lower()
        self.keep_completed = bool(self.cfg.get("keep_completed", True))
        self.interval_generations = max(1, int(self.cfg.get("interval_generations", 1)))
        self.algorithm_path, self.meta_path = checkpoint_paths(config, dataset_name)

    def make_callback(self):
        if not self.enabled:
            return None
        return NSGA3CheckpointCallback(self)

    def ensure_ready(self) -> None:
        if self.enabled:
            self._load_dill()

    def _load_dill(self):
        try:
            import dill
        except ImportError as exc:
            raise SearchCheckpointError(
                "nia_search.checkpoint.enabled=true requires the 'dill' package. "
                "Install project dependencies before running checkpointed search."
            ) from exc
        return dill

    def _load_metadata(self) -> dict[str, Any] | None:
        if not self.meta_path.exists():
            return None
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _assert_compatible(self, metadata: dict[str, Any]) -> bool:
        expected = _checkpoint_state_fingerprint(self.contract)
        observed = _checkpoint_state_fingerprint(metadata)
        if expected == observed:
            expected_config = self.contract.get("config_fingerprint")
            observed_config = metadata.get("config_fingerprint")
            if expected_config != observed_config:
                Log.info(
                    "NSGA3_CHECKPOINT_BUDGET_UPDATED "
                    f"path={self.meta_path} "
                    f"previous_termination={metadata.get('termination')} "
                    f"current_termination={self.contract.get('termination')}"
                )
            return True
        message = (
            "NSGA3 checkpoint contract mismatch. "
            f"path={self.meta_path} expected_state={expected} observed_state={observed}"
        )
        if self.on_mismatch == "fail":
            raise SearchCheckpointError(message)
        Log.warning(f"{message}; ignoring checkpoint because on_mismatch={self.on_mismatch}")
        return False

    def load_algorithm(self, *, problem, termination, callback, n_jobs: int):
        if not self.enabled or self.resume_mode in {"false", "off", "none", "disabled"}:
            return None
        if not self.algorithm_path.exists() or not self.meta_path.exists():
            return None

        metadata = self._load_metadata() or {}
        if not self._assert_compatible(metadata):
            return None
        dill = self._load_dill()
        with self.algorithm_path.open("rb") as handle:
            algorithm = dill.load(handle)

        algorithm.problem = problem
        algorithm.termination = termination
        # pymoo's time termination measures from algorithm.start_time. A
        # serialized timestamp belongs to the previous process/job, so each
        # resumed invocation needs a fresh wall-clock budget origin.
        algorithm.start_time = time.time()
        if callback is not None:
            algorithm.callback = callback
        algorithm.n_jobs = n_jobs
        generation = getattr(algorithm, "n_gen", getattr(algorithm, "n_iter", None))
        Log.info(
            f"NSGA3_CHECKPOINT_LOADED path={self.algorithm_path} "
            f"generation={generation} resume_mode={self.resume_mode} "
            "job_timer_reset=true"
        )
        return algorithm

    def save_algorithm(self, algorithm, *, completed: bool = False) -> None:
        if not self.enabled:
            return
        dill = self._load_dill()
        self.algorithm_path.parent.mkdir(parents=True, exist_ok=True)

        generation = getattr(algorithm, "n_gen", getattr(algorithm, "n_iter", None))
        evaluations = getattr(getattr(algorithm, "evaluator", None), "n_eval", None)
        metadata = dict(self.contract)
        metadata.update(
            {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "checkpoint_file": self.algorithm_path.name,
                "completed": bool(completed),
                "generation": generation,
                "evaluations": evaluations,
                "process_id": os.getpid(),
            }
        )

        original_problem = getattr(algorithm, "problem", None)
        original_callback = getattr(algorithm, "callback", None)
        try:
            algorithm.problem = None
            algorithm.callback = None
            tmp_algorithm = self.algorithm_path.with_suffix(self.algorithm_path.suffix + ".tmp")
            with tmp_algorithm.open("wb") as handle:
                dill.dump(algorithm, handle)
            tmp_algorithm.replace(self.algorithm_path)
        finally:
            algorithm.problem = original_problem
            algorithm.callback = original_callback

        tmp_meta = self.meta_path.with_suffix(self.meta_path.suffix + ".tmp")
        tmp_meta.write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True), encoding="utf-8")
        tmp_meta.replace(self.meta_path)
        Log.info(
            f"NSGA3_CHECKPOINT_SAVED path={self.algorithm_path} "
            f"generation={generation} completed={str(completed).lower()}"
        )

    def finish(self, algorithm) -> None:
        if not self.enabled:
            return
        if self.keep_completed:
            self.save_algorithm(algorithm, completed=True)
            return
        for path in (self.algorithm_path, self.meta_path):
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:
                Log.warning(f"NSGA3_CHECKPOINT_CLEANUP_FAILED path={path} reason={exc}")
