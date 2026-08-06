"""Versioned, atomic artifact storage for one controlled MetroPT study."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pymoo
import sklearn
import torch

from .config import StudyConfig
from .dataloaders.metropt import PreparedMetroPTData

ARTIFACT_SCHEMA_VERSION = "1.0"


@contextmanager
def _exclusive_file_lock(path: Path, timeout_seconds: float = 120.0):
    """Serialize manifest read-modify-write operations across Slurm jobs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"\0")
        handle.flush()
    deadline = time.monotonic() + float(timeout_seconds)
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            while not locked:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out waiting for artifact lock: {path}") from None
                    time.sleep(0.05)
        else:
            import fcntl

            while not locked:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out waiting for artifact lock: {path}") from None
                    time.sleep(0.05)
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=_json_default,
        allow_nan=False,
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def atomic_write_csv(path: str | Path, frame: pd.DataFrame, *, index: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=index)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_contract_fingerprint() -> str:
    """Hash the installed runtime source independently of Git/container layout."""
    import hashlib

    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    sources = sorted(path for path in package_root.rglob("*.py") if "__pycache__" not in path.parts)
    if not sources:
        raise RuntimeError(f"No Python sources found under installed package: {package_root}")
    for source in sources:
        relative = source.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = source.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def git_revision(repository: str | Path) -> dict[str, Any]:
    root = Path(repository).resolve()

    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


class StudyArtifactStore:
    """Owns every durable output and validates the shared study contract."""

    def __init__(self, root: str | Path, study_id: str) -> None:
        self.root = Path(root).expanduser().resolve() / str(study_id)
        self.shared_dir = self.root / "shared"
        self.search_dir = self.root / "search"
        self.workflows_dir = self.root / "workflows"
        self.comparison_dir = self.root / "comparison"

    @classmethod
    def from_config(cls, config: StudyConfig) -> StudyArtifactStore:
        return cls(config.artifacts.root, config.artifacts.study_id)

    @property
    def manifest_path(self) -> Path:
        return self.root / "study_manifest.json"

    @property
    def prepared_cache_path(self) -> Path:
        return self.shared_dir / "prepared_metropt.joblib"

    @property
    def manifest_lock_path(self) -> Path:
        return self.root / ".study_manifest.lock"

    def workflow_dir(self, workflow_id: str) -> Path:
        return self.workflows_dir / str(workflow_id)

    def workflow_manifest_path(self, workflow_id: str) -> Path:
        return self.workflow_dir(workflow_id) / "run_manifest.json"

    def initialize(
        self,
        config: StudyConfig,
        prepared: PreparedMetroPTData,
        *,
        repository: str | Path,
        config_source: str | Path | None = None,
    ) -> dict[str, Any]:
        if self.manifest_path.exists():
            raise FileExistsError(f"Study is already initialized: {self.manifest_path}")
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(
                f"Refusing to initialize into non-empty study root without a manifest: {self.root}"
            )
        for directory in (
            self.shared_dir,
            self.search_dir,
            self.workflows_dir,
            self.comparison_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        atomic_write_json(self.shared_dir / "resolved_config.json", config.as_dict())
        atomic_write_json(self.shared_dir / "preprocessor.json", prepared.preprocessor.as_dict())
        data_contract = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "dataset_hash": prepared.dataset_hash,
            "feature_hash": prepared.feature_hash,
            "feature_names": list(prepared.feature_names),
            "schedule_hash": prepared.schedule_hash,
            "preprocessing_fingerprint": prepared.preprocessor.fingerprint,
            "calibration_reference_scope": config.calibration.reference_scope,
            "calibration_index_hash": _selected_index_hash(prepared.calibration_mask),
            "evaluation_index_hash": _selected_index_hash(prepared.evaluation_mask),
            "data_contract_fingerprint": prepared.data_contract_fingerprint,
            "baseline_rows": int(prepared.baseline_mask.sum()),
            "baseline_train_rows": int(prepared.baseline_train_mask.sum()),
            "baseline_validation_rows": int(prepared.baseline_validation_mask.sum()),
            "calibration_anchors": int(prepared.calibration_mask.sum()),
            "evaluation_rows": int(prepared.evaluation_mask.sum()),
            "cycles": [asdict(cycle) for cycle in prepared.cycles],
            "maintenance_events": [asdict(event) for event in prepared.events],
        }
        atomic_write_json(self.shared_dir / "data_contract.json", data_contract)
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "study_id": config.artifacts.study_id,
            "study_name": config.study_name,
            "status": "prepared",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "study_config_fingerprint": config.fingerprint(),
            "resolved_config_fingerprint": config.resolved_fingerprint(),
            "source_contract_fingerprint": source_contract_fingerprint(),
            "data_contract_fingerprint": prepared.data_contract_fingerprint,
            "preprocessing_fingerprint": prepared.preprocessor.fingerprint,
            "config_source": str(Path(config_source).resolve()) if config_source else None,
            "repository": git_revision(repository),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "joblib": joblib.__version__,
                "scikit_learn": sklearn.__version__,
                "pymoo": pymoo.__version__,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            },
            "workflows_expected": list(config.workflows),
            "workflows_completed": [],
            "paths_relative_to": "study_root",
        }
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def record_execution_config(self, config: StudyConfig) -> None:
        """Record the current replaceable search budget without changing study identity."""
        atomic_write_json(self.shared_dir / "resolved_config.json", config.as_dict())

        def update(manifest: dict[str, Any]) -> None:
            if manifest.get("study_config_fingerprint") != config.fingerprint():
                raise ValueError("Cannot record execution config for a different study contract.")
            manifest["resolved_config_fingerprint"] = config.resolved_fingerprint()
            manifest["search_execution_budget"] = {
                "max_generations": config.search.max_generations,
                "max_time": config.search.max_time,
            }
            manifest["updated_at"] = utc_now()

        self.update_study_manifest(update)

    def save_prepared_cache(self, prepared: PreparedMetroPTData) -> Path:
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.prepared_cache_path.with_suffix(".joblib.tmp")
        joblib.dump(prepared, temporary, compress=0)
        os.replace(temporary, self.prepared_cache_path)
        cache_record = {
            "path": self.relative(self.prepared_cache_path),
            "sha256": sha256_file(self.prepared_cache_path),
            "data_contract_fingerprint": prepared.data_contract_fingerprint,
        }

        def update(manifest: dict[str, Any]) -> None:
            manifest["prepared_cache"] = cache_record
            manifest["updated_at"] = utc_now()

        self.update_study_manifest(update)
        return self.prepared_cache_path

    def load_prepared_cache(self) -> PreparedMetroPTData:
        if not self.prepared_cache_path.is_file():
            raise FileNotFoundError(f"Missing prepared-data cache: {self.prepared_cache_path}")
        manifest = read_json(self.manifest_path)
        cache = manifest.get("prepared_cache") or {}
        expected_hash = cache.get("sha256")
        if not expected_hash or sha256_file(self.prepared_cache_path) != expected_hash:
            raise ValueError("Prepared-data cache hash does not match the study manifest.")
        prepared = joblib.load(self.prepared_cache_path)
        if not isinstance(prepared, PreparedMetroPTData):
            raise ValueError("Prepared-data cache has an unexpected object type.")
        if prepared.data_contract_fingerprint != cache.get("data_contract_fingerprint"):
            raise ValueError("Prepared-data cache contract does not match its manifest record.")
        return prepared

    def assert_initialized(
        self, config: StudyConfig, prepared: PreparedMetroPTData
    ) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Study is not prepared at {self.root}. Run the prepare command first."
            )
        manifest = read_json(self.manifest_path)
        _require_schema(manifest, self.manifest_path)
        checks = {
            "study_config_fingerprint": config.fingerprint(),
            "source_contract_fingerprint": source_contract_fingerprint(),
            "data_contract_fingerprint": prepared.data_contract_fingerprint,
            "preprocessing_fingerprint": prepared.preprocessor.fingerprint,
        }
        for key, expected in checks.items():
            observed = manifest.get(key)
            if observed != expected:
                raise ValueError(
                    f"Prepared study contract mismatch for {key}: "
                    f"expected {expected}, observed {observed}. "
                    "Use a new study_id when any controlled constant or input changes."
                )
        return manifest

    def begin_workflow(
        self,
        workflow_id: str,
        *,
        config: StudyConfig,
        prepared: PreparedMetroPTData,
        architecture: dict[str, Any],
    ) -> dict[str, Any]:
        directory = self.workflow_dir(workflow_id)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "study_id": config.artifacts.study_id,
            "workflow_id": workflow_id,
            "status": "running",
            "started_at": utc_now(),
            "completed_at": None,
            "study_config_fingerprint": config.fingerprint(),
            "resolved_config_fingerprint": config.resolved_fingerprint(),
            "source_contract_fingerprint": source_contract_fingerprint(),
            "data_contract_fingerprint": prepared.data_contract_fingerprint,
            "preprocessing_fingerprint": prepared.preprocessor.fingerprint,
            "calibration_reference_scope": config.calibration.reference_scope,
            "selection_scope": config.evaluation.selection_scope,
            "architecture": architecture,
            "cycle_lineage": [],
            "outputs": {},
            "output_sha256": {},
        }
        atomic_write_json(self.workflow_manifest_path(workflow_id), payload)
        return payload

    def complete_workflow(
        self,
        workflow_id: str,
        manifest: dict[str, Any],
        *,
        outputs: dict[str, str],
        cycle_lineage: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        completed = dict(manifest)
        completed.pop("error", None)
        completed.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "cycle_lineage": list(cycle_lineage),
                "outputs": dict(outputs),
                "output_sha256": {
                    label: sha256_file(self.root / relative) for label, relative in outputs.items()
                },
            }
        )
        atomic_write_json(self.workflow_manifest_path(workflow_id), completed)

        def update(study: dict[str, Any]) -> None:
            finished = sorted({*study.get("workflows_completed", []), workflow_id})
            study["workflows_completed"] = finished
            study["updated_at"] = utc_now()
            expected = set(study.get("workflows_expected", []))
            if expected and expected.issubset(finished):
                study["status"] = "workflows_completed"

        self.update_study_manifest(update)
        return completed

    def mark_failed(self, workflow_id: str, error: BaseException) -> None:
        path = self.workflow_manifest_path(workflow_id)
        if not path.exists():
            return
        manifest = read_json(path)
        manifest.update(
            {
                "status": "failed",
                "completed_at": utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        atomic_write_json(path, manifest)

    def validate_study(self, expected_workflows: Iterable[str]) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing study manifest: {self.manifest_path}")
        study = read_json(self.manifest_path)
        _require_schema(study, self.manifest_path)
        data_contract_path = self.shared_dir / "data_contract.json"
        if not data_contract_path.is_file():
            raise FileNotFoundError(f"Missing data contract: {data_contract_path}")
        data_contract = read_json(data_contract_path)
        expected = tuple(str(value) for value in expected_workflows)
        errors: list[str] = []
        workflow_manifests: dict[str, dict[str, Any]] = {}
        if tuple(study.get("workflows_expected", [])) != expected:
            errors.append("study manifest workflow set/order is inconsistent")
        cache = study.get("prepared_cache") or {}
        if not self.prepared_cache_path.is_file() or sha256_file(
            self.prepared_cache_path
        ) != cache.get("sha256"):
            errors.append("prepared-data cache is missing or has an invalid hash")
        if "nianetvae_per_maintenance" in expected:
            selected_architecture = self.search_dir / "selected_architecture.json"
            search_manifest = self.search_dir / "search_manifest.json"
            if not selected_architecture.is_file() or not search_manifest.is_file():
                errors.append("NiaNetVAE workflow is missing fresh shared-core search artifacts")
            else:
                selected_payload = read_json(selected_architecture)
                search_payload = read_json(search_manifest)
                if search_payload.get("status") != "completed":
                    errors.append("shared-core architecture search is not completed")
                if selected_payload.get("search_contract_fingerprint") != search_payload.get(
                    "search_contract_fingerprint"
                ):
                    errors.append("search manifest and selected architecture contracts differ")
                required_search_outputs = {
                    "selected_architecture",
                    "candidates_csv",
                    "candidate_database",
                    "algorithm_checkpoint",
                    "checkpoint_metadata",
                }
                search_outputs = search_payload.get("outputs", {})
                if required_search_outputs - set(search_outputs):
                    errors.append("shared-core search manifest has incomplete outputs")
                for label, relative in search_outputs.items():
                    output = self.root / str(relative)
                    expected_hash = search_payload.get("output_sha256", {}).get(label)
                    if (
                        not output.is_file()
                        or not expected_hash
                        or sha256_file(output) != expected_hash
                    ):
                        errors.append(f"shared-core search output hash mismatch: {label}")
        for workflow_id in expected:
            path = self.workflow_manifest_path(workflow_id)
            if not path.is_file():
                errors.append(f"missing workflow manifest: {workflow_id}")
                continue
            payload = read_json(path)
            workflow_manifests[workflow_id] = payload
            try:
                _require_schema(payload, path)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if payload.get("status") != "completed":
                errors.append(f"workflow {workflow_id} status={payload.get('status')!r}")
            for key in (
                "study_config_fingerprint",
                "source_contract_fingerprint",
                "data_contract_fingerprint",
                "preprocessing_fingerprint",
            ):
                if payload.get(key) != study.get(key):
                    errors.append(f"workflow {workflow_id} has inconsistent {key}")
            for label, relative in payload.get("outputs", {}).items():
                output_path = self.root / relative
                if not output_path.is_file():
                    errors.append(f"workflow {workflow_id} missing output {label}: {relative}")
                    continue
                expected_hash = payload.get("output_sha256", {}).get(label)
                if not expected_hash or sha256_file(output_path) != expected_hash:
                    errors.append(f"workflow {workflow_id} output hash mismatch: {label}")

            lineage = payload.get("cycle_lineage", [])
            if len(lineage) != len(data_contract.get("cycles", [])):
                errors.append(f"workflow {workflow_id} has incomplete cycle lineage")
            prediction_total = 0
            for cycle_result in lineage:
                prediction_total += int(cycle_result.get("prediction_count", 0))
                for label in ("checkpoint", "predictions", "calibration"):
                    artifact = self.root / str(cycle_result.get(label, ""))
                    expected_hash = cycle_result.get(f"{label}_sha256")
                    if (
                        not artifact.is_file()
                        or not expected_hash
                        or sha256_file(artifact) != expected_hash
                    ):
                        errors.append(
                            f"workflow {workflow_id} cycle {cycle_result.get('cycle_id')} "
                            f"{label} hash mismatch"
                        )
                calibration = self.root / str(cycle_result.get("calibration", ""))
                if not calibration.is_file():
                    errors.append(
                        f"workflow {workflow_id} cycle {cycle_result.get('cycle_id')} "
                        "missing calibration"
                    )
                    continue
                calibration_payload = read_json(calibration)
                if calibration_payload.get("reference_index_hash") != data_contract.get(
                    "calibration_index_hash"
                ):
                    errors.append(
                        f"workflow {workflow_id} cycle {cycle_result.get('cycle_id')} "
                        "changed calibration timestamps"
                    )

            if prediction_total != int(data_contract.get("evaluation_rows", -1)):
                errors.append(f"workflow {workflow_id} has a different evaluation population size")
            predictions_relative = payload.get("outputs", {}).get("predictions")
            if predictions_relative and (self.root / predictions_relative).is_file():
                timestamps = pd.read_csv(
                    self.root / predictions_relative,
                    usecols=["timestamp"],
                    parse_dates=["timestamp"],
                )["timestamp"]
                if timestamps.duplicated().any():
                    errors.append(f"workflow {workflow_id} has duplicate prediction timestamps")
                if _timestamp_index_hash(pd.DatetimeIndex(timestamps)) != data_contract.get(
                    "evaluation_index_hash"
                ):
                    errors.append(f"workflow {workflow_id} changed evaluation timestamps")

            selected_relative = payload.get("outputs", {}).get("selected_operating_point")
            if selected_relative and (self.root / selected_relative).is_file():
                selected = read_json(self.root / selected_relative)
                if selected.get("selection_scope") != "retrospective_full_timeline":
                    errors.append(f"workflow {workflow_id} has an invalid selection scope")

            metrics_relative = payload.get("outputs", {}).get("event_metrics")
            if metrics_relative and (self.root / metrics_relative).is_file():
                metrics = read_json(self.root / metrics_relative)
                required_metrics = {
                    "event_scores",
                    "ttd",
                    "first_alarm_accuracy",
                    "far",
                    "coverage",
                    "mtia",
                    "nab",
                    "pr_leadtime",
                }
                if required_metrics - set(metrics):
                    errors.append(f"workflow {workflow_id} has incomplete event metrics")

        comparison_manifest_path = self.comparison_dir / "comparison_manifest.json"
        if not comparison_manifest_path.is_file():
            errors.append("missing cross-workflow comparison manifest")
        else:
            comparison = read_json(comparison_manifest_path)
            try:
                _require_schema(comparison, comparison_manifest_path)
            except ValueError as exc:
                errors.append(str(exc))
            if tuple(comparison.get("workflows", [])) != expected:
                errors.append("comparison manifest workflow set/order is inconsistent")
            if comparison.get("study_id") != study.get("study_id"):
                errors.append("comparison manifest belongs to another study")
            for label, relative in comparison.get("outputs", {}).items():
                output = self.root / relative
                expected_hash = comparison.get("output_sha256", {}).get(label)
                if (
                    not output.is_file()
                    or not expected_hash
                    or sha256_file(output) != expected_hash
                ):
                    errors.append(f"comparison output hash mismatch: {label}")
        report = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "validated_at": utc_now(),
            "study_id": study.get("study_id"),
            "valid": not errors,
            "errors": errors,
            "expected_workflows": list(expected),
            "completed_workflows": sorted(workflow_manifests),
        }
        atomic_write_json(self.root / "validation_report.json", report)
        if errors:
            raise ValueError("Study validation failed: " + "; ".join(errors))

        def update(current: dict[str, Any]) -> None:
            current["status"] = "validated"
            current["updated_at"] = utc_now()

        self.update_study_manifest(update)
        return report

    def update_study_manifest(
        self,
        updater: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        """Atomically update the shared manifest without losing concurrent fields."""
        with _exclusive_file_lock(self.manifest_lock_path):
            manifest = read_json(self.manifest_path)
            updater(manifest)
            atomic_write_json(self.manifest_path, manifest)
            return manifest

    @contextmanager
    def exclusive_lock(self, name: str, *, timeout_seconds: float = 120.0):
        """Prevent duplicate long-running work for one logical artifact target."""
        normalized = str(name).strip()
        if not normalized or not normalized.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Artifact lock names may contain only letters, digits, '-' and '_'.")
        with _exclusive_file_lock(
            self.root / f".{normalized}.lock",
            timeout_seconds=timeout_seconds,
        ):
            yield

    def relative(self, path: str | Path) -> str:
        return Path(path).resolve().relative_to(self.root).as_posix()


def _selected_index_hash(mask: pd.Series) -> str:
    return _timestamp_index_hash(mask[mask.astype(bool)].index)


def _timestamp_index_hash(index: pd.Index) -> str:
    import hashlib

    timestamps = [pd.Timestamp(value).isoformat() for value in index]
    encoded = json.dumps(timestamps, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_schema(payload: dict[str, Any], path: Path) -> None:
    observed = payload.get("schema_version")
    if observed != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported artifact schema in {path}: expected "
            f"{ARTIFACT_SCHEMA_VERSION}, observed {observed!r}."
        )
