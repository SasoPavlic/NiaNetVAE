"""Explicit, hash-verified migration of compatible architecture-search evidence."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    StudyArtifactStore,
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now,
)
from ..config import StudyConfig
from ..dataloaders.metropt import PreparedMetroPTData
from .engine import search_contract

SEARCH_RUNTIME_PATTERNS = (
    "config.py",
    "contracts.py",
    "dataloaders/**/*.py",
    "evaluation/calibration.py",
    "evaluation/risk.py",
    "models/**/*.py",
    "search/__init__.py",
    "search/checkpointing.py",
    "search/engine.py",
    "search/genome.py",
    "search/objectives.py",
    "search/storage.py",
    "training/**/*.py",
)

_COMPATIBILITY_KEYS = (
    "schema_version",
    "data_contract_fingerprint",
    "n_partitions",
    "genome_dimension",
    "candidate_min_epochs",
    "candidate_max_epochs",
    "training",
    "objectives",
    "winner_weights",
    "invalid_penalty",
    "replaceable_budget",
)


def search_runtime_sources(package_root: str | Path | None = None) -> tuple[Path, ...]:
    root = (
        Path(package_root).expanduser().resolve()
        if package_root is not None
        else Path(__file__).resolve().parents[1]
    )
    sources = {
        path.resolve()
        for pattern in SEARCH_RUNTIME_PATTERNS
        for path in root.glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    }
    if not sources:
        raise RuntimeError(f"No architecture-search runtime sources found under {root}.")
    return tuple(sorted(sources, key=lambda path: path.relative_to(root).as_posix()))


def search_runtime_fingerprint(package_root: str | Path | None = None) -> str:
    """Hash only code that can affect candidate training, objectives, or selection."""
    root = (
        Path(package_root).expanduser().resolve()
        if package_root is not None
        else Path(__file__).resolve().parents[1]
    )
    digest = hashlib.sha256()
    for source in search_runtime_sources(root):
        relative = source.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = source.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def migrate_search_artifacts(
    config: StudyConfig,
    prepared: PreparedMetroPTData,
    store: StudyArtifactStore,
    *,
    donor_study_root: str | Path,
    donor_search_runtime_fingerprint: str,
) -> dict[str, Any]:
    """Import a completed search only when its scientific runtime is byte-identical."""
    store.assert_initialized(config, prepared)
    donor_root = Path(donor_study_root).expanduser().resolve()
    if donor_root == store.root:
        raise ValueError("Search migration donor and target study roots must differ.")
    donor_manifest_path = donor_root / "search" / "search_manifest.json"
    donor_selected_path = donor_root / "search" / "selected_architecture.json"
    if not donor_manifest_path.is_file() or not donor_selected_path.is_file():
        raise FileNotFoundError("Donor study is missing completed architecture-search artifacts.")
    if (store.search_dir / "search_manifest.json").exists() or (
        store.search_dir / "selected_architecture.json"
    ).exists():
        raise FileExistsError("Target study already contains architecture-search artifacts.")

    donor_manifest = read_json(donor_manifest_path)
    donor_selected = read_json(donor_selected_path)
    if donor_manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Donor search uses an unsupported artifact schema.")
    if donor_manifest.get("status") != "completed":
        raise ValueError("Donor architecture search is not completed.")
    donor_study_id = str(donor_manifest.get("study_id", ""))
    if not donor_study_id or donor_study_id == config.artifacts.study_id:
        raise ValueError("Search migration requires a distinct donor study_id.")
    if donor_selected.get("study_id") != donor_study_id:
        raise ValueError("Donor selected architecture belongs to another study.")
    if donor_selected.get("search_contract_fingerprint") != donor_manifest.get(
        "search_contract_fingerprint"
    ):
        raise ValueError("Donor selected architecture and search manifest contracts differ.")

    current_runtime_fingerprint = search_runtime_fingerprint()
    if donor_search_runtime_fingerprint != current_runtime_fingerprint:
        raise ValueError(
            "Donor and target architecture-search runtime fingerprints differ; "
            "a fresh search is required."
        )

    target_contract = search_contract(config, prepared)
    compatibility_errors = [
        key for key in _COMPATIBILITY_KEYS if donor_manifest.get(key) != target_contract.get(key)
    ]
    if compatibility_errors:
        raise ValueError(
            "Donor search contract is incompatible with the target study: "
            + ", ".join(compatibility_errors)
        )

    required_outputs = {
        "selected_architecture",
        "candidates_csv",
        "candidate_database",
        "algorithm_checkpoint",
        "checkpoint_metadata",
    }
    donor_outputs = donor_manifest.get("outputs", {})
    if required_outputs - set(donor_outputs):
        raise ValueError("Donor search manifest has incomplete outputs.")

    copied_outputs: dict[str, dict[str, str]] = {}
    target_outputs: dict[str, str] = {}
    migration_root = store.search_dir / "migration_source"
    for label, relative in donor_outputs.items():
        source = donor_root / str(relative)
        expected_hash = donor_manifest.get("output_sha256", {}).get(label)
        if not source.is_file() or not expected_hash or sha256_file(source) != expected_hash:
            raise ValueError(f"Donor search output hash mismatch: {label}")
        source_relative = Path(str(relative))
        try:
            within_search = source_relative.relative_to("search")
        except ValueError as exc:
            raise ValueError(f"Donor search output escapes its search directory: {label}") from exc
        destination = migration_root / within_search
        _atomic_copy(source, destination)
        copied_relative = store.relative(destination)
        copied_outputs[label] = {
            "source_relative": source_relative.as_posix(),
            "copied_relative": copied_relative,
            "sha256": expected_hash,
        }
        if label != "selected_architecture":
            target_outputs[label] = copied_relative

    migration = {
        "method": "verified_search_runtime_v1",
        "migrated_at": utc_now(),
        "donor_study_id": donor_study_id,
        "donor_study_root": str(donor_root),
        "donor_source_contract_fingerprint": donor_manifest.get("source_contract_fingerprint"),
        "donor_search_contract_fingerprint": donor_manifest.get("search_contract_fingerprint"),
        "donor_search_runtime_fingerprint": donor_search_runtime_fingerprint,
        "target_search_runtime_fingerprint": current_runtime_fingerprint,
        "donor_outputs": copied_outputs,
    }
    selected = dict(donor_selected)
    selected.update(
        {
            "study_id": config.artifacts.study_id,
            "search_contract_fingerprint": target_contract["search_contract_fingerprint"],
            "migration": migration,
        }
    )
    selected_path = atomic_write_json(store.search_dir / "selected_architecture.json", selected)
    target_outputs["selected_architecture"] = store.relative(selected_path)

    manifest = {
        **target_contract,
        "status": "completed",
        "execution_mode": "verified_search_migration",
        "completed_at": utc_now(),
        "candidate_count": donor_manifest.get("candidate_count"),
        "valid_candidate_count": donor_manifest.get("valid_candidate_count"),
        "pareto_candidate_count": donor_manifest.get("pareto_candidate_count"),
        "selected_architecture": store.relative(selected_path),
        "execution_budget": donor_manifest.get("execution_budget"),
        "pymoo_result": donor_manifest.get("pymoo_result"),
        "migration": migration,
        "outputs": target_outputs,
        "output_sha256": {
            label: sha256_file(store.root / relative) for label, relative in target_outputs.items()
        },
    }
    atomic_write_json(store.search_dir / "search_manifest.json", manifest)

    def update(study: dict[str, Any]) -> None:
        study["search"] = {
            "status": "completed",
            "execution_mode": "verified_search_migration",
            "search_contract_fingerprint": target_contract["search_contract_fingerprint"],
            "selected_architecture": store.relative(selected_path),
            "donor_study_id": donor_study_id,
            "donor_search_runtime_fingerprint": donor_search_runtime_fingerprint,
        }
        study["updated_at"] = utc_now()

    store.update_study_manifest(update)
    return manifest


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
