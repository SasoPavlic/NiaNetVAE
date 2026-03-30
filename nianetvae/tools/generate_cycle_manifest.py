#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

from nianetvae.dataloaders.metropt_dataloader import MetroPTDataLoader


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=yaml.Loader)
    return data or {}


def _load_merged_config(config_path: Path) -> dict:
    config = _load_yaml(config_path)
    seen = set()
    base_dir = config_path.parent

    for _ in range(5):
        dataset_cfg = (config.get("dataset") or {}).get("config_file")
        if not dataset_cfg:
            break

        candidate = Path(str(dataset_cfg))
        if not candidate.is_absolute():
            # Primary behavior: resolve relative to the config file directory.
            candidate_resolved = (base_dir / candidate).resolve()
            if candidate_resolved.exists():
                candidate = candidate_resolved
            else:
                # Backward compatibility for config paths authored relative to project root.
                # If base_dir resolution fails, try parent directory resolution.
                candidate = (base_dir.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        candidate_key = str(candidate)
        if candidate_key in seen:
            raise ValueError(f"Recursive dataset config reference detected: {candidate}")
        seen.add(candidate_key)

        dataset_payload = _load_yaml(candidate)
        config.update(dataset_payload)
        next_cfg = (dataset_payload.get("dataset") or {}).get("config_file")
        if not next_cfg:
            if isinstance(config.get("dataset"), dict):
                config["dataset"].pop("config_file", None)
            break

    shared_data_loader_params = config.get("data_loader_params", {})
    config.setdefault("data_params", {})
    config["data_params"].update(shared_data_loader_params)
    return config


def _parse_cycles(spec: str) -> list[int]:
    out = []
    for part in str(spec).split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            out.extend(list(range(int(start), int(end) + 1)))
        else:
            out.append(int(token))
    return sorted(set(out))


def _config_fingerprint(config: dict) -> str:
    data_params = config.get("data_params", {})
    payload = {
        "dataset_name": data_params.get("dataset_name"),
        "data_path": data_params.get("data_path"),
        "rolling_window": data_params.get("rolling_window"),
        "seq_len": data_params.get("seq_len"),
        "stride": data_params.get("stride"),
        "train_minutes": data_params.get("train_minutes"),
        "post_train_minutes": data_params.get("post_train_minutes"),
        "pre_maint_minutes": data_params.get("pre_maint_minutes"),
        "train_phases": data_params.get("train_phases"),
        "test_phases": data_params.get("test_phases"),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _cycle_trainable(config: dict, cycle_id: int) -> tuple[bool, str | None]:
    probe = copy.deepcopy(config)
    probe.setdefault("data_params", {})
    probe["data_params"]["regime"] = "per_maint"
    probe["data_params"]["cycle_id"] = int(cycle_id)
    try:
        datamodule = MetroPTDataLoader(**probe["data_params"])
        datamodule.setup()
        return True, None
    except Exception as exc:
        return False, str(exc)


def _read_meta(meta_path: Path) -> dict:
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _cycle_key(cycle_id: int) -> str:
    return f"{int(cycle_id):02d}"


def _extract_meta_provenance(metadata: dict) -> dict:
    provenance = metadata.get("provenance") if isinstance(metadata.get("provenance"), dict) else {}
    experiment_mode = provenance.get("experiment_mode") or metadata.get("workflow_mode")
    source_cycle = provenance.get("source_cycle")
    seed_source = provenance.get("seed_source")
    search_init_mode = provenance.get("search_init_mode")
    warm_start = provenance.get("warm_start")
    out = {
        "experiment_mode": experiment_mode,
        "source_cycle": source_cycle,
        "seed_source": seed_source,
        "search_init_mode": search_init_mode,
    }
    if warm_start is not None:
        out["warm_start"] = warm_start
    return out


def _validate_manifest_contract(manifest: dict) -> None:
    if "cycles" not in manifest or not isinstance(manifest["cycles"], dict):
        raise ValueError("Manifest contract violation: missing top-level 'cycles' object.")

    allowed_statuses = {"trained", "alias", "missing"}
    for key, entry in manifest["cycles"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest contract violation: cycle {key} entry must be an object.")
        status = str(entry.get("status", "")).strip().lower()
        if status not in allowed_statuses:
            raise ValueError(
                f"Manifest contract violation: cycle {key} has unsupported status={status!r}."
            )
        if status == "trained":
            if not entry.get("model_path") or not entry.get("meta_path"):
                raise ValueError(
                    f"Manifest contract violation: cycle {key} status=trained requires model_path and meta_path."
                )
        if status == "alias":
            if entry.get("alias_to") is None:
                raise ValueError(
                    f"Manifest contract violation: cycle {key} status=alias requires alias_to."
                )


def build_manifest(config: dict, export_root: Path, cycles: list[int], paths_relative_to: Path) -> dict:
    dataset_name = str(config.get("data_params", {}).get("dataset_name", "dataset")).strip() or "dataset"
    regime = str(config.get("data_params", {}).get("regime", "per_maint")).strip().lower() or "per_maint"
    workflow_mode = str((config.get("workflow") or {}).get("mode", "")).strip().lower() or None
    seed_source = (config.get("exp_params") or {}).get("manual_seed")
    dataset_root = export_root / dataset_name
    paths_relative_to = paths_relative_to.resolve()

    manifest = {
        "schema_version": "1.0",
        "dataset": dataset_name,
        "regime": regime,
        "workflow_mode": workflow_mode,
        "seed_source": seed_source,
        "generated_at": datetime.now().isoformat(),
        "config_fingerprint": _config_fingerprint(config),
        "paths_relative_to": "manifest_directory",
        "active_cycle_order": [_cycle_key(cycle_id) for cycle_id in cycles],
        "cycles": {},
        "notes": [],
    }

    last_trained_cycle: int | None = None
    alias_cycles: list[str] = []
    missing_cycles: list[str] = []

    for cycle_id in cycles:
        key = _cycle_key(cycle_id)
        cycle_dir = dataset_root / f"cycle_{key}"
        model_path = cycle_dir / "model.pt"
        meta_path = cycle_dir / "model_meta.json"
        summary_path = cycle_dir / "search_summary.json"

        has_artifacts = model_path.exists() and meta_path.exists()
        trainable, trainable_error = _cycle_trainable(config, cycle_id)

        if has_artifacts:
            metadata = _read_meta(meta_path)
            summary_payload = _read_meta(summary_path) if summary_path.exists() else {}
            status = "trained"
            artifact_dir_rel = os.path.relpath(cycle_dir, paths_relative_to).replace("\\", "/")
            model_path_rel = os.path.relpath(model_path, paths_relative_to).replace("\\", "/")
            meta_path_rel = os.path.relpath(meta_path, paths_relative_to).replace("\\", "/")
            summary_path_rel = (
                os.path.relpath(summary_path, paths_relative_to).replace("\\", "/")
                if summary_path.exists()
                else None
            )
            entry = {
                "cycle_id": int(cycle_id),
                "status": status,
                "trainable": bool(trainable),
                "artifact_dir": artifact_dir_rel,
                "model_path": model_path_rel,
                "meta_path": meta_path_rel,
                "summary_path": summary_path_rel,
                "hash_id": metadata.get("hash_id"),
                "run_uuid": metadata.get("run_uuid"),
                "created_at": metadata.get("created_at"),
            }
            provenance = _extract_meta_provenance(metadata)
            if provenance.get("source_cycle") is None:
                search_payload = summary_payload.get("search", {}) if isinstance(summary_payload, dict) else {}
                inferred_source_cycle = search_payload.get("source_cycle_id")
                if inferred_source_cycle is None:
                    warm_start_payload = search_payload.get("warm_start", {})
                    if isinstance(warm_start_payload, dict):
                        inferred_source_cycle = warm_start_payload.get("source_cycle_id")
                if inferred_source_cycle is not None:
                    provenance["source_cycle"] = inferred_source_cycle
            entry.update(provenance)
            last_trained_cycle = int(cycle_id)
        elif not trainable:
            if last_trained_cycle is not None:
                status = "alias"
                alias_cycles.append(key)
                entry = {
                    "cycle_id": int(cycle_id),
                    "status": status,
                    "trainable": False,
                    "alias_to": int(last_trained_cycle),
                    "alias_to_key": _cycle_key(last_trained_cycle),
                    "source_cycle": int(last_trained_cycle),
                    "reason": trainable_error or "non_trainable_cycle",
                }
            else:
                status = "missing"
                missing_cycles.append(key)
                entry = {
                    "cycle_id": int(cycle_id),
                    "status": status,
                    "trainable": False,
                    "reason": "non_trainable_cycle_without_prior_trained_alias",
                    "detail": trainable_error,
                }
        else:
            status = "missing"
            missing_cycles.append(key)
            entry = {
                "cycle_id": int(cycle_id),
                "status": status,
                "trainable": True,
                "reason": "trainable_cycle_missing_artifact",
                }

        entry["experiment_mode"] = entry.get("experiment_mode") or workflow_mode
        if entry.get("seed_source") is None:
            entry["seed_source"] = seed_source
        manifest["cycles"][key] = entry

    if alias_cycles:
        manifest["notes"].append(
            f"Alias cycles (empty-gap/non-trainable) mapped to previous trained cycle: {', '.join(alias_cycles)}"
        )
    if missing_cycles:
        manifest["notes"].append(
            f"Missing cycle artifacts: {', '.join(missing_cycles)}"
        )
    _validate_manifest_contract(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate MetroPT per-maint cycle manifest from exported NiaNetVAE artifacts.")
    parser.add_argument(
        "--config",
        default="configs/main_config.yaml",
        help="Path to NiaNetVAE main config yaml.",
    )
    parser.add_argument(
        "--export-root",
        default=None,
        help="Override model export root. Defaults to logging_params.model_export_dir.",
    )
    parser.add_argument(
        "--cycles",
        default="0-21",
        help="Cycle set specification, e.g. '0-21' or '0,1,2,5-8'.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output manifest path. Defaults to <export_root>/<dataset>/cycle_manifest.json",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_merged_config(config_path)
    data_params = config.get("data_params", {})
    data_params["regime"] = "per_maint"
    config["data_params"] = data_params

    export_root = args.export_root or config.get("logging_params", {}).get("model_export_dir", "logs/per_maint_models")
    export_root_path = Path(str(export_root)).resolve()
    cycles = _parse_cycles(args.cycles)
    dataset_name = str(config.get("data_params", {}).get("dataset_name", "dataset")).strip() or "dataset"
    output_path = Path(args.output).resolve() if args.output else (export_root_path / dataset_name / "cycle_manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        config=config,
        export_root=export_root_path,
        cycles=cycles,
        paths_relative_to=output_path.parent,
    )
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Manifest written: {output_path}")


if __name__ == "__main__":
    main()
