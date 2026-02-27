#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
            candidate = (base_dir / candidate).resolve()
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


def build_manifest(config: dict, export_root: Path, cycles: list[int]) -> dict:
    dataset_name = str(config.get("data_params", {}).get("dataset_name", "dataset")).strip() or "dataset"
    regime = str(config.get("data_params", {}).get("regime", "per_maint")).strip().lower() or "per_maint"

    manifest = {
        "schema_version": "1.0",
        "dataset": dataset_name,
        "regime": regime,
        "generated_at": datetime.now().isoformat(),
        "config_fingerprint": _config_fingerprint(config),
        "active_cycle_order": [_cycle_key(cycle_id) for cycle_id in cycles],
        "cycles": {},
        "notes": [],
    }

    last_trained_cycle: int | None = None
    alias_cycles: list[str] = []
    missing_cycles: list[str] = []

    for cycle_id in cycles:
        key = _cycle_key(cycle_id)
        cycle_dir = export_root / dataset_name / f"cycle_{key}"
        model_path = cycle_dir / "model.pt"
        meta_path = cycle_dir / "model_meta.json"
        summary_path = cycle_dir / "search_summary.json"

        has_artifacts = model_path.exists() and meta_path.exists()
        trainable, trainable_error = _cycle_trainable(config, cycle_id)

        if has_artifacts:
            metadata = _read_meta(meta_path)
            status = "trained"
            entry = {
                "cycle_id": int(cycle_id),
                "status": status,
                "trainable": bool(trainable),
                "artifact_dir": str(cycle_dir),
                "model_path": str(model_path),
                "meta_path": str(meta_path),
                "summary_path": str(summary_path) if summary_path.exists() else None,
                "hash_id": metadata.get("hash_id"),
                "run_uuid": metadata.get("run_uuid"),
                "created_at": metadata.get("created_at"),
            }
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

        manifest["cycles"][key] = entry

    if alias_cycles:
        manifest["notes"].append(
            f"Alias cycles (empty-gap/non-trainable) mapped to previous trained cycle: {', '.join(alias_cycles)}"
        )
    if missing_cycles:
        manifest["notes"].append(
            f"Missing cycle artifacts: {', '.join(missing_cycles)}"
        )
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
    manifest = build_manifest(config, export_root_path, cycles)

    dataset_name = str(config.get("data_params", {}).get("dataset_name", "dataset")).strip() or "dataset"
    output_path = Path(args.output).resolve() if args.output else (export_root_path / dataset_name / "cycle_manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Manifest written: {output_path}")


if __name__ == "__main__":
    main()
