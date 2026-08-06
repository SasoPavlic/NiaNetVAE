"""Command-line entry point for the controlled MetroPT study."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .artifacts import StudyArtifactStore, read_json
from .config import DEFAULT_WORKFLOWS, StudyConfig, load_study_config
from .dataloaders.metropt import PreparedMetroPTData, metropt_file_hash, prepare_metropt
from .experiments import WorkflowRunner, build_comparison
from .search import SearchEngine

DEFAULT_CONFIG = "configs/metropt_study.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nianetvae",
        description="Controlled MetroPT architecture-search and workflow comparison.",
    )
    parser.add_argument("--config", "-c", default=DEFAULT_CONFIG, help="Study YAML path.")
    parser.add_argument("--verbose", action="store_true", help="Enable diagnostic logging.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("validate-config", help="Validate configuration without loading data.")
    subcommands.add_parser("prepare", help="Build and freeze the shared data contract.")
    subcommands.add_parser("search", help="Run or resume fresh cycle-0 NSGA-III search.")

    run = subcommands.add_parser("run", help="Run/resume a controlled workflow.")
    run.add_argument("--workflow", required=True, choices=DEFAULT_WORKFLOWS)
    run.add_argument("--cycle-id", type=int, default=None, help="Run exactly one cycle.")

    finalize = subcommands.add_parser("finalize", help="Combine completed cycle artifacts.")
    finalize.add_argument("--workflow", required=True, choices=DEFAULT_WORKFLOWS)

    subcommands.add_parser("run-all", help="Search, run all five workflows, compare, and validate.")
    subcommands.add_parser("compare", help="Build the cross-workflow comparison table.")
    subcommands.add_parser("validate-study", help="Fail unless all artifacts share one contract.")
    return parser


def _context(
    config_path: str | Path,
) -> tuple[StudyConfig, PreparedMetroPTData, StudyArtifactStore]:
    config_source = Path(config_path).expanduser().resolve()
    config = load_study_config(config_source)
    store = StudyArtifactStore.from_config(config)
    repository = Path(__file__).resolve().parents[2]
    if store.manifest_path.is_file() and store.prepared_cache_path.is_file():
        manifest = read_json(store.manifest_path)
        if manifest.get("study_config_fingerprint") != config.fingerprint():
            raise ValueError(
                "Existing study_id uses a different configuration. Choose a new study_id."
            )
        contract = read_json(store.shared_dir / "data_contract.json")
        current_dataset_hash = metropt_file_hash(config.data.input_path)
        if current_dataset_hash != contract.get("dataset_hash"):
            raise ValueError(
                "MetroPT input changed after study preparation. "
                "Choose a new study_id and prepare again."
            )
        prepared = store.load_prepared_cache()
        store.assert_initialized(config, prepared)
    else:
        prepared = prepare_metropt(config.data, config.preprocessing.policy)
        if store.manifest_path.is_file():
            store.assert_initialized(config, prepared)
        else:
            store.initialize(
                config,
                prepared,
                repository=repository,
                config_source=config_source,
            )
        store.save_prepared_cache(prepared)
    store.record_execution_config(config)
    return config, prepared, store


def _emit(payload: Any) -> None:
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict(orient="records")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "validate-config":
        config = load_study_config(args.config)
        _emit(
            {
                "valid": True,
                "study_id": config.artifacts.study_id,
                "study_config_fingerprint": config.fingerprint(),
                "resolved_config_fingerprint": config.resolved_fingerprint(),
                "workflows": list(config.workflows),
            }
        )
        return 0

    config, prepared, store = _context(args.config)
    if args.command == "prepare":
        _emit(
            {
                "prepared": True,
                "study_root": str(store.root),
                "data_contract_fingerprint": prepared.data_contract_fingerprint,
                "preprocessing_fingerprint": prepared.preprocessor.fingerprint,
                "evaluation_anchors": int(prepared.evaluation_mask.sum()),
            }
        )
    elif args.command == "search":
        _emit(SearchEngine(config, prepared, store).run())
    elif args.command == "run":
        runner = WorkflowRunner(config, prepared, store)
        if args.cycle_id is None:
            _emit(runner.run_workflow(args.workflow))
        else:
            _emit(runner.run_cycle(args.workflow, args.cycle_id))
    elif args.command == "finalize":
        _emit(WorkflowRunner(config, prepared, store).finalize_workflow(args.workflow))
    elif args.command == "run-all":
        SearchEngine(config, prepared, store).run()
        runner = WorkflowRunner(config, prepared, store)
        summaries = [runner.run_workflow(workflow_id) for workflow_id in config.workflows]
        comparison = build_comparison(config, store)
        validation = store.validate_study(config.workflows)
        _emit(
            {
                "summaries": summaries,
                "comparison": comparison.to_dict(orient="records"),
                "validation": validation,
            }
        )
    elif args.command == "compare":
        _emit(build_comparison(config, store))
    elif args.command == "validate-study":
        _emit(store.validate_study(config.workflows))
    else:  # pragma: no cover - argparse enforces the command choices.
        parser.error(f"Unsupported command: {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
