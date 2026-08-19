# NiaNetVAE repository guidance

## Scope

This is the active MetroPT-only controlled-study monorepo. It owns shared data preparation, five workflow implementations, fresh NiaNetVAE search and training, evaluation, cross-workflow comparison, and schema-1.0 study artifacts.

The only supported dataset is MetroPT. Do not restore legacy Yahoo, KPI, MSL, SMAP, SMD, UCR, SWAT, WADI, or NAB loaders or configuration paths.

## Entry points

- `main.py`: checkout-friendly wrapper around `nianetvae.cli`.
- `src/nianetvae/cli.py`: authoritative command-line interface.
- `configs/metropt_study.yaml`: production study configuration.
- `src/nianetvae/dataloaders/`: shared MetroPT preparation, preprocessing, and sequence construction.
- `src/nianetvae/experiments/runner.py`: controlled workflow execution.
- `src/nianetvae/search/engine.py`: cycle-0 NSGA-III search.
- `src/nianetvae/artifacts.py`: schema-1.0 artifact and validation contract.
- `slurm_scripts/`: production HPC orchestration.

## Controlled workflows

The exact workflow set is `iforest_static`, `iforest_per_maintenance`, `sae_static`, `vae_static`, and `nianetvae_per_maintenance`. All workflows must use the shared preparation, preprocessing, calibration, risk, and evaluation core.

## Validation

Run formatting, Ruff, the full pytest suite, configuration validation, and `validate-study` for completed evidence. Production search and training belong on Slurm, not a local IDE or login node.

Artifact schema is `1.0`. A study is immutable by `study_id`; change the ID after any input, controlled constant, contract, or source change. Do not accept historical schema-v2 artifacts or bypass source/config/data fingerprint failures.
