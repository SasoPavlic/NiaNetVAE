# AGENTS.md

## Purpose and Role
`NiaNetVAE` is the model-production repository in the MetroPT workflow:
- It searches recurrent VAE architectures per maintenance cycle.
- It trains candidates, scores them by multi-objective vectors, and selects the best model.
- It exports cycle artifacts and generates a manifest consumed downstream by `metropt-pdm-framework`.

## Main Entry Points
- `main.py`: config loading/merge, CLI overrides, dataloader selection, DB connector setup, search bootstrap.
- `nianetvae/search/runner.py`: typed runtime orchestrator (`SearchRunner`) for NSGA3 search/fine-tune/warm-start flows.
- `nianetvae/search/`: split runtime helpers (`objective_engine.py`, `winner_selection.py`, `runtime_artifacts.py`, `cycle_warmstart.py`).
- `nianetvae/tools/generate_cycle_manifest.py`: manifest generation from exported cycle artifacts.
- `slurm_scripts/submit_per_maint_pipeline.sh`: HPC submission wrapper (array training + dependent manifest job).

## Pipeline Map (Where Things Live)
- Dataset loading + cycle segmentation: `nianetvae/dataloaders/metropt_dataloader.py`
- Architecture gene mapping / model build: `nianetvae/models/rnn_vae.py`
- Training/test runtime and metrics accumulation: `nianetvae/experiments/rnn_vae_experiment.py`
- Fitness objective and search orchestration:
  - orchestrator: `nianetvae/search/runner.py`
  - objective engine: `nianetvae/search/objective_engine.py`
  - winner selection: `nianetvae/search/winner_selection.py`
  - runtime artifacts/export: `nianetvae/search/runtime_artifacts.py`
  - cycle warm-start/fine-tune helpers: `nianetvae/search/cycle_warmstart.py`
- Persistence layer (SQLite/Postgres): `nianetvae/storage/experiment_storage.py`
- Exported manifest tool: `nianetvae/tools/generate_cycle_manifest.py`

## Stable Contracts (Do Not Change Silently)
- **Cycle semantics** (`MetroPTDataLoader`):
  - `regime="per_maint"` with `cycle_id=0` as `pre_W1`.
  - `cycle_id=1..21` maps to Davari window order.
  - Phase `2` is excluded from train/test filtering in this adaptation.
- **Export layout** (current default):
  - `logs/per_maint_models/<dataset>/cycle_XX/model.pt`
  - `logs/per_maint_models/<dataset>/cycle_XX/model_meta.json`
  - `logs/per_maint_models/<dataset>/cycle_XX/search_summary.json`
  - `logs/per_maint_models/<dataset>/cycle_manifest.json`
- **Schema expectations**:
  - `model_meta.json`, `search_summary.json`, and `cycle_manifest.json` currently emit `schema_version: "1.0"`.
  - Manifest paths are written relative to manifest directory (`paths_relative_to: manifest_directory`).

If you change export formats, manifest fields, cycle naming, feature expectations, or model-loading assumptions, you must evaluate and update downstream `metropt-pdm-framework` compatibility in the same change set.

## Change Rules for Search/Fitness
- Preserve config-driven flow (`configs/main_config.yaml` + dataset config merge).
- Keep candidate dimensionality contract (`RNNVAE` 6-gene architecture vector) unless explicitly versioned and coordinated.
- Keep penalty semantics (`9e10`) for invalid/failed candidates unless there is a deliberate migration.
- When extending fitness logic, keep metric normalization + DB min/max update behavior coherent and backward compatible.
- Prefer minimal, reviewable edits over broad refactors.

## Training/Inference Assumption Safety
- Use `Log` (`log.py`) for run diagnostics; avoid ad-hoc `print` paths in core runtime.
- Keep `--cycle-id` behavior intact for per-cycle HPC runs (`main.py` + Slurm scripts).
- Do not remove final deterministic training/export path after search when `export_enabled=true`.
- Do not hardcode secrets; Postgres credentials are expected through `.env` (`NIANETVAE_DB_*`).

## Local Execution Environment
- For local commands in this workspace, use the dedicated Poetry interpreter:
  - `/mnt/c/Users/sasop/AppData/Local/pypoetry/Cache/virtualenvs/nianetvae-ET2fLSr5-py3.11/Scripts/python.exe`
- Optional shell helper for shorter commands:
  - `PYTHON_BIN=/mnt/c/Users/sasop/AppData/Local/pypoetry/Cache/virtualenvs/nianetvae-ET2fLSr5-py3.11/Scripts/python.exe`

## HPC Per-Maint Workflow Operations
Use `slurm_scripts/submit_per_maint_pipeline.sh` for MetroPT per-maint HPC campaigns. The script reads `configs/main_config.yaml`, so first confirm `workflow.mode`, export directory, DB table, search size, and training epochs in that config.

Recommended production submission from repository root:

```bash
IMAGE_SYNC=0 CHAIN_DEPENDENCY_TYPE=afterany START_CYCLE=0 END_CYCLE=21 RESUME_FROM=auto ./slurm_scripts/submit_per_maint_pipeline.sh --detach
```

If already inside `slurm_scripts/`, use:

```bash
IMAGE_SYNC=0 CHAIN_DEPENDENCY_TYPE=afterany START_CYCLE=0 END_CYCLE=21 RESUME_FROM=auto ./submit_per_maint_pipeline.sh --detach
```

For a known failed cycle, resume from that exact cycle, not the next one:

```bash
IMAGE_SYNC=0 CHAIN_DEPENDENCY_TYPE=afterany START_CYCLE=0 END_CYCLE=21 RESUME_FROM=<failed_cycle_id> ./slurm_scripts/submit_per_maint_pipeline.sh --detach
```

Operational dependency rule:
- Prefer `CHAIN_DEPENDENCY_TYPE=afterany` for long production campaigns. It keeps jobs sequential but allows later cycles to start after a failed cycle ends, avoiding multi-day downtime from `DependencyNeverSatisfied`.
- Use `CHAIN_DEPENDENCY_TYPE=afterok` only for strict fail-fast runs where later cycles must not run unless the previous cycle exported successfully.
- After an `afterany` campaign, rerun missing cycles with `RESUME_FROM=auto`; the script detects completed cycles by `model.pt` + `model_meta.json` or `skipped_non_trainable` status.

Warm-start versus fine-tune implication:
- Warm-start search is naturally more tolerant of `afterany`: if one cycle fails, later cycles can warm-start from the latest available previous trained cycle and mix carry-over, perturbed, and random candidates.
- Fine-tune also works operationally with `afterany`, but it is scientifically more fragile because it expects a previous exported model. If a predecessor failed, later fine-tune results may be less clean or may fall back depending on available artifacts. For final thesis-grade fine-tune comparisons, prefer filling missing cycles and confirming predecessor continuity before downstream evaluation.

Observed HPC failure modes and current mitigations:
- `DependencyNeverSatisfied`: usually caused by strict `afterok` after a failed cycle. Cancel stale dependent jobs and resubmit with `CHAIN_DEPENDENCY_TYPE=afterany`.
- OOM kill: observed with repeated candidate training and DataLoader worker state. Current mitigation is `--mem-per-gpu=64GB`, `num_workers=2`, `persistent_workers=False`, and `pin_memory=False`.
- Time limit: observed because NSGA-III cannot stop cleanly until the active generation finishes; one slow candidate can consume hours. Current mitigation is a small first generation (`n_partitions=2`, effective population 6) plus a larger Slurm search buffer.

Useful HPC diagnostics:

```bash
squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" --me
scontrol show job <job_id> -dd
squeue --start -j <job_id>
tail -f /d/hpc/home/sasop/outputs/nianetvae-nianetvae-metropt<job_id>.out
tail -f /d/hpc/home/sasop/outputs/nianetvae-nianetvae-metropt<job_id>.err
```

When cancelling stale dependent jobs, cancel only pending jobs from the failed chain, for example:

```bash
scancel $(seq <first_stale_job_id> <last_stale_job_id>)
```

## Validation Expectations
Run from repository root:
- Unit/split tests:
  - `$PYTHON_BIN -m pytest tests/test_metropt_dataloader.py`
- Local per-cycle smoke run:
  - `$PYTHON_BIN main.py -c configs/main_config.yaml -met SMAPE --cycle-id 0`
- Manifest generation smoke:
  - `$PYTHON_BIN -m nianetvae.tools.generate_cycle_manifest --config configs/main_config.yaml --cycles 0-21`
- Export verification:
  - confirm `model.pt`, `model_meta.json`, `search_summary.json` exist for trained cycles and manifest resolves statuses.

## Generated Artifacts and Large Files
- Treat `logs/`, `results/`, and exported model artifacts as generated outputs; do not use them as source-of-truth code inputs.
- Do not modify datasets under `data/` as part of routine code changes.
- Avoid committing large generated files/checkpoints unless explicitly required by the task.

## Explicit Assumptions
- Assumption: `metropt-pdm-framework` consumes the current manifest/artifact schema and relative-path behavior.
- Assumption: Production per-maint runs target cycle range `0..21` for MetroPT.
- Assumption: HPC scripts are environment-specific (paths, partition, image location) and may require local cluster adaptation.
