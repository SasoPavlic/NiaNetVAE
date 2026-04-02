# AGENTS.md

## Purpose and Role
`NiaNetVAE` is the model-production repository in the MetroPT workflow:
- It searches recurrent VAE architectures per maintenance cycle.
- It trains candidates, scores them by fitness (reconstruction error + complexity), and selects the best model.
- It exports cycle artifacts and generates a manifest consumed downstream by `metropt-pdm-framework`.

## Main Entry Points
- `main.py`: config loading/merge, CLI overrides, dataloader selection, DB connector setup, search bootstrap.
- `nianetvae/rnn_vae_architecture_search.py`: NSGA3 search loop, objective computation, final retraining, export.
- `nianetvae/tools/generate_cycle_manifest.py`: manifest generation from exported cycle artifacts.
- `slurm_scripts/submit_per_maint_pipeline.sh`: HPC submission wrapper (array training + dependent manifest job).

## Pipeline Map (Where Things Live)
- Dataset loading + cycle segmentation: `nianetvae/dataloaders/metropt_dataloader.py`
- Architecture gene mapping / model build: `nianetvae/models/rnn_vae.py`
- Training/test runtime and metrics accumulation: `nianetvae/experiments/rnn_vae_experiment.py`
- Fitness objective and search orchestration: `nianetvae/rnn_vae_architecture_search.py`
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
- Keep candidate dimensionality contract (`RNNVAE` 7-gene vector) unless explicitly versioned and coordinated.
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
