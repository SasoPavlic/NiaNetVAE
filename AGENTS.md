# AGENTS.md

## 1. PROJECT OVERVIEW
NiaNetVAE is a config-driven ML system for evolutionary search of recurrent variational autoencoder architectures for time-series reconstruction and anomaly-oriented evaluation.

Engineering behavior:
- Entry point: `main.py`.
- Search engine: `pymoo` with `NSGA2` in `nianetvae/rnn_vae_architecture_search.py`.
- Model factory/search space: `nianetvae/models/rnn_vae.py` (7-gene solution vector maps to layer type, depth/step, activation, optimizer).
- Training runtime: PyTorch Lightning (`Trainer`) through `nianetvae/experiments/rnn_vae_experiment.py`.
- Data ingestion: dataset-specific LightningDataModules in `nianetvae/dataloaders/`.
- Storage:
  - SQLite (`logging_params.db_storage`) or
  - Postgres (`logging_params.db_backend: postgres`, `logging_params.db_params`).
- MetroPT integration supports:
  - `regime: single`
  - `regime: per_maint` with `--cycle-id`
  - DB dataset isolation suffix `MetroPT_cycleXX` for per-cycle jobs.

ML objective:
- Multi-objective optimization of reconstruction quality + architecture complexity.
- Fitness normalization uses observed metric min/max persisted in DB.
- Invalid architectures are penalized with worst-case values.

## 2. DEVELOPMENT ENVIRONMENT
- Python: `>=3.10,<3.12` (from `pyproject.toml`).
- Dependency management:
  - Primary: Poetry (`pyproject.toml`, `poetry.lock`).
  - Docker install path uses `requirements.txt`.
- CUDA/GPU:
  - GPU expected for production runs.
  - Local Poetry workflow commonly requires:
    - `poetry run poe autoinstall-torch-cuda`
- Docker:
  - Main image uses `Dockerfile` with Lightning CUDA base.
  - Data is mounted at runtime; datasets are not copied into image.

## 3. RUN AND BUILD COMMANDS
From repo root (`NiaNetVAE/`):

### Install (Poetry) ONLY USE THIS WHEN EXPLICITLY ASKED, BECAUSE IT TAKES A VERY LONG TIME
```bash
poetry install
poetry run poe autoinstall-torch-cuda
```

### Local training/search
```bash
python main.py -c configs/main_config.yaml -alg particle_swarm -met SMAPE
```

### MetroPT per-cycle run
```bash
python main.py -c configs/main_config.yaml -alg particle_swarm -met SMAPE --cycle-id 5
```

### Tests
```bash
pytest tests/test_metropt_dataloader.py
```
or
```bash
poetry run pytest tests/test_metropt_dataloader.py
```

### Docker build
```bash
docker build -t spartan300/nianet:vaepymoo -f Dockerfile .
```

### Docker run (GPU, mounted volumes)
```bash
docker run --rm -it --gpus all \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs \
  -w /app \
  spartan300/nianet:vaepymoo \
  python main.py -c configs/main_config.yaml -alg particle_swarm -met SMAPE
```

### HPC submit
```bash
sbatch nianetvae.sh
```

## 4. PROJECT STRUCTURE FOR AGENTS
- `main.py`
  - Config loading/merging, CLI overrides, dataloader selection, search bootstrap.
- `configs/`
  - `main_config.yaml`: global run/search/logging/trainer/db settings.
  - `*_config.yaml`: dataset-level `data_params` (e.g., `metropt_config.yaml`).
- `nianetvae/`
  - Core package.
- `nianetvae/dataloaders/`
  - Dataset-specific LightningDataModules and sequence datasets.
- `nianetvae/models/`
  - VAE base + `RNNVAE` search-space implementation.
- `nianetvae/experiments/`
  - Lightning experiment module, reconstruction metrics, anomaly metrics.
- `nianetvae/storage/`
  - SQLite/Postgres connectors and experiment persistence.
- `training/`
  - No dedicated `training/` directory exists. Training flow is implemented in `main.py` + `nianetvae/experiments/` + `nianetvae/rnn_vae_architecture_search.py`.

### 4.A Excluded folders (do not use as implementation targets)
- `helpers/`
- `logs/`
- `data/`
- `results/`

## 5. CODING CONVENTIONS
- Follow existing architecture:
  - Config-first workflow (`YAML` + runtime merge).
  - Dataset selection by `data_params.dataset_name`.
- Logging:
  - Use `Log` class from `log.py` (`Log.info/debug/warning/error`), not `print`.
- Model creation pattern:
  - Construct model from solution vector (`RNNVAE(solution, **config)`).
  - Respect `model.is_valid` and penalty behavior.
- Experiment pattern:
  - Use `RNNVAExperiment` with Lightning callbacks (`FineTuneLearningRateFinder`, `EarlyStopping`).
- Storage pattern:
  - Use `get_db_connector(config, table_name)` factory.
  - Keep SQLite and Postgres support backward compatible.
- Typing:
  - Mixed style in codebase (partial typing in newer modules). Match local file style.
- Avoid introducing new abstractions unless repeated pain is proven.

## 6. TESTING AND VALIDATION
Minimum before merge:
1. Run targeted test:
   - `pytest tests/test_metropt_dataloader.py`
2. Run one smoke training invocation with your edited config.
3. Verify no regressions in:
   - config loading
   - dataloader selection
   - DB connector initialization
   - end-of-run best model save path.

Dataset considerations:
- Do not require large real datasets for unit tests.
- Follow current test approach: synthetic temp CSV for MetroPT loader.
- Keep sequence shapes and split semantics stable.

## 7. HPC + DOCKER WORKFLOW
- Build/push container externally, then run on cluster via Singularity/Apptainer.
- `nianetvae.sh` is the operational reference:
  - GPU partition.
  - `--array=1-21` for cycle-per-job MetroPT.
  - Bind mounts:
    - `$(pwd)/logs:/app/logs`
    - `$(pwd)/data:/app/data`
    - `$(pwd)/configs:/app/configs`
- Do not break bind-mounted path assumptions (`/app/...`).
- Do not remove `--cycle-id ${SLURM_ARRAY_TASK_ID}` from MetroPT array runs.
- Keep image entry command consistent with `python main.py ...`.

## 8. SAFETY AND CONSTRAINTS
- Do not modify datasets under `data/`.
- Do not alter or delete prior experiment outputs/log artifacts.
- Preserve backward compatibility for non-MetroPT datasets.
- Keep both DB backends functional (`sqlite` + `postgres`).
- Do not hardcode environment-specific credentials in new code.
- Avoid schema-breaking DB changes unless explicitly requested with migration plan.
- Do not silently change search objective semantics or metric names.

## 9. AGENT BEHAVIOR RULES
- Prefer minimal edits over large refactors.
- Follow existing architecture instead of inventing new abstractions.
- Always align new modules with current folder patterns.
- Keep changes localized; avoid cross-cutting rewrites.
- Validate with targeted tests first, then broader smoke checks.
- If runtime assumptions are unclear (HPC paths, DB availability, CUDA), surface them explicitly before implementation.