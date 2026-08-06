# NiaNetVAE controlled MetroPT study

This repository is a MetroPT-only research system for comparing global and per-maintenance anomaly detectors under one shared experimental contract. It owns the complete path from frozen data preparation and NiaNetVAE architecture search to event-level predictive-maintenance evaluation.

The legacy multi-dataset runtime, model-specific preprocessing paths, PostgreSQL candidate state, schema-v2 export bridge, and separate downstream evaluation runtime are intentionally unsupported.

## Controlled workflows

| Workflow | Model | Update strategy |
|---|---|---|
| `iforest_static` | Isolation Forest | Initial baseline only |
| `iforest_per_maintenance` | Isolation Forest | Refit after trainable maintenance cycles |
| `sae_static` | Recurrent sparse autoencoder | Initial baseline only |
| `vae_static` | Recurrent variational autoencoder | Initial baseline only |
| `nianetvae_per_maintenance` | Searched recurrent VAE | Sequential fine-tuning |

All workflows use the same MetroPT preparation, frozen preprocessing, calibration timestamps, sequence-anchor evaluation population, risk construction, threshold selection, and event metrics. This is a controlled system comparison rather than a causal architecture-versus-adaptation ablation.

## Local setup

The controlled environment uses Python 3.11 and exact direct-dependency versions:

```powershell
poetry env use C:\Users\sasop\AppData\Local\Programs\Python\Python311\python.exe
poetry install --sync
poetry run python main.py --config configs/metropt_study.yaml validate-config
poetry run pytest -q
```

Safe PyCharm review consists of configuration validation, the full tests, and the synthetic end-to-end integration test. `prepare` reads the real MetroPT dataset and initializes an artifact root. Production `search` and `run-all` workloads belong on Slurm.

## CLI

```powershell
poetry run python main.py --config configs/metropt_study.yaml validate-config
poetry run python main.py --config configs/metropt_study.yaml prepare
poetry run python main.py --config configs/metropt_study.yaml search
poetry run python main.py --config configs/metropt_study.yaml run --workflow iforest_static
poetry run python main.py --config configs/metropt_study.yaml finalize --workflow iforest_static
poetry run python main.py --config configs/metropt_study.yaml compare
poetry run python main.py --config configs/metropt_study.yaml validate-study
```

Use a new `artifacts.study_id` after any input, controlled constant, architecture-search contract, or implementation change. Completed schema-1.0 evidence is accepted only when `validate-study` succeeds.

## Production execution

Submit heavy work through the provided Slurm scripts:

```bash
IMAGE_SYNC=0 CONFIG_PATH=configs/metropt_study.yaml bash slurm_scripts/submit_study.sh
```

`metropt-pdm-framework` remains a historical evaluation oracle and version-3 evidence archive. It does not consume new schema-1.0 artifacts implicitly.
