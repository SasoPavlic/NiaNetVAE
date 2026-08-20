# NiaNetVAE repository guidance

## Scope

This is the active MetroPT-only controlled-study monorepo. It owns shared data preparation, five workflow implementations, fresh NiaNetVAE search and training, evaluation, cross-workflow comparison, and schema-1.0 study artifacts.

The only supported dataset is MetroPT. Do not restore legacy Yahoo, KPI, MSL, SMAP, SMD, UCR, SWAT, WADI, or NAB loaders or configuration paths.

## Entry points

- `main.py`: checkout-friendly wrapper around `nianetvae.cli`.
- `src/nianetvae/cli.py`: authoritative command-line interface.
- `configs/metropt_study_v4.yaml`: current production study configuration.
- `configs/metropt_study.yaml`, `configs/metropt_study_v3.yaml`: superseded
  study identities, retained for provenance.
- `src/nianetvae/dataloaders/`: shared MetroPT preparation, preprocessing, and sequence construction.
- `src/nianetvae/experiments/runner.py`: controlled workflow execution.
- `src/nianetvae/search/engine.py`: cycle-0 NSGA-III search.
- `src/nianetvae/artifacts.py`: schema-1.0 artifact and validation contract.
- `slurm_scripts/`: production HPC orchestration.

## Current evidence

`metropt_controlled_v4` is the accepted study: schema 1.0, all five workflows
completed, `validate-study` reports `valid: true`. It was produced by commit
`5bf4272` from image `spartan300/nianet:5bf4272`
(`sha256:47aaf1756ce68092f386c5181a5b86a061779fde59a2b47a9dcf3f033c61c43e`), with
the architecture search migrated from `metropt_controlled_v2` under
search-runtime fingerprint
`0ce1ade8e02cb31e18bffc7aeac9a0dd64b7851c751c03de75b770817ed69f5c`.

Earlier study identities are superseded: `v1` was abandoned before any workflow
completed, `v2` holds the original multi-day NSGA-III search, and `v3` aborted
at NiaNetVAE cycle 10. Do not present them as evidence.

## Controlled workflows

The exact workflow set is `iforest_static`, `iforest_per_maintenance`, `sae_static`, `vae_static`, and `nianetvae_per_maintenance`. All workflows must use the shared preparation, preprocessing, calibration, risk, and evaluation core.

## HPC execution

Production `search`, `run`, and `run-all` workloads execute on the Arnes SLURM
cluster. They never run on a login node or a local IDE.

### Connection

The agent cannot open the connection: the key is passphrase-protected and login
requires an interactive OTP. The operator opens one SSH master, and the agent
reuses it read-only.

```powershell
wsl ssh -fMN arnes-hpc
```

```bash
wsl ssh -O check arnes-hpc
wsl ssh -o BatchMode=yes arnes-hpc hostname
```

`arnes-hpc` is a WSL SSH alias for `hpc-login4.arnes.si` with `ControlMaster
auto` and `ControlPersist 8h`. This repository contains no key, passphrase, OTP
seed, or OTP code. Quote remote commands carefully; prefer writing a script
locally and piping it over `ssh` instead of nesting quotes.

- Repository checkout: `/d/hpc/home/sasop/NiaNetVAE`
- SLURM logs: `/d/hpc/home/sasop/outputs`
- Images: `/d/hpc/home/sasop/images`

### Code runs from the image, not the checkout

Jobs execute `nianetvae` from inside the Singularity image
(`/opt/conda/lib/python3.11/site-packages/nianetvae`). Only `data/`, `configs/`,
and `artifacts/` are bind-mounted from the checkout. **Editing Python in the HPC
checkout changes nothing at run time.**

Images are named per commit, built from `Dockerfile` and pulled by digest:

```bash
docker build -t spartan300/nianet:<short-sha> .
docker push spartan300/nianet:<short-sha>
singularity pull --force /d/hpc/home/sasop/images/nianet-<short-sha>.sif \
    docker://spartan300/nianet@sha256:<digest>
```

Pull by digest, not by tag, so the SIF provably corresponds to the commit.
`IMAGE_PATH` must always be passed explicitly; the sbatch default does not exist.

### Search-runtime fingerprint gate

`migrate-search` imports a completed architecture search only when the donor and
target search runtimes are byte-identical. Before submitting a migrated study,
verify the fingerprint **inside the target image**:

```bash
singularity exec -e --pwd /app <image.sif> \
    python -m nianetvae.cli fingerprint-search-runtime
```

It must equal the donor's recorded `donor_search_runtime_fingerprint`. If it
differs, migration fails and a fresh multi-day search is required.

Files matched by `SEARCH_RUNTIME_PATTERNS` in `src/nianetvae/search/migration.py`
(`config.py`, `contracts.py`, `dataloaders/**`, `evaluation/calibration.py`,
`evaluation/risk.py`, `models/**`, most of `search/**`, `training/**`) change that
fingerprint. When a defect can be corrected outside that set — for example in
`artifacts.py` or `experiments/runner.py` — prefer the outside fix and keep the
migrated search valid.

Note that `source_contract_fingerprint` hashes **every** `.py` file in the
package and is re-checked by `assert_initialized` on every command. Any source
change therefore makes an existing study unresumable and requires a new
`study_id`, independent of the search-runtime question.

### Submission

```bash
CONFIG_PATH=configs/metropt_study_v4.yaml \
IMAGE_PATH=/d/hpc/home/sasop/images/nianet-<short-sha>.sif \
DONOR_STUDY_ROOT=artifacts/metropt_controlled_v2 \
DONOR_SEARCH_RUNTIME_FINGERPRINT=<verified fingerprint> \
bash slurm_scripts/submit_migrated_study.sh
```

`submit_study.sh` runs a fresh search instead of migrating one.
`submit_migrated_study.sh` submits 31 dependent jobs: prepare, search migration,
four baseline workflows, 22 sequential NiaNetVAE cycles, finalize, compare, and
`validate-study`. Monitor with `squeue --me`; a failed cycle cancels every
dependent job via `afterok`.

## Research knowledge base

Use the configured `kvaltko-wiki` MCP server for all Outline reads and updates
to the `PhD` collection. The previous direct Outline API token has been revoked;
do not use direct bearer-token API calls or restore that token.


## Validation

Run formatting, Ruff, the full pytest suite, configuration validation, and `validate-study` for completed evidence. Production search and training belong on Slurm, not a local IDE or login node.

Artifact schema is `1.0`. A study is immutable by `study_id`; change the ID after any input, controlled constant, contract, or source change. Do not accept historical schema-v2 artifacts or bypass source/config/data fingerprint failures.
