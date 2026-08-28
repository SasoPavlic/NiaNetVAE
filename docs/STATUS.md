# Project status and decision history

Read this together with `AGENTS.md`. That file explains **how the system works**;
this one explains **where the work stands and why the design is what it is**.
Both live in the repository so a new session starts from git rather than from a
pasted prompt.

Last updated: 2026-08-28, after the `metropt_v7_pilot` review.

---

## 1. What this project is

A MetroPT-only controlled study comparing five anomaly-detection workflows for
predicting compressor failures on a metro train. Four are baselines; the fifth,
`nianetvae_per_maintenance`, is the contributed method: a recurrent VAE whose
architecture is chosen by NSGA-III search and which is fine-tuned after each
maintenance intervention.

Everything is scored on one shared population of **1,146,067 evaluation anchors**
with identical preprocessing, calibration, risk construction, threshold sweep and
event metrics, so differences between workflows come from the workflows.

**Deadline: the paper must be finished by 31 December 2026.** The production
search costs roughly two weeks. There is room for one more full search, not
three. Weigh further methodological iteration against that.

## 2. Immediate next step

**Launch the production v7 search.** Everything required is merged and verified.

```bash
# 1. build and publish the image from the current main commit
docker build -t spartan300/nianet:<sha> .
docker push spartan300/nianet:<sha>

# 2. on HPC, pull BY DIGEST (never by tag)
singularity pull --force /d/hpc/home/sasop/images/nianet-<sha>.sif \
    docker://spartan300/nianet@sha256:<digest>

# 3. verify inside the SIF before submitting
singularity exec -e --pwd /app <sif> python -m nianetvae.cli \
    --config configs/metropt_study_v7.yaml validate-config     # study_id metropt_controlled_v7

# 4. submit phase 1 from /d/hpc/home/sasop/NiaNetVAE
CONFIG_PATH=configs/metropt_study_v7.yaml \
IMAGE_PATH=/d/hpc/home/sasop/images/nianet-<sha>.sif \
bash slurm_scripts/submit_search_ladder.sh
```

The ladder default already points at the v7 steps, so `LADDER` need not be set.
Phase 1 submits prepare, four baselines in parallel, and four chained search
steps at 25, 50, 75 and 100 generations. Expect roughly **two weeks** at the
measured 3.16 h/generation.

**Do not submit phase 2 until the search is reviewed.** The first NiaNetVAE cycle
permanently freezes the search budget. `submit_nianet_workflow.sh` enforces this
and derives the cycle count from the data contract.

## 3. Current state

### Repository

`main` only, no other branches. All three checkouts synced: laptop
(`Razer-LT`), HPC (`/d/hpc/home/sasop/NiaNetVAE`), and the VM
(`agent-worker-01:~/workspaces/NiaNetVAE`).

29 tests, ruff clean, 8 configurations validate.

### Environments

| Machine | Role | Notes |
|---|---|---|
| `agent-worker-01` | Primary agent host | SSH, Docker, Python 3.11, Poetry all verified |
| `Razer-LT` | Original laptop | Same repo, same fingerprints |
| Arnes HPC | Execution | `arnes-hpc` alias, `ControlPersist yes` |

The dataset is byte-identical on all three (`db30ccb4...`), and the source
fingerprints match, so local verification is trustworthy.

The SSH master is opened by the operator: `ssh -fMN arnes-hpc`, passphrase then
OTP. **An agent can never open it** — that is 2FA, not an obstacle to engineer
around. Check with `~/check-hpc.sh` on the VM, or:

```bash
ssh -O check arnes-hpc && ssh -o BatchMode=yes arnes-hpc hostname
```

### Studies on HPC

| Study | State | Notes |
|---|---|---|
| `metropt_controlled_v2` | search only | Original multi-day NSGA-III run |
| `metropt_controlled_v4` | **validated** | Last study under the 21-row schedule |
| `metropt_controlled_v5` | cancelled | Superseded; 25-generation search preserved |
| `metropt_v6_pilot` | **validated** | First merged-failure study, all five workflows |
| `metropt_v7_pilot` | search only | Objective-independence check, phase 2 not run |
| `metropt_controlled_v7` | **not started** | The production study |

## 4. Why the event schedule changed

Failure times come from **Davari et al. (2021), Table II**, carried verbatim in
`DEFAULT_MAINTENANCE_WINDOWS`.

That table reports multi-day failures **one calendar day at a time**. Five
consecutive rows are separated by exactly one minute at midnight, so `#16`, `#17`
and `#18` are one 3,150-minute failure written as three rows.

The 21 reported rows describe **16 physical failures**. Treating them separately
inflated event denominators, placed the pre-failure window of each continuation
*inside* the still-running failure it was meant to predict, and produced five
one-minute cycles with zero anchors.

`merge_contiguous_events` joins rows closer than `data.failure_merge_gap_minutes`
(default 1). Cycles go from 22 to 17, empty cycles from five to none, and the
anchor count is unchanged, so merged studies stay comparable with v4.

**These are failures, not scheduled maintenance.** Some identifiers still say
"maintenance" for continuity; the paper should say failures.

## 5. What the merge changed scientifically

In v4 no workflow reached the target region (`recall >= 0.60`,
`coverage < 0.20`) and NiaNetVAE had the **lowest** recall of five. In the v6
pilot three workflows reach it and NiaNetVAE is on the Pareto frontier:

| Feasible workflow | Recall | Coverage % | FAR/day | F1 | NAB | Mean TTD |
|---|---|---|---|---|---|---|
| `iforest_static` | 0.625 | 12.03 | 4.463 | 0.0314 | 16.31 | 64.7 |
| `iforest_per_maintenance` | 0.688 | 16.49 | 6.853 | 0.0231 | 13.62 | 69.9 |
| `nianetvae_per_maintenance` | 0.625 | 18.24 | **2.314** | **0.0597** | **27.93** | **96.7** |

The model barely changed; the event schedule did.

**Do not overstate this.** It is a Pareto-frontier result *within this controlled
comparison*, at n=16 failures, one seed, no confidence intervals, with the
operating point chosen retrospectively. It is not a state-of-the-art claim, and
10 detections against 11 is not a distinguishable difference.

## 6. Why the search objective changed twice, and what settled it

The search optimises three objectives on **cycle 0 only**, the only data
available without look-ahead into the evaluation period.

**The original second objective was unusable.** Point-wise AUROC against the
pre-failure label, on a cycle holding one failure: 726 positive windows at 0.25%
prevalence. Across 893 v5 candidates **not one exceeded chance** (median 0.247).
Label polarity was verified correct; this is a property of the population.

**The first replacement was still redundant.** Calibration drift correlated with
alarm burden at r=+0.547, slightly worse than the +0.495 it replaced.

**The current set** came from the full correlation matrix, which showed
`alarm_burden` — smoothed risk above a quantile, hence a compound of level,
persistence and reconstruction quality — entangled with every axis:

```
              error    drift   burden    level
  error       1.000    0.004   -0.414   -0.034
  drift       0.004    1.000    0.547    0.059
  burden     -0.414    0.547    1.000    0.648   <- entangled
  level      -0.034    0.059    0.648    1.000
```

Objectives are now, all label-free so none depends on the single cycle-0 failure:

1. **`SMAPE` reconstruction error** — did the model learn normal behaviour at all
2. **`calibration_level_v1`** — how far the false-alarm rate sits from nominal 5%
3. **`calibration_drift_v1`** — how much that rate shifts across the window

Weights `[0.30, 0.35, 0.35]`. Alarm burden is still measured and reported.

### The v7 pilot confirmed this, with one caveat

| | v6 | v7 measured |
|---|---|---|
| Worst pairwise correlation | 0.547 | **0.142** |
| level vs drift | — | 0.006 |
| Pareto front | 5 | 13 |
| Baselines | — | bit-identical to v6 |

**Training is fully deterministic.** 47 architectures appear in both pilots with
bit-identical objectives, so there is no run-to-run noise. A proposed noise
diagnostic was therefore unnecessary.

**Caveat that remains open.** `level` and `drift` separate candidates across only
about 9% of their range, against 76% for SMAPE. The signal is real, not noise,
but weak: SMAPE will dominate ranking with the calibration terms breaking ties.
If the production search converges to something SMAPE-optimal and
calibration-mediocre, this is why. The candidate ledger records all three scores
for every architecture, so it can be checked.

## 7. An idea tested and rejected

Masking evaluation to compressor-active rows made things **worse**: pre-failure
separation fell from 1.50x to 1.15x, per-cycle direction from 11/14 to 9/14.
These are air-leakage failures, whose signature is pressure lost while the system
should be holding, so the idle 84% of the record carries the evidence. Do not
reintroduce without new evidence.

## 8. Known open problems

1. **Multi-seed confidence intervals.** `training.seed` sits inside the immutable
   search contract, so every seed is a distinct study whose search cannot be
   migrated. Affordable intervals need the search seed separated from the
   training seed. Strongest answer to "n=16, one seed".
2. **Cycle 0 is unrepresentative.** Never fine-tuned, accumulates 41 days of
   drift, and its pre-failure risk is *inverted*. Adapted cycles are correct in
   11 of 13. This is evidence for adaptation, but it also means architecture
   selection is guided by the worst available cycle.
3. **Calibration goes stale immediately** — 38% exceedance against a nominal 5%.
4. **Architecture and adaptation are not fully separable.** There is no static
   NiaNetVAE control. The IForest pair isolates adaptation for a classical model
   only. Frame the study as a controlled system comparison, not a causal ablation.

## 9. Documentation

The `PhD` collection at `wiki.kvaltko.com` was rewritten on 2026-08-27 and is
current. Nine pages covering research question, dataset and failure labels, drift
evidence, system map, architecture search, evaluation framework, reproducibility
and provenance, experimental evidence, and open problems.

Result tables are **generated, never typed**:

```bash
python docs/build_results.py artifacts/<study_id> -o results.md
```

Access Outline through the `kvaltko-wiki` MCP server. Reads are ordinary work;
**publishing requires the operator to say so**.

## 10. Working agreements

- Production search and training run on Slurm, never on a login node or laptop.
- A study is immutable by `study_id`; any source change requires a new one.
- Never rebuild or overwrite a SIF that queued jobs point at.
- Pull images by digest, never by tag.
- Prefer fixes outside `SEARCH_RUNTIME_PATTERNS` when they preserve a migratable
  search; see `AGENTS.md`.
- Claims in the paper must be traceable to a `validate-study` accepted artifact.
- State limits alongside claims. The frontier result in section 5 is defensible
  only because its boundaries are given with it.
