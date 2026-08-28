# Project status and decision history

Read this together with `AGENTS.md`. That file explains **how the system works**;
this one explains **where the work currently stands and why the design is what it
is**. Both are kept in the repository so a new session can start from git rather
than from a pasted prompt.

Last updated: 2026-08-27, before the `metropt_v7_pilot` search completed.

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

## 2. Current state

### Repository

| | |
|---|---|
| `main` | `efc8ee0` — predates the v6/v7 work |
| `feat/v6-failure-semantics` | `97eee9f` — pushed, **not merged**, 3 commits ahead |
| Uncommitted | `docs/build_results.py` (objective-label fix, held pending v7 review) |

The branch must be merged to `main` once the v7 pilot is reviewed.

### Studies on HPC (`/d/hpc/home/sasop/NiaNetVAE/artifacts/`)

| Study | State | Notes |
|---|---|---|
| `metropt_controlled_v2` | search only | Original multi-day NSGA-III run; the rest was deleted |
| `metropt_controlled_v4` | **validated** | Last complete study under the old event schedule |
| `metropt_controlled_v5` | cancelled | Superseded; 25-generation search preserved |
| `metropt_v6_pilot` | **validated** | First study with merged failures; see §4 |
| `metropt_v7_pilot` | search running | Objective-independence check; see §5 |

### Images

Named per commit, pulled by digest. `nianet-97eee9f.sif` is current.
The study a job belongs to is pinned to its image; never overwrite an existing SIF
while jobs referencing it are queued.

## 3. Why the event schedule changed (the most important correction)

Failure times come from **Davari et al. (2021), Table II**, which the repository
carries verbatim in `DEFAULT_MAINTENANCE_WINDOWS`.

That table reports multi-day failures **one calendar day at a time**. Five
consecutive rows are separated by exactly one minute at midnight, so `#16`, `#17`
and `#18` are one 3,150-minute failure written as three rows.

The 21 reported rows describe **16 physical failures**. Treating them separately:

- inflated event-level denominators;
- placed the pre-failure window of each continuation *inside* the still-running
  failure it was supposed to predict;
- produced five one-minute "cycles" with zero evaluation anchors.

`merge_contiguous_events` in `dataloaders/metropt.py` joins rows closer than
`data.failure_merge_gap_minutes` (default 1). Cycles go from 22 to 17, empty
cycles from five to none, and the anchor count is unchanged — so merged studies
stay directly comparable with v4.

**These are failures, not scheduled maintenance.** Some identifiers still say
"maintenance" for continuity; the events themselves are unplanned failures, and
the paper should say so.

## 4. What the merge changed scientifically

In v4, no workflow reached the target region (`recall >= 0.60` and
`coverage < 0.20`) and NiaNetVAE had the **lowest** recall of the five. In the v6
pilot, three workflows reach it and NiaNetVAE is on the Pareto frontier:

| Feasible workflow | Recall | Coverage % | FAR/day | F1 | NAB | Mean TTD |
|---|---:|---:|---:|---:|---:|---:|
| `iforest_static` | 0.625 | 12.03 | 4.463 | 0.0314 | 16.31 | 64.7 |
| `iforest_per_maintenance` | 0.688 | 16.49 | 6.853 | 0.0231 | 13.62 | 69.9 |
| `nianetvae_per_maintenance` | 0.625 | 18.24 | **2.314** | **0.0597** | **27.93** | **96.7** |

`sae_static` and `vae_static` remain `fallback`.

The model barely changed; the event schedule did. The earlier conclusion that no
method reaches the target was substantially an artifact of the mis-split list.

**Do not overstate this.** It is a Pareto-frontier result *within this controlled
comparison*, at n=16 failures, one seed, no confidence intervals, with the
operating point chosen retrospectively on the same timeline. It is not a
state-of-the-art claim, and 10 detections versus 11 is not a distinguishable
difference.

## 5. Why the search objective changed twice

The search optimises three objectives on **cycle 0 only**, because that is the
only data available without look-ahead into cycles the model is later scored on.

**The original second objective was unusable.** It was point-wise AUROC of
smoothed risk against the pre-failure label. Cycle 0 contains one failure: 726
positive windows at 0.25% prevalence. Across 893 v5 candidates **not one exceeded
chance** (median AUROC 0.247), so the heaviest-weighted term was noise. The label
polarity was verified correct; this is a property of the population, not a bug.

**The first replacement was still redundant.** Calibration drift was measured
against alarm burden over 270 v6 candidates at **r = +0.547** — slightly worse
than the term it replaced (+0.495). Searching three objectives where two
correlate at 0.55 explores roughly two dimensions.

**The current set** comes from the full correlation matrix:

```
              error    drift   burden    level
  error       1.000    0.004   -0.414   -0.034
  drift       0.004    1.000    0.547    0.059
  burden     -0.414    0.547    1.000    0.648   <- entangled with everything
  level      -0.034    0.059    0.648    1.000

  (error, drift, level )  worst pair |r| = 0.059   <- chosen
  (error, drift, burden)  worst pair |r| = 0.547
```

`alarm_burden` is smoothed risk above a quantile, hence a compound of level,
persistence and reconstruction quality — which is why it correlates with
everything. The v7 objectives are therefore:

1. **`SMAPE` reconstruction error** — did the model learn normal behaviour at all.
2. **`calibration_level_v1`** — how far the false-alarm rate sits from the nominal
   5%. Normal operation currently exceeds the calibration quantile **38%** of the
   time, so this is the dominant measured pathology.
3. **`calibration_drift_v1`** — how much that rate shifts between the first and
   second half of the window, i.e. whether the model's sense of normal survives
   the machine drifting.

All three are label-free, so none depends on the single failure in cycle 0.
Alarm burden is still measured and reported, just not optimised.

**The v7 pilot exists to test this**: do the three stay independent on fresh
candidates, and do candidates actually spread out on them? Objective spreads have
been narrow (IQR/range around 0.1), so if the new terms do not discriminate, the
set needs another look **before** committing to a multi-week search.

## 6. An idea that was tested and rejected

Masking evaluation to compressor-active rows was proposed and measured. It made
things **worse**: pre-failure separation fell from 1.50x to 1.15x, and per-cycle
direction from 11/14 correct to 9/14. These are air-leakage failures, whose
signature is pressure lost while the system should be holding, so the idle 84% of
the time carries the evidence. Do not reintroduce this without new evidence.

## 7. Known open problems

1. **Multi-seed confidence intervals.** `training.seed` sits inside the immutable
   search contract, so every seed is a distinct study whose search cannot be
   migrated. Affordable confidence intervals need the search seed separated from
   the training seed. This is the strongest answer to "n=16, one seed".
2. **Cycle 0 is unrepresentative.** It never gets fine-tuned, accumulates 41 days
   of drift, and its pre-failure risk is *inverted*. Adapted cycles are correct in
   11 of 13. This is evidence for the value of adaptation, but it also means
   architecture selection is guided by the worst available cycle.
3. **Calibration goes stale immediately** — 38% exceedance against a nominal 5%.
4. **The wiki is stale.** The `PhD` collection on `wiki.kvaltko.com` still presents
   retired `metropt-pdm-framework` numbers as current evidence. Those numbers
   contradict v4 and later. **Do not write the paper from the wiki until it is
   rewritten.**

## 8. Immediate next steps

1. Review the `metropt_v7_pilot` search when it completes.
2. Commit the held `docs/build_results.py` fix plus anything the review surfaces.
3. Merge `feat/v6-failure-semantics` into `main`; delete the branch.
4. Rewrite the wiki from `docs/wiki/`, generating the results page with
   `docs/build_results.py` rather than by hand.
5. Decide the production run: `metropt_controlled_v7`, ladder to 100 generations,
   roughly two weeks at the measured 3.16 h/generation.

## 9. Working agreements

- Production search and training run on Slurm, never on a login node or laptop.
- A study is immutable by `study_id`; any source change requires a new one.
- Never rebuild or overwrite a SIF that queued jobs point at.
- Prefer fixes outside `SEARCH_RUNTIME_PATTERNS` when they preserve a migratable
  search; see `AGENTS.md`.
- Outline writes are not implicitly authorised. Reads are ordinary work; publishing
  to the `PhD` collection needs the operator to say so.
- Claims in the paper must be traceable to a `validate-study` accepted artifact.
