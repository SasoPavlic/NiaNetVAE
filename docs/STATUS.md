# Project status and decision history

Read this together with `AGENTS.md`. That file explains **how the system works**;
this one explains **where the work stands and why the design is what it is**.
Both live in the repository so a new session starts from git rather than from a
pasted prompt.

Last updated: 2026-09-14. The experiment is closed and the codebase is frozen.

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
search cost two weeks and is now spent. There is no budget for a second one.

## 2. The experiment is closed

**`metropt_controlled_v7` is the accepted evidence**, `validate-study`
`valid: true`, five workflows, zero errors. The baseline-convergence diagnostic
that was the last open worry has been run and settled it (section 8). Nothing is
queued on HPC.

**The codebase is frozen at commit `a57acab`**, the commit that produced v7.
The experimental phase ended on 2026-09-14 by the operator's decision.

The remaining work is writing the paper. Return to this repository only when the
manuscript needs something specific from it — a number, or a figure generated
from artifacts that already exist.

**Do not propose further experiments.** Several were considered and explicitly
declined: a second convergence diagnostic with a raised `max_epochs`, and a
static NiaNetVAE control that would have separated the architecture-search
effect from the adaptation effect. Both were reasonable; both were refused,
because each meant a new `study_id`, more compute and more delay against a
31 December 2026 deadline. A residual methodological gap is now something to
**state in the limitations section**, not something to go and measure.

## 3. Current state

### Repository

`main` only, no other branches. Checkouts synced: laptop (`Razer-LT`), HPC
(`/d/hpc/home/sasop/NiaNetVAE`), and the VM (`agent-worker-01:~/workspaces/NiaNetVAE`).

29 tests, ruff clean, 8 configurations validate.

### Environments

| Machine | Role | Notes |
|---|---|---|
| `agent-worker-01` | Primary agent host | SSH, Docker, Python 3.11, Poetry all verified |
| `Razer-LT` | Original laptop | Same repo, same fingerprints |
| Arnes HPC | Execution | `arnes-hpc` alias, `ControlPersist 8h` |

The dataset is byte-identical on all three (`db30ccb4...`).

The SSH master is opened by the operator: `ssh -fMN arnes-hpc`, passphrase then
OTP. **An agent can never open it** — that is 2FA, not an obstacle to engineer
around. It expires after 8 hours and must be reopened. Check with
`~/check-hpc.sh` on the VM.

### Studies on HPC

| Study | State | Notes |
|---|---|---|
| `metropt_controlled_v2` | search only | Original multi-day NSGA-III run |
| `metropt_controlled_v4` | validated | Last study under the 21-row schedule |
| `metropt_controlled_v5` | cancelled | Superseded; 25-generation search preserved |
| `metropt_v6_pilot` | validated | First merged-failure study, 6-generation search |
| `metropt_v7_pilot` | search only | Objective-independence check |
| `metropt_controlled_v7` | **accepted** | **The production study, and the evidence** |
| `metropt_v7_patience_diag` | diagnostic | Baseline convergence check; not evidence |

### How v7 was produced

| Link | Value |
|---|---|
| Commit | `a57acab`, clean working tree |
| Image | `spartan300/nianet:a57acab` |
| Registry digest | `sha256:40e2c9375c8157d83f4312d6df2c7cc481605d6b18ef67dbdb5fb8ecc8d47aa8` |
| SIF | `/d/hpc/home/sasop/images/nianet-a57acab.sif`, pulled by digest |
| `source_contract_fingerprint` | `5c00be697edc6738f92de35b54606d6fcba1fdb5c1ba125bd94cf677c5d5db90` |
| `study_config_fingerprint` | `0489691458e03e6e13081ab7a86d360d728a3a031a0a59d40e2f693625e0ccdc` |
| `data_contract_fingerprint` | `101ace39be5c12c5dce908bd379e53945fbc9bde14d86eaee4b28f9094a05d47` |

The source fingerprint was verified identical in three places: the local
checkout at `a57acab`, the SIF on HPC, and the recorded study manifest. v7 is
byte-reproducible.

**Provenance gap worth knowing.** `study_manifest.json` records neither the
image digest nor the git commit — its `repository` block is `{branch: null,
commit: null, dirty: null}`, because the code runs inside the image where no git
repository exists, and the sbatch worker does not echo `IMAGE_PATH`. The chain
above is therefore reconstructed from the submission, not read out of the
artifact. The `source_contract_fingerprint` still ties the study to exact source
bytes, which is the scientifically load-bearing link, but recording the image
digest in the manifest would be a cheap improvement for the next study.

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

v7's prepared contract confirms this: 16 `maintenance_events`, 17 cycles, zero
empty cycles, per-cycle anchors summing to exactly 1,146,067, with the merges
visible as events `#2+#3`, `#9+#10`, `#12+#13` and `#16+#17+#18`.

**These are failures, not scheduled maintenance.** Some identifiers still say
"maintenance" for continuity; the paper should say failures.

## 5. The accepted result

`metropt_controlled_v7`, five workflows, 1,146,067 anchors, 16 failures.

| Workflow | Selection | Recall | Coverage % | FAR/day | F1 | NAB | Mean TTD | TP/FN |
|---|---|---|---|---|---|---|---|---|
| `iforest_static` | feasible | 0.6250 | 12.03 | 4.463 | 0.0314 | 16.31 | 64.7 | 10/6 |
| `iforest_per_maintenance` | feasible | 0.6875 | 16.49 | 6.853 | 0.0231 | 13.62 | 69.9 | 11/5 |
| `sae_static` | fallback | 0.4375 | 15.44 | 1.681 | 0.0569 | 21.78 | 83.2 | 7/9 |
| `vae_static` | fallback | 0.5625 | 22.97 | 2.729 | 0.0465 | 22.71 | 91.3 | 9/7 |
| `nianetvae_per_maintenance` | feasible | 0.6250 | 18.75 | **2.269** | **0.0610** | **29.07** | **102.9** | 10/6 |

Three workflows reach the target region (`recall >= 0.60`, `coverage < 0.20`).
Among those three, NiaNetVAE leads on false-alarm rate, F1, NAB and warning time.

**Read the `fallback` rows carefully.** `sae_static` shows the lowest FAR/day of
all five at 1.681, but it never reaches target recall, so it is not a comparison
point. Only the three feasible rows can be compared.

**Do not overstate this.** It is a Pareto-frontier result *within this controlled
comparison*, at n=16 failures, one seed, no confidence intervals, with the
operating point chosen retrospectively. It is not a state-of-the-art claim, and
10 detections against 11 is not a distinguishable difference.

### What 100 generations bought over 6

The v6 pilot searched 6 generations; v7 searched 100, evaluating 4,500
architectures over 13 days. The effect on the final metrics was small:

| Metric | v6 pilot | v7 production |
|---|---|---|
| Recall | 0.6250 | 0.6250 |
| Coverage % | 18.24 | 18.75 |
| FAR/day | 2.314 | 2.269 |
| F1 | 0.0597 | 0.0610 |
| NAB | 27.93 | 29.07 |
| Mean TTD | 96.69 | 102.89 |

The four baselines are bit-identical between the two studies, so the comparison
is clean. Recall did not move at all; the gains are in alarm economy and warning
time, and they are modest.

**State this honestly in the paper.** Combined with section 4, the picture is
that the event-schedule correction produced the result and the architecture
search refined it. A two-week search buying a 2% FAR improvement is a finding
about the method's sensitivity, not an embarrassment — but presenting the search
as the source of the result would not survive scrutiny.

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
persistence and reconstruction quality — entangled with every axis.

Objectives are now, all label-free so none depends on the single cycle-0 failure:

1. **`SMAPE` reconstruction error** — did the model learn normal behaviour at all
2. **`calibration_level_v1`** — how far the false-alarm rate sits from nominal 5%
3. **`calibration_drift_v1`** — how much that rate shifts across the window

Weights `[0.30, 0.35, 0.35]`. Alarm burden is still measured and reported.

### The production search settled the open caveat

The v7 pilot left one worry: `level` and `drift` separated candidates across
only about 9% of their range, so SMAPE might dominate ranking and the search
might converge to something SMAPE-optimal and calibration-mediocre.

Measured over all 4,500 production candidates, that did not happen. Objective
independence held from start to finish:

| | gen 35 | gen 81 | gen 100 |
|---|---|---|---|
| worst pair, `level` vs `drift` | 0.320 | 0.326 | **0.339** |
| `error` vs `level` | −0.019 | +0.017 | **+0.019** |

Against v6's entangled 0.547. Error and level are effectively independent.

The winner, `nsga3_gru_3x3_latent37`, ranks:

| Objective | Value | Rank of 4,500 | Distance from best, as % of population range |
|---|---|---|---|
| `error` (SMAPE) | 0.9495 | 263 | 5.2% |
| `level` | 0.3664 | 998 | 8.4% |
| `drift` | 0.0727 | 864 | 17.9% |

The predicted pattern is visible — SMAPE is the strongest axis — but the winner
is in the top 20% on both calibration terms, not near the median. It is a
balanced compromise, and the caveat can be reported as tested and not realised.

**The search converged early.** Best-per-objective values were identical at
generation 35 and generation 100 across 2,000 additional candidates. No new
champion appeared in 65 generations. Pareto front: 1,398 of 4,500.

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

2. **Baseline convergence: tested and closed.** The worry was that the
   baselines were undertrained. Under `patience: 2` cycle-0 training ran 3
   epochs for `sae_static` (best at 1), 6 for NiaNetVAE (best at 4), and 8 for
   `vae_static` (best at 6). The contributed method trained **less** than the
   VAE baseline under an identical budget and stopping rule, so there was never
   evidence of bias; the live question was **power**.

   Study `metropt_v7_patience_diag` re-ran both static autoencoders at
   `patience: 10`, changing nothing else. Its data contract fingerprint matches
   v7, so both score the same 1,146,067 anchors.

   | Model | Epochs at patience 2 | at patience 10 | Outcome |
   |---|---|---|---|
   | `sae_static` | 3 | 19 | Metrics unmoved; recall still 0.4375 |
   | `vae_static` | 8 | 30 | Gained one detection; recall 0.5625 to 0.6250 |

   `sae_static` is weak on its merits, not from lack of training. `vae_static`
   genuinely had been undertrained, but its coverage rose to 25.90% and it
   remains `fallback`. **The ordering did not change.**

   The diagnostic also strengthened the study: `vae_static` now sits at exactly
   NiaNetVAE's recall, so the two can be compared like for like. At recall
   0.6250, NiaNetVAE wins on coverage (18.75 vs 25.90), FAR/day (2.269 vs
   2.978), F1 (0.0610 vs 0.0473), NAB (29.07 vs 24.37) and mean TTD (102.9 vs
   96.9).

   Two residual limits, to be stated rather than investigated: `vae_static` hit
   the `max_epochs: 30` ceiling with its best at epoch 23, so where it converges
   is unknown; and the diagnostic covered only the baselines, since the
   diagnostic study has no search artifacts, so NiaNetVAE was not re-run under a
   longer patience.

3. **Cycle 0 is unrepresentative.** Never fine-tuned, accumulates 41 days of
   drift, and its pre-failure risk is *inverted*. The v7 winner's recorded
   `smoothed_auroc` on cycle 0 is 0.292, below chance, consistent with every
   candidate ever measured. Adapted cycles are correct in 11 of 13. This is
   evidence for adaptation, but it also means architecture selection is guided by
   the worst available cycle.

4. **Architecture ranking rests on a 4-epoch fit.** Candidates are capped at
   `candidate_max_epochs: 4`; the winner used all four with its best at epoch 2.
   The selected architecture is then trained up to 30 epochs in phase 2. The
   cheap proxy is deliberate, but the gap between proxy and final training is
   large and unmeasured.

5. **Calibration goes stale immediately** — 38% exceedance against a nominal 5%.
   Every one of the 4,500 candidates over-fires; none under-fires.

6. **Architecture and adaptation are not fully separable.** There is no static
   NiaNetVAE control. The IForest pair isolates adaptation for a classical model
   only. Frame the study as a controlled system comparison, not a causal ablation.

## 9. Documentation

The `PhD` collection at `wiki.kvaltko.com` is current as of 2026-09-10. Nine
pages covering research question, dataset and failure labels, drift evidence,
system map, architecture search, evaluation framework, reproducibility and
provenance, experimental evidence, and open problems.

Result tables are **generated, never typed**:

```bash
python docs/build_results.py artifacts/<study_id> -o results.md
```

Note that this needs Python 3.10 or newer. The HPC login node runs 3.9, so run
it inside the SIF:

```bash
singularity exec -e -B /d/hpc/home/sasop/NiaNetVAE:/work --pwd /work \
    /d/hpc/home/sasop/images/nianet-a57acab.sif \
    python docs/build_results.py artifacts/metropt_controlled_v7 -o /work/results.md
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
