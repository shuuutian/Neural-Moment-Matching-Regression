# MAR-NMMR — research diary

A running log of validation runs and notable findings. Each entry has a date,
the run config, the result summary, and a brief interpretation. Raw dump
directories live under `/dumps/<timestamp>/` (gitignored).

---

## 2026-04-28 — Part B opener: four-role validation at n=5000, 100 reps

**Context.** First full-scale validation pass after Pass 1–4 implementation
(commits `bd8c50f` … `94917e6`). Sanity-checking that the MAR-NMMR estimator
behaves as the §4.4 plug-in advertises before scaling out further.

**Setup.** All four roles run on the demand DGP (`name="demand_mar"`):
`n_sample=5000, missing_rate=0.3, Z_noise=W_noise=1`, network 4×80, lr=3e-3,
l2=3e-6, U-statistic loss. Placeholder roles run minibatched
(n_epochs=30, batch_size=1000 → 5×30=150 SGD steps). MAR roles run full-batch
(n_epochs=150, batch_size=5000 → 1×150=150 SGD steps). Step counts matched per
the Pass 4 schedule heuristic.

Run as `uv run python main.py configs/mar_pci_configs/<role>.json ate -t 8`,
exploiting the new `ProcessPoolExecutor` rep-level parallelism (this entry
was the trigger for adding it). 100 reps in ~3 min wall on the 12-core CPU.

**Headline result (per-rep MSE between predicted ATE curve and structural):**

| role | MSE mean | MSE median | MSE std | bias L2 |
|---|---:|---:|---:|---:|
| oracle_baseline | 60.40 | 57.10 | 23.25 | 7.18 |
| baseline (mar_naive) | 74.02 | 60.98 | 61.24 | 5.04 |
| oracle_modified | 64.82 | 61.05 | 23.06 | 7.61 |
| modified | 86.58 | 80.37 | 45.91 | 7.85 |

Ratios:
- `oracle_modified / oracle_baseline = 1.073` — small gap; consistent with the
  matched-step asymmetric schedule not being a perfect equaliser.
- `modified / oracle_modified = 1.336` — gap from the bias-correction under
  partial data.
- `modified / baseline = 1.170` — **the bias-corrected estimator does NOT beat
  the naive drop-rows baseline on this metric at this scale.**

**Mean ATE curve (predicted β̂(a) averaged over 100 reps; structural in
last column):**

| a | oracle_baseline | baseline | oracle_modified | modified | structural |
|---:|---:|---:|---:|---:|---:|
| 10.00 | 73.69 | 69.29 | 74.90 | 76.39 | 56.43 |
| 12.22 | 70.62 | 67.02 | 71.58 | 73.45 | 61.29 |
| 14.44 | 67.33 | 64.62 | 68.07 | 70.22 | 63.49 |
| 16.67 | 63.75 | 61.95 | 64.31 | 66.63 | 63.62 |
| 18.89 | 59.83 | 58.99 | 60.26 | 62.67 | 62.25 |
| 21.11 | 55.62 | 55.75 | 55.93 | 58.43 | 59.65 |
| 23.33 | 51.17 | 52.30 | 51.38 | 54.01 | 55.94 |
| 25.56 | 46.58 | 48.72 | 46.69 | 49.53 | 51.68 |
| 27.78 | 42.17 | 45.26 | 42.14 | 45.25 | 47.20 |
| 30.00 | 38.27 | 42.12 | 38.06 | 41.38 | 42.67 |

**Observations.**

1. **All four roles fail to capture the structural curvature.** The structural
   ATE peaks at a≈16.7 (β≈63.6), but every role produces a monotonically
   decreasing curve from ~70+ at a=10 to ~38–42 at a=30. This is a shared
   feature, not a regression — likely a model-class / training-budget limit
   of the upstream demand setup at this scale, not anything algorithmic in
   Pass 1–4.

2. **`oracle_modified` ≈ `oracle_baseline`** (MSE +7%). The MAR machinery is
   mostly inert under δ ≡ 1 (as designed: when δ=1 everywhere the imputation
   term drops out of both the loss and the predict). The +7% MSE gap is
   plausibly explained by the matched-step heuristic failing to fully equate
   full-batch and minibatched optimization trajectories. **OK so far.**

3. **`baseline` has lower bias L2 than `oracle_baseline`** (5.04 vs 7.18) but
   per-rep variance ~2× higher (std 11.5 vs 5.4 at a=10). MSE blows up because
   the variance penalty dominates: 74.0 > 60.4. The "bias" win is largely
   because dropping rows under MAR reweights the conditional distribution in
   a way that happens to bracket the mid-range structural values better.

4. **`modified` has the highest MSE (86.58)** of the four roles — the bias
   correction pays a per-rep MSE penalty here. Looking at the bias column:
   - At low a (10), modified bias = 19.96, the **largest** of any role.
   - At mid a (18–21), modified bias is small (≤1.2 in absolute) — the
     **smallest** of any role.
   - At high a (28–30), modified bias is ~−1 to −2, similar to baseline,
     better than the oracles.
   So the q̂ smoother is doing *something* — at mid/high a it produces the
   tightest bias of any role — but at low a it amplifies the existing
   overshoot. The standalone-MSE comparison hides this structure.

5. **Variance under `modified` (std≈8.5 at a=10) is between `oracle_modified`
   (std≈5.0) and `baseline` (std≈11.5).** Consistent with using more rows
   than baseline (variance reduction) but with imputation noise added on top.

**Provisional read.** The implementation is doing what the math says — the
cross-fit q̂ smoother is contributing, oracle_modified collapses correctly to
oracle_baseline. But the headline metric (MSE on the ATE curve at n=5000) does
not yet show the bias-correction story the paper predicts. Three plausible
causes, in order of perceived likelihood:

  (a) **Sample size / step count.** Cross-fit nuisance estimation needs more
      observed neighbours per fold; bandwidth heuristic may also be looser
      than ideal at n=5000. Try n=10000 and/or n_epochs=300 for the MAR path.
  (b) **Bandwidth.** Median-pairwise-distance on standardised L⁺ is a default,
      not tuned. Sweep `mar_bandwidth ∈ {0.5×, 1×, 2×}`.
  (c) **Outcome-only DGP.** The demand DGP has no covariate X, so L⁺ = (A,Z,Y)
      and L = (A,Z). Smoothing over Y in q̂ when Y is itself the regression
      target may add the bias we're seeing. Worth diagnosing.

**Followups (Part B continuation).**
- [ ] Re-run all four roles at n=10000 to test sample-size sensitivity.
- [ ] Sweep `mar_bandwidth` ∈ {0.5, 1, 2} × default on the modified config.
- [ ] Sweep MAR-path n_epochs ∈ {150, 300, 600} on oracle_modified to bound
      the schedule-asymmetry contribution.
- [ ] Compare per-rep ATE curves of modified vs baseline at the same seed —
      is modified always-biased-low-a or only sometimes?
- [ ] Run `nmmr_u_repro` (`configs/demand_noise_configs/nmmr_u_demandnoise.json`)
      for the upstream Figure 3 sanity check; confirm 60.4 for oracle_baseline
      is roughly what upstream gets.
- [ ] Confirm with Wen Zhou's paper (paper §5) what β̂(a) MSE the demand DGP
      gives at n=5000 in the full-data setting, to calibrate expectations.

**Plots.** `plots/oos/bias_distribution.png`, `plots/oos/ate_curves.png`.

**Dump directories.**
- `dumps/nmmr_04-28-09-51-29/` — oracle_baseline, sequential
- `dumps/nmmr_04-28-09-52-17/` — baseline, sequential
- `dumps/nmmr_04-28-09-52-52/` — oracle_modified, sequential
- `dumps/nmmr_04-28-10-05-25/` — modified, parallel (-t 8)

---

## 2026-04-28 — In-support test range [22, 36]: OOD hypothesis confirmed

**Hypothesis.** Training prices follow N(27.15, 6.62²) so the 5th–95th support
is roughly [15.7, 36.5]. The default test grid [10, 30] places its first 3
points at a ∈ {10.0, 12.2, 14.4} *below* the 5th percentile — well outside the
training support. The systematic overshoot we saw at low a in the previous
entry could simply be the model extrapolating where it has no signal.

**Action.** Added `test_a_min` / `test_a_max` / `test_n_grid` knobs to
`generate_test_demand_pv_mar` (defaults preserve [10, 30] behaviour). New
configs `configs/mar_pci_configs/{baseline,oracle_baseline,oracle_modified,modified}_insupport.json`
with the test grid moved to [22, 36] (10 points), keeping all other config
knobs identical to the OOS run.

**Result.** Same 100-rep validation, only the test grid changed:

| role | OOS [10, 30] MSE | **IS [22, 36] MSE** | OOS bias L2 | **IS bias L2** | OOS mean bias | **IS mean bias** |
|---|---:|---:|---:|---:|---:|---:|
| oracle_baseline | 60.40 | **19.45** | 7.18 | 4.20 | +0.48 | −4.07 |
| baseline | 74.02 | **15.00** | 5.04 | 2.55 | +0.18 | −0.82 |
| oracle_modified | 64.82 | **19.75** | 7.61 | 4.29 | +0.91 | −4.23 |
| modified | 86.58 | **6.84** | 7.85 | 1.51 | +3.37 | −1.03 |

**Headline reversal.** The MSE ranking *flips* between OOS and IS:

  - OOS [10, 30]: `oracle_baseline (60.4) < oracle_modified (64.8) <
    baseline (74.0) < modified (86.6)`
  - IS [22, 36]:  `modified (6.84) < baseline (15.0) <
    oracle_baseline (19.5) ≈ oracle_modified (19.8)`

In support, **`modified` is the best estimator of the four**, beating
`baseline` 2.2× on MSE and beating both oracles 2.9×. The bias-correction
story the §4.4 plug-in tells is being borne out — but only when the test
grid sits inside the training support.

**What the plots show**
(`plots/insupport/bias_distribution.png`, `plots/insupport/ate_curves.png`):

  - Bias distribution is clearly bimodal in IS: a left mode at ~−4 contains
    `oracle_baseline` + `oracle_modified` (systematic underestimation across
    the in-support range), and a right mode near 0 contains `baseline` +
    `modified`. The two oracles' distributions are tighter (lower variance)
    but biased; the MAR-friendly methods are wider but centered on truth.
  - ATE-curve plot: the two oracles' mean curves track ~3–5 below the
    structural curve across [22, 36]; `modified` and `baseline` overlap the
    structural curve closely.

**Why the inversion?** Two things going on, both consistent with the data.

  1. **OOS overshoot dominates the [10, 30] MSE.** The structural ATE peaks
     at ~63 around a=17, then decreases. At a ∈ [10, 14] (heavily OOD), all
     four methods extrapolate to values 17–20 above truth. That single tail
     contributes ~50% of the per-curve MSE in the OOS metric, and it
     contaminates `modified` worst because the q̂ smoother amplifies the
     extrapolation. Strip those points (move to [22, 36]) and the metric
     becomes a clean comparison on the in-support fit, where `modified` is
     legitimately better.
  2. **Oracles systematically underestimate in the right half of the
     support.** Both oracle paths converge to a flatter curve than the
     truth in [22, 36] — they undershoot by ~4 across the range. The MAR
     loss (full-batch SGD over the imputed-residual U-statistic) ends up
     in a different optimum that fits this part of the curve better. The
     matched-step heuristic (Pass 4) controls *total* step count but not
     full-batch vs. minibatched optimizer trajectory; that asymmetry is
     visible here as a different inductive bias, not just a noise-level
     difference. Verifying this is one of the followups.

**Updated read.** Pass 1–4 implementation looks correct. The estimator
delivers what the §4.4 paper predicts on data within the training support.
The OOS pathology is real but largely an artefact of the default test grid
straddling the training support boundary at the low end.

**Followups (refined from 2026-04-28-am).**
- [ ] Investigate why oracle paths underestimate in [22, 36] — minibatched
      vs. full-batch optimization difference, or something deeper? Try
      bumping oracle_baseline `n_epochs` to 150 with `batch_size=5000`
      (full-batch) to isolate the optimization-trajectory contribution.
- [ ] Plot bias *per a-value* (not just per-rep mean bias) so the
      OOD-extrapolation failure at a∈[10, 14] is visible alongside the
      in-support fit quality.
- [ ] Bandwidth sweep on `modified` (defer until the optimization
      trajectory question is resolved — bandwidth is a smaller effect).
- [ ] Confirm with paper §5 / Wen Zhou what scale of MSE the demand DGP
      gives at n=5000 in the full-data setting on a similar test grid,
      to calibrate expectations.

**Plots.** `plots/insupport/bias_distribution.png`,
`plots/insupport/ate_curves.png`.

**Dump directories.**
- `dumps/nmmr_04-28-10-46-30/` — oracle_baseline_insupport
- `dumps/nmmr_04-28-10-46-56/` — baseline_insupport
- `dumps/nmmr_04-28-10-47-15/` — oracle_modified_insupport
- `dumps/nmmr_04-28-10-50-50/` — modified_insupport

---

## 2026-04-28 — Oracle underestimate diagnostic: optimizer/step-count rejected

**Question carried in from the previous entry.** Both oracle paths
(`oracle_baseline`, `oracle_modified`) systematically underestimate β(a) by
~4 across [22, 36]. The leading hypothesis was *optimizer-trajectory
asymmetry*: the placeholder path is minibatched (bs=1000, 30 ep × 5 mb = 150
SGD steps), while the MAR path is full-batch (bs=5000, 150 ep × 1 mb = 150
SGD steps). Pass 4 matched step count but not trajectory, and full-batch
SGD on the MAR-imputed loss might converge to a different minimum than
minibatched SGD on the upstream loss.

**Two diagnostics run in parallel** (n=5000, 100 reps, test∈[22, 36],
otherwise identical to the four-role IS run):

| Tag | Path | Schedule | Total SGD steps | Tests |
|---|---|---|---:|---|
| **D1** `oracle_baseline_fullbatch` | placeholder, no MAR loss | bs=5000, n_epochs=150 | 150 | optimizer trajectory |
| **D2** `oracle_baseline_5xsteps`   | placeholder, no MAR loss | bs=1000, n_epochs=150 | 750 | under-convergence |

Both compared against the existing `oracle_baseline_insupport` (minibatch,
150 steps) and `oracle_modified_insupport` (full-batch, 150 steps, MAR loss
inert under δ ≡ 1).

**Headline result.** Neither hypothesis explains the bias. All three
oracle_baseline variants land on essentially the same fit:

| config | MSE_mean | bias_mean | bias@22 | bias@28 | bias@36 |
|---|---:|---:|---:|---:|---:|
| oracle_baseline (minibatch, 150 steps)       | 19.45 | −4.07 | −4.41 | −4.96 | −2.02 |
| **D1**: fullbatch, 150 steps                  | 19.85 | −4.23 | −4.15 | −5.02 | −2.69 |
| **D2**: minibatch, 750 steps (5×)             | 17.52 | −3.97 | −4.96 | −4.61 | −1.96 |
| oracle_modified (fullbatch, 150 steps, MAR)   | 19.75 | −4.23 | −4.14 | −5.02 | −2.68 |

Read out:
- D1 vs minibatch oracle: bias differs by 0.16 — well within Monte Carlo
  noise. Optimizer trajectory **does not** drive the bias.
- D2 vs minibatch oracle: bias differs by 0.10 even with 5× more steps.
  Under-convergence **does not** drive the bias either.
- **D1 ≈ oracle_modified to two decimal places at every grid point.** This
  is a clean runtime confirmation that the MAR loss reduces algebraically
  to the upstream loss under δ ≡ 1, as the Pass 2 algebra check claimed.

**The bias has structure across a.** It's not a flat additive shift; the
bias is U-shaped, deepest at a≈26–28 (~−5) and shallower at the endpoints
(~−2 at a=36). The full-data oracle fit *systematically misses curvature*
in the mid-range of [22, 36]. See `plots/insupport/bias_per_treatment.png`
and `plots/oracle_diag/bias_per_treatment.png`.

**Updated read.** The systematic ~−4 oracle bias is intrinsic to the
upstream-NMMR fit on this DGP at n=5000 with the upstream hyperparameters.
It is NOT specific to the MAR machinery, NOT an artefact of the matched-
step heuristic from Pass 4, and NOT relievable by a 5× training-step
budget.

**What remains as candidate causes (refined):**
1. **Model class + kernel choice mismatch with the demand-DGP curvature.**
   The kernel uses `length_scale=1` in (A, Z) space (upstream default;
   `kernel_utils.rbf_kernel`) while A has SD≈6.62 — the kernel is heavily
   localized. The U-statistic loss on a tightly-localized kernel may give
   a fit that smooths out exactly the kind of mid-range curvature we see
   missing here. This is upstream behavior, not a MAR-NMMR artefact.
2. **Demand-DGP scale.** Wen Zhou's paper §5 reports their results at this
   DGP; we have not yet calibrated against their numbers. The −4 mean bias
   might match upstream-expected performance.
3. **Test-grid / MC plug-in interaction.** `cal_structural` Monte-Carlos β(a)
   over 10000 demand draws (uniform on [0, 10]) → views drawn from
   7·ψ(demand)+45+N(0,1). If the val W's used by `_predict_placeholder`
   come from a slightly different W marginal than `cal_structural` assumes
   (e.g. tail effects), there's an extra MC bias on top of any model-fit
   bias. Worth eyeballing.

**Followups (refined).**
- [ ] Run upstream `nmmr_u_repro` (`configs/demand_noise_configs/nmmr_u_demandnoise.json`)
      at the (Z=1, W=1) grid point and compare its Figure 3 numbers to our
      `oracle_baseline`. If upstream reports the same ~−4 bias, the
      observation is calibration, not regression.
- [ ] Sanity-check val-W vs `cal_structural`-W marginals: confirm whether
      `_predict_placeholder` is averaging over a representative W population.
- [ ] **Defer** the bandwidth sweep on `mar_bandwidth` until after the
      upstream control above — bandwidth only affects q̂, and q̂ doesn't
      enter the oracle bias.
- [ ] Bigger network (e.g. 6×128) on `oracle_baseline_insupport` to test
      capacity hypothesis.

**Plots.**
- `plots/insupport/bias_per_treatment.png` — new per-a bias curve for the
  four-role IS run; shows the U-shape clearly.
- `plots/oracle_diag/{bias_per_treatment,ate_curves,bias_distribution}.png`
  — six lines, the three oracle variants (minibatch, fullbatch, 5× steps)
  overlapping each other across [22, 36] while modified/baseline diverge.

**Plotter changes** (`scripts/plot_validation.py`).
- Now accepts `path::label` to override the auto-detected role tag, so
  multiple dumps with the same data.mode + use_mar_modified can coexist
  in one plot (e.g. three oracle variants).
- Added `bias_per_treatment.png` output (mean ±1 std bias curve per role).

**Dump directories.**
- `dumps/nmmr_04-28-11-05-44/` — D1 oracle_baseline_fullbatch
- `dumps/nmmr_04-28-11-12-24/` — D2 oracle_baseline_5xsteps

---

## 2026-04-28 — Upstream NMMR-U control: byte-identical match → bias is upstream

**Question.** Carry over from previous entry: the systematic oracle bias might
be intrinsic to upstream NMMR-U on this demand DGP, not a regression from the
MAR wrapper. Settled by running the upstream `data.name = "demand"` pipeline
(no MAR machinery — `NMMR_Trainer_DemandExperiment`, `generate_train_demand_pv`)
at the same (Z=1, W=1, n=5000, 100 reps) point on both test grids, and
comparing per-rep predictions to our `oracle_baseline`.

To make the [22, 36] view possible from the upstream pipeline, extended
`generate_test_demand_pv` to accept the same `test_a_min/max/n_grid` knobs
already supported by the MAR version.

**Configs.** [`configs/mar_pci_configs/upstream_demand_oos.json`](configs/mar_pci_configs/upstream_demand_oos.json)
and `upstream_demand_is.json`. Same hyperparameters as our placeholder oracle
(n_epochs=30, batch_size=1000, lr=3e-3, l2=3e-6, U-statistic, MLP 4×80).

**Result. Byte-identical match across all 100 reps and all 10 grid points
in both views:**

| | OOS [10, 30] |  | IS [22, 36] |  |
|---|---:|---:|---:|---:|
| | upstream | oracle_baseline (MAR) | upstream | oracle_baseline_insupport |
| MSE_mean    | 60.40   | 60.40   | 19.45   | 19.45   |
| bias_mean   | +0.481  | +0.481  | −4.074  | −4.074  |
| max abs diff per (rep, a) | — | **0.0000** | — | **0.0000** |

The two pipelines produce identical predictions modulo floating-point — as
expected from the structural argument: under δ ≡ 1 the MAR wrapper changes
nothing observable about training (it just ignores `delta_w`). The fact that
this holds *byte-identically* over 1000 (rep, a) pairs is a strong sanity
check that the data generation, RNG seeding, trainer dispatch, and predict
plug-in are all carried through faithfully.

**Conclusion. The systematic bias structure — overshoot at low a (+17 at
a=10), undershoot at mid/high a (−5 at a=27), monotone-decreasing fit that
misses the structural curve's peak at a≈16.7 — is upstream NMMR-U behavior
on this DGP at the upstream hyperparameters.** It is *not* introduced by
the MAR extension, the §4.4 plug-in, or any choice we made.

**What this changes about Part B's interpretation.**
- The `modified` role's IS [22, 36] win (MSE 6.84 vs oracle 19.45) is a
  legitimate bias-correction story: the §4.4 plug-in *partially* corrects
  the upstream NMMR-U fit's curvature mismatch, not just the missingness.
- The `oracle_baseline` ≈ `oracle_modified` agreement at the IS bias of
  −4 is now firmly attributed to a shared *upstream* fit pathology rather
  than a shared *training-pipeline* pathology.
- The remaining open question is no longer "is our MAR pipeline buggy" —
  it is "why does upstream NMMR-U miss the peak at this DGP/scale". This
  is a question about the upstream method, not our extension.

**Followups.**
- [ ] Bandwidth sweep on the U-statistic kernel (`length_scale ∈ {1, 2, 5,
      10}` or median-heuristic on (A, Z) raw space) — leading suspect for
      the missed peak. Run on upstream `demand` since it's the same fit.
- [ ] Optionally: input standardisation of (A, W, Z) before training. Other
      NMMR-style codebases standardise; ours doesn't.
- [ ] **Independent of the bias question:** patch the off-by-one ReLU bug
      in `MLP_for_NMMR.forward` (output layer applies ReLU because
      `ix == network_depth + 1` is never true; should be `ix == network_depth`).
      Verified empirically the fix doesn't change the bias on this DGP, but
      the constraint h(W, A) ≥ 0 is unintended and could bite on other DGPs.

**Code change for this entry.**
- `src/data/ate/demand_pv.py` — `generate_test_demand_pv` now accepts
  `test_a_min/max/n_grid` (kwargs default to upstream behavior, [10, 30] × 10).

**Dump directories.**
- `dumps/nmmr_04-28-17-21-37/` — upstream demand at [10, 30]
- `dumps/nmmr_04-28-17-30-41/` — upstream demand at [22, 36]

---

## Consistency investigation — moved out

The consistency-theorem experiments (sample-size sweep at fixed schedule,
paper-schedule rerun with growing-network, theoretical analysis of NMMR
Theorem 1's d_k² approximation-error term) have been parked in a separate
file: [`CONSISTENCY_INVESTIGATION.md`](CONSISTENCY_INVESTIGATION.md).

Status: parked pending audit of the bias-evaluation pipeline. Issues with
the prior conclusions are not yet explained. See the new file's header for
the audit checklist and open followups.
