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

**Dump directories.**
- `dumps/nmmr_04-28-09-51-29/` — oracle_baseline, sequential
- `dumps/nmmr_04-28-09-52-17/` — baseline, sequential
- `dumps/nmmr_04-28-09-52-52/` — oracle_modified, sequential
- `dumps/nmmr_04-28-10-05-25/` — modified, parallel (-t 8)
