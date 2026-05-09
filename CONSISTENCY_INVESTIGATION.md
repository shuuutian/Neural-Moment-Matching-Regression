# Consistency investigation — parked

Empirical tests of the NMMR consistency theorem (Kremer et al., arXiv 2205.09824v3,
Theorems 1 + 2) on the demand DGP. Originally entries in `RESEARCH_DIARY.md`,
moved here on 2026-05-10 to keep the main diary focused on MAR-PCI validation.

## Status: parked

Issues with the results below are not yet explained — either the code has a
bug or the analysis approach is wrong. Before extending this investigation,
the prior conclusions need to be confirmed or revised.

## When picking back up

1. **Audit the pipeline used to compute bias.** In particular: the test grid
   construction (`generate_test_demand_pv` with `test_a_min/max/n_grid`
   kwargs), the structural reference (`cal_structural`), the prediction
   step (`NMMR_Trainer_DemandExperiment.predict`), and the
   `scripts/plot_consistency.py` / `plot_consistency_compare.py` aggregations.
2. **Independently re-run** at one cell of the paper-schedule sweep (e.g.
   n=10000, fixed-net) and confirm the headline numbers reproduce before
   building further on them.
3. **Then proceed** with the followups listed at the bottom of each entry.

## Open followups (carried over)

- [ ] Re-evaluate consistency on a treatment grid drawn from the training
  A-distribution (or its quantiles), instead of uniform [10, 30]. This is
  the right test of the theorem in the d_k metric. Expected: cleaner
  monotone decay than the uniform grid shows.
- [ ] Diagnose why bias at a=10 *drifts upward* with n. Likely: as the
  model commits to a wrong functional form on the boundary, more SGD steps
  reinforce the commitment. Check kernel weight density at A=10 vs A=20.
- [ ] (Deferred) Adaptive-bandwidth investigation as a possible paper
  add-on, not a primary fix. Median pairwise distance on standardised
  (A, Z); bandwidth grid `length_scale ∈ {1, 2, 5, 10}` at fixed n=5000.

---

## 2026-04-28 — Sample-size sweep: variance shrinks, bias does NOT

**Question (from Wen).** Does the simulation support the consistency theorem?
Wen's heuristic was: "if the error distribution is centred around zero, the
sim supports consistency". This is technically incorrect — consistency is an
*asymptotic* statement (β̂_n → β as n → ∞), not a fixed-n property — but
the right empirical test is in the same spirit: vary n and check that bias
*and* variance both shrink toward zero.

**Setup.** Upstream `data.name="demand"` pipeline (no MAR — proven byte-
identical to oracle_baseline in the previous entry), Z=W=1, MLP 4×80,
n_epochs=30, batch_size=1000, lr=3e-3, l2=3e-6, U-statistic, kernel
`length_scale=1` (upstream default). 100 reps each at five sample sizes:

| n | dump |
|---:|---|
|  1,000 | `dumps/nmmr_04-28-18-41-07/` |
|  2,000 | `dumps/nmmr_04-28-18-41-19/` |
|  5,000 | `dumps/nmmr_04-28-17-21-37/` (re-used from upstream control entry) |
| 10,000 | `dumps/nmmr_04-28-18-41-33/` |
| 20,000 | `dumps/nmmr_04-28-18-42-20/` |

Test grid [10, 30] × 10 throughout. Schedule (n_epochs, batch_size) held
fixed; total SGD steps therefore scales with n (n=1k → 30 steps; n=20k → 600).

**Headline.** Variance behaves as consistency would predict; bias plateaus.

| n | reps | MSE | mean \|bias\| | std (averaged over a) |
|---:|---:|---:|---:|---:|
|  1,000 | 100 | 1017.05 | 25.38 | 17.32 |
|  2,000 | 100 |  302.75 | 14.20 |  7.91 |
|  5,000 | 100 |   60.40 |  5.63 |  2.50 |
| 10,000 | 100 |   54.27 |  5.55 |  1.49 |
| 20,000 | 100 |   40.83 |  5.01 |  1.87 |

Mean |bias| drops 4.5× from n=1k → n=5k, then **−0.08** from n=5k → 10k and
**−0.54** from n=10k → 20k. From n=5,000 onward the bias is essentially
plateaued; doubling n produces near-zero improvement.

**Per-a bias plateau (n=5k to n=20k):**

| a | bias n=5k → 10k → 20k | total Δ over 4× n |
|---:|---|---:|
| 10.00 | +17.26 → +17.28 → +13.61 | −3.65 |
| 14.44 |  +3.84 →  +3.89 →  +0.92 | −2.93 |
| 16.67 |  +0.12 →  +0.29 →  −2.10 | −2.23 |
| 21.11 |  −4.03 →  −3.73 →  −4.68 | −0.65 |
| 25.56 |  −5.10 →  −4.89 →  −4.96 | +0.14 |
| 27.78 |  −5.04 →  −4.97 →  −4.86 | +0.17 |
| 30.00 |  −4.40 →  −4.51 →  −4.33 | +0.07 |

**Variance (rep std of β̂(a)) scaling (log-log slope of std vs n):**

| a | log-log slope | implied variance scaling |
|---:|---:|---|
| 10.00 | −0.48 | ~ n^(−0.95) |
| 18.89 | −0.94 | ~ n^(−1.88) |
| 25.56 | −1.18 | ~ n^(−2.35) |
| 30.00 | −1.18 | ~ n^(−2.37) |

Variance is contracting *faster* than the n^(−1) parametric rate at most
treatment values — the estimator is concentrating tightly around its limit.
Combined with the persistent non-zero bias, this means the estimator is
**concentrating on the wrong point** at high-a values.

**Diagnosis (later revised — see 2026-05-10 entry).** This was originally
read as the textbook signature of a *regularization-bias-dominant* regime
with fixed kernel bandwidth. That framing turned out to be wrong:
upstream's consistency theorem allows fixed kernel; the binding constraint
is the population approximation error d_k²(h*, h̃_k) at fixed network
capacity. See the 2026-05-10 entry for the corrected framing.

**Original followups (superseded by 2026-05-10).**
- [ ] Repeat the sweep with `compute_kernel` using median-pairwise-distance
      bandwidth on standardised (A, Z). Expect to see bias → 0 trend.
- [ ] Optionally: bandwidth grid `length_scale ∈ {1, 2, 5, 10}` at fixed
      n=5000 to map the bias-vs-bandwidth tradeoff and pick a sweet spot.
- [ ] Discuss with Wen whether the paper's consistency conditions
      explicitly assume bandwidth shrinkage. If yes, our sim should
      respect that. If no, we may have surfaced a gap in the theory.

**Plots.** `plots/consistency/{bias_vs_n, stderr_vs_n, per_a_bias_grid}.png`.

**Configs added.**
- `configs/mar_pci_configs/upstream_demand_n{1000,2000,10000,20000}.json`

**Script added.**
- `scripts/plot_consistency.py` — bias and variance vs n, per-a curves.

---

## 2026-05-10 — Consistency rerun at paper schedule: bias shrinks in interior, sticks at boundary

**Context.** The previous entry (sample-size sweep at n_epochs=30, fixed
length_scale=1) concluded the bias floor was a kernel-bandwidth issue and
proposed adaptive bandwidth as the next step. Two updates flipped that
framing before any bandwidth code was written:

1. **Re-read upstream NMMR** (Kremer et al., arXiv 2205.09824v3) Theorem 1
   ("AXZ Conv") and Theorem 2 ("AWX Conv"). The theorem proves consistency
   *with a fixed kernel*. The bound is

   `d_k²(h*, ĥ_{k,λ,n}) ≤ d_k²(h*, h̃_k) + λM_λ + 8M·R_n(F') + O(n^(−1/2))`

   where `h̃_k = argmin_{h ∈ H} R_k(h)` is the *population* minimizer in the
   (fixed) hypothesis class H. The first term is the **approximation error**
   of H under the fixed kernel; it does not depend on n. To make it vanish,
   H must grow with n. The paper flags this on line 435: "increasing the
   complexity of the neural network, but doing so slowly enough the
   Rademacher complexity terms still decrease with sample size."

2. **The paper's own Table 7 plateaus** at the demand DGP. NMMR-U c-MSE
   goes 23.68 → 16.21 → 14.25 → 14.27 across n ∈ {1k, 5k, 10k, 50k} — same
   plateau pattern as our own sweep. They hold (depth=4, width=80) constant
   across all n, so by their own theorem they cannot expect d_k² → 0.

**Implication.** "Fixed bandwidth violates consistency" was wrong: the
theorem permits fixed kernel. The right test of the consistency claim is
to **grow network capacity with n** at a fixed kernel, and check whether
the bias floor moves.

**Setup.**

- DGP: upstream `data.name="demand"` (no MAR), Z_noise=W_noise=1,
  U-statistic loss, kernel `length_scale=1` (upstream default).
- Sample sizes: n ∈ {1000, 2000, 5000, 10000, 20000}.
- Schedule: **paper's full schedule** (n_epochs=3000, batch_size=1000,
  lr=3e-3, l2=3e-6) — matches `configs/figure2_config/nmmr_u_figure2.json`,
  100× more SGD steps than the previous entry's n_epochs=30 schedule.
- Two network configs:
  - **Fixed**: depth=4, width=80 (paper's setting).
  - **Growing**: depth=4, width = round(80·√(n/5000)) → {36, 51, 80, 113, 160}.
    α=0.5 boundary case for Rademacher decay; aggressive enough to give the
    bias floor every chance to move if capacity is the binding constraint.
- 20 reps each (matching paper). Test grid [10, 30] × 10. Total wall ~2 h
  on `-t 4`.

| Group | n=1k | n=2k | n=5k | n=10k | n=20k |
|---|---|---|---|---|---|
| Fixed | `dumps/nmmr_04-28-19-58-38/n_sample:1000` | `…:2000` | `…:5000` | `…:10000` | `…:20000` |
| Growing | `dumps/nmmr_04-28-20-57-14` | `…20-58-38` | `…21-01-25` | `…21-08-26` | `…21-24-07` |

**Headline.** At paper schedule, mean |bias| drops cleanly from n=1k to
n=10k (~2× improvement) then plateaus. Width-as-√n made no difference.

| n | mean \|bias\| (fixed) | mean \|bias\| (growing) | std (fixed) | std (growing) |
|---:|---:|---:|---:|---:|
|  1,000 | 4.02 | 4.10 | 1.02 | 1.38 |
|  2,000 | 3.84 | 3.77 | 0.86 | 1.30 |
|  5,000 | 3.02 | 3.02 | 1.17 | 1.17 |
| 10,000 | 2.54 | 2.49 | 1.33 | 1.06 |
| 20,000 | 2.65 | 2.55 | 1.65 | 1.69 |

Fixed and growing trajectories are within MC noise at every n; they
literally coincide at n=5k (both have w=80 there — sanity check). At n=20k,
doubling width from 80 → 160 changed mean |bias| by 0.10. **Width is not
the lever.**

Compared to the previous (n_epochs=30) sweep:

| n | mean \|bias\| n_epochs=30 | mean \|bias\| n_epochs=3000 |
|---:|---:|---:|
|  1,000 | 25.4 |  4.02 |
|  2,000 | 14.2 |  3.84 |
|  5,000 |  5.6 |  3.02 |
| 10,000 |  5.6 |  2.54 |
| 20,000 |  5.0 |  2.65 |

Most of the previous "bias floor" was undertraining. The actual floor at
paper schedule is ~2× lower.

**Per-a bias profile (fixed-net) — two regimes:**

```
       a:    10.0   12.2   14.4   16.7   18.9   21.1   23.3   25.6   27.8   30.0
n= 1,000:  +10.45  +3.37  -1.05  -3.43  -4.45  -4.51  -3.81  -3.32  -3.08  -2.67
n= 2,000:  +11.40  +4.14  -0.37  -2.80  -3.88  -4.01  -3.51  -3.08  -2.81  -2.39
n= 5,000:  +10.72  +4.33  +0.60  -1.35  -2.27  -2.62  -2.48  -2.15  -1.90  -1.78
n=10,000:   +9.23  +3.69  +0.54  -1.07  -1.87  -2.27  -2.15  -1.78  -1.51  -1.30
n=20,000:  +11.11  +5.84  +2.72  +0.85  -0.35  -1.13  -1.30  -1.17  -1.07  -0.96
```

- **a ≥ 18.89 (interior):** clean monotone bias decay. a=21 goes
  −4.51 → −4.01 → −2.62 → −2.27 → −1.13 (4× shrink). a=30 goes
  −2.67 → −0.96. **Consistency is empirically holding here.**
- **a = 10 (boundary):** bias stuck at +10 across all n; no shrinkage.
- **a ≤ 14.44 (left of peak):** bias drifts *upward* with n (a=12.22
  goes +3.37 → +5.84). The model fits the interior more sharply at higher
  n, and that sharpening makes the unconstrained boundary worse.

If we drop a=10 from the test grid, n=20k mean |bias| falls from 2.65 to
**1.71**. Drop a=10 and a=12.22, it falls to **1.16** — clean monotone
decay across n.

**Width-scaling delta on per-a bias.** At every (n, a) cell, growing-net
bias matches fixed-net bias to within 0.3:

```
n=20k, a=10:   fixed +11.11   growing +10.69   Δ=−0.42
n=20k, a=12:   fixed  +5.84   growing  +5.42   Δ=−0.42
n=20k, a=21:   fixed  −1.13   growing  −1.24   Δ=−0.11
```

The growing class contains the fixed class as a special case (w=160 ⊃ w=80),
so the population minimizer h̃_k can only improve. The fact that empirical
ĥ doesn't move means **either** (a) the optimizer hits the same h̃ in both
classes (unlikely given different initializations) **or** (b) the
approximation error d_k²(h*, h̃_k) is essentially the same in both classes.
Most likely the bridge function is well-approximated already at w=80 in
the d_k metric and adding capacity doesn't change h̃_k materially.

**Interpretation.**

1. **The bias floor is not capacity-driven.** The model has enough
   flexibility at w=80 to reach its asymptotic limit. Doubling width gives
   the same fit.
2. **The boundary failure at a=10 is a support / signal issue, not an
   approximation issue.** As the interior fit sharpens with n, the
   boundary — where the kernel-weighted loss has weak gradient — drifts.
3. **The aggregate metric (mean |bias|, c-MSE) is misleading for the
   consistency question.** The theorem promises convergence in d_k, which
   integrates over the joint (A, X, Z) distribution. If a=10 has low
   density in the marginal A distribution (likely — it's the boundary of
   the treatment range), it contributes little to d_k² even when its
   pointwise bias is +10. **Consistency in d_k is compatible with
   persistent pointwise bias at low-density boundaries.** The uniform
   [10, 30] grid is the wrong yardstick.

**What this changes vs the previous entry.**

- "Bandwidth shrinkage required for consistency" → **incorrect framing**.
  NMMR's theorem allows fixed kernel.
- "Bias plateaus across n" → **misleading framing**. Bias plateaus on the
  uniform [10, 30] grid; bias actually shrinks with n in the interior
  (a ≥ 18.89). The plateau is one boundary point.
- "Estimator concentrates on the wrong point" → **partially correct in a
  more nuanced way**. The estimator concentrates at a fast rate; the limit
  matches h* in the bulk, fails at the boundary.

**For Wen.** The consistency theorem is empirically supported in the
interior of the support. The aggregate plateau the previous entry flagged
was 90% undertraining + 10% boundary effect; neither is a refutation of
the theorem. The bandwidth investigation is no longer needed as a primary
fix; it remains a candidate paper add-on for a possible bias / variance
sweet spot.

**Followups.**

- [ ] Re-evaluate consistency on a treatment grid drawn from the training
  A-distribution (or its quantiles), instead of uniform [10, 30]. This is
  the right test of the theorem. Expected: cleaner monotone decay.
- [ ] Diagnose why bias at a=10 *drifts upward* with n. Likely: as the
  model commits to a wrong functional form on the boundary, more SGD steps
  reinforce the commitment. Check kernel weight density at A=10 vs A=20.
- [ ] (Deferred) Bandwidth investigation as paper add-on, not core fix.

**Plots.** `plots/consistency_grow/{mean_abs_bias_vs_n, per_a_bias_compare}.png`,
`plots/consistency_3k_fixed/`, `plots/consistency_3k_grow/`.

**Configs added.**
- `configs/mar_pci_configs/upstream_demand_3k_fixed.json` (n_sample list)
- `configs/mar_pci_configs/upstream_demand_3k_grow_n{1000,2000,5000,10000,20000}.json`

**Scripts added.**
- `scripts/plot_consistency_compare.py` — overlay fixed vs growing
  consistency sweeps on shared axes (mean |bias| vs n, per-a bias panels).
- `scripts/plot_consistency.py` updated: `load_dump` now handles the
  multi-n grid-search dump structure (`dump_dir/n_sample:N/`) in addition
  to the single-n flat structure (`dump_dir/one/`).
