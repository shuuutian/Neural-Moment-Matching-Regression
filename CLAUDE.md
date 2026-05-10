# Global preferences

1. Do not do extra things not required.
2. Use Chinese as primary communication (even if user uses English with you), but keep keywords or professional expressions as they were.
3. When you try to explain current situation, or demonstrate your judgement, or propose any next step or plan to me to get my decision, try to use simple language and first-principle logic if you can.


# MAR-NMMR — implementation notes

This repo is the upstream NMMR codebase (Kremer et al.) extended for the MAR-PCI series. The MAR extension follows §4.4 of Wen Zhou's working paper (`/Users/apple/2602_WenPaper/PCI_paper_draft_overleaf`). Sister implementations live at `/Users/apple/DeepFeatureProxyVariable` (DFPV) and `/Users/apple/DeepGMM` (DeepGMM).

The plan we are executing is on Notion — *0426 — Implementation and Validation Plan*. Read that page before changing anything material.

## Entry points

- `python main.py <config> ate` → `src/experiment.py:experiments` → dispatches per `model.name` (`nmmr`, `dfpv`, `pmmr`, etc.) and per `data.name` (`demand`, `dsprite`, `rhc`).
- For the MAR extension, `data.name == "demand_mar"` routes through the new MAR-aware DGP (`src/data/ate/demand_pv_mar.py`) and `model.name == "nmmr_mar"` (introduced in Pass 1+) routes through the MAR-aware NMMR trainer.
- Outputs go to `dumps/<model_name>_<MM-DD-HH-MM-SS>/<dump_name>/<mdl_dump_name>/` per [main.py](main.py) and [src/experiment.py](src/experiment.py).

## Module → paper-section mapping

| NMMR module | Paper section / equation |
|---|---|
| `src/data/ate/demand_pv.py` (`generatate_demand_core`, `cal_outcome`) | §5 demand DGP (Xu et al. 2021) |
| `src/data/ate/demand_pv_mar.py` (Pass 1) | §5 demand DGP + §3 MAR mechanism (linear logistic on standardised L⁺) |
| `src/data/ate/data_class_mar.py` (Pass 1) | §3.1 observed-data tuple `O^obs = (Y, A, X, Z, δ_W, δ_W·W)` |
| `src/models/NMMR/NMMR_loss.py:NMMR_loss` (U-statistic branch) | §4.4 full-data empirical risk R̂_full(θ) = (1/n(n−1)) Σ_{i≠j} r_i(θ) r_j(θ) k(L_i,L_j) |
| `src/models/NMMR/NMMR_loss.py:NMMR_loss_mar` (Pass 2) | §4.4 imputed risk R̂_MAR(θ) with r̃_i(θ) = δ_W r_i(θ) + (1−δ_W) m̂_θ(L⁺) |
| `src/models/NMMR/mar_imputer.py:precompute_nw_weights` (Pass 2) | §4.4 leave-fold-out NW smoother on standardised L⁺; row-normalised, fold-aware, observed-only — closed-form per θ |
| `src/models/NMMR/kernel_utils.py` (RBF on (A,Z[,X])) | §4.4 RKHS kernel; kernel inputs L = (A,Z,X) ⊂ L⁺ are MAR-immune |
| `src/models/NMMR/NMMR_trainers.py:NMMR_Trainer_DemandExperiment.train` | SGD over the U-statistic loss |
| `NMMR_Trainer_DemandExperiment.predict` | §4.4 ATE plug-in (full-data variant): β̂(a) = mean over W draws of h_θ̂(W,a) |
| `NMMR_Trainer_DemandMARExperiment._predict_mar` (Pass 3) | §4.4 β̂(a) = (1/n) Σ_k Σ_{i∈I_k} { δ_W h_θ̂(W_i,a,X_i) + (1−δ_W) q̂_{a,θ̂}(L⁺_i) }; reuses the Pass 2 cross-fit weight matrix to evaluate q̂_{a,θ̂}(L⁺_i) = Σ_j W[i,j] · h_θ̂(W_j,a) |

## Four roles in the MAR-PCI validation set

(see Notion plan, §B.2)

| Role | Method | Data | Where it lives |
|---|---|---|---|
| **baseline** | Original NMMR, naive | Observed-only subsample (drop δ_W=0) | `data.mode = "mar_naive"` + `use_mar_modified = false` |
| **oracle_baseline** | Original NMMR | Full data, no missing | `data.mode = "oracle"` + `use_mar_modified = false` |
| **oracle_modified** | MAR-NMMR | Full data, no missing | `data.mode = "oracle"` + `use_mar_modified = true` |
| **modified** | MAR-NMMR | Partial MAR data | `data.mode = "mar_modified"` + `use_mar_modified = true` |
| **nmmr_u_repro** (control) | Original NMMR | Upstream demand-noise grid | unchanged upstream config `configs/demand_noise_configs/nmmr_u_demandnoise.json` |

In **Pass 2**, `use_mar_modified=true` switches the trainer to a full-batch SGD path that uses `NMMR_loss_mar` with a precomputed cross-fit NW weight matrix on L⁺. The imputed residual is `r̃_i = δ_i · r_i + (1 − δ_i) · Σ_j W[i,j] · r_j`; the kernel matrix on L = (A, Z) is unchanged from upstream. Reduces exactly to the upstream loss when δ ≡ 1, so `oracle_modified` should converge to the same ATE as `oracle_baseline` in expectation.

In **Pass 3**, `predict()` branches on `use_mar_modified`. The MAR-aware path (`_predict_mar`) replaces the upstream Monte-Carlo plug-in with the §4.4 estimator over training rows: for each test treatment `a`, observed rows contribute `h_θ̂(W_i, a)` directly, while missing rows contribute `q̂_{a,θ̂}(L⁺_i) = Σ_j W[i,j] · h_θ̂(W_j, a)` using the same cross-fit weight matrix from Pass 2. The non-MAR path (`baseline`, `oracle_baseline`) keeps the upstream MC plug-in over val W. Under δ ≡ 1, `_predict_mar` reduces exactly (byte-identical) to the upstream MC plug-in evaluated on training data — verified algebraically.

In **Pass 4**, full-scale configs land at `configs/mar_pci_configs/{baseline,oracle_baseline,oracle_modified,modified}.json` (n=5000, n_repeat=100, missing_rate=0.3). The placeholder schedule (n_epochs=30, batch_size=1000) matches upstream `nmmr_u_demandnoise.json` and runs 5 minibatches/epoch × 30 = 150 SGD steps. The MAR schedule (n_epochs=150, batch_size=5000) runs full-batch (1 step/epoch) × 150 = 150 SGD steps — picked so the two paths get the same total step count and oracle_modified vs oracle_baseline is a clean apples-to-apples comparison. The full-batch design is intrinsic: the cross-fit NW weight matrix is precomputed once on the entire L⁺ and minibatching would require a per-batch fold strategy. Runtime is ~3-6 s/rep on CPU, so a full sweep across the four roles is ~30 min wall.

## Sibling references

- `/Users/apple/DeepFeatureProxyVariable/src/data/ate/data_class_mar.py` — canonical `PVTrainDataSetMAR`.
- `/Users/apple/DeepGMM/scenarios/demand_scenario.py` — canonical demand-DGP MAR mechanism (`_mar_delta`, bisection on logistic intercept).
- *0425 — Systematic Validation Plan* (Notion) — the validation protocol carried over with NMMR-specific tweaks.
