"""Cross-fitted Nadaraya-Watson smoother for the MAR-NMMR residual imputation.

Paper §4.4 calls for the leave-fold-out smoother

    m̂^{(-k)}_θ(L⁺_i)
        = Σ_{j ∈ I_{-k}, δ_j=1} K_h(L⁺_i, L⁺_j) · r_j(θ)
          / Σ_{j ∈ I_{-k}, δ_j=1} K_h(L⁺_i, L⁺_j),     for i ∈ I_k.

The weights depend only on L⁺ (which is fixed across SGD), not on θ. So we
precompute a single (n, n) weight matrix W once at the start of training, and
the per-step imputation reduces to a matrix-vector product W @ residual.

L⁺ is component-wise standardised before forming pairwise distances, so the
kernel is not dominated by Y (which lives on a much larger scale than A and Z
in the demand DGP). The bandwidth defaults to the median pairwise-distance
heuristic on observed rows, computed in the same standardised space.

Caveat: dense (n, n) — fine for n up to a few thousand. Chunking is a Pass 4
performance concern, not a Pass 2 correctness one.
"""

from typing import List, Optional, Tuple

import torch


def _standardise(features: torch.Tensor) -> torch.Tensor:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True) + 1e-8
    return (features - mean) / std


def _median_pairwise_distance(features: torch.Tensor) -> float:
    n = features.shape[0]
    if n < 2:
        return 1.0
    dists = torch.cdist(features, features, p=2)
    iu = torch.triu_indices(n, n, offset=1, device=features.device)
    median = torch.median(dists[iu[0], iu[1]])
    return float(median.item()) if float(median.item()) > 0 else 1.0


def precompute_nw_weights(
    l_plus: torch.Tensor,
    fold_indices: List[torch.Tensor],
    delta_w: torch.Tensor,
    bandwidth: Optional[float] = None,
) -> Tuple[torch.Tensor, float]:
    """Build the cross-fit NW weight matrix W (n × n).

    For row i, W[i, j] is non-zero only if j is observed (δ_j = 1) AND j is in
    a different fold than i. Each row is normalised to sum to 1 over its valid
    neighbours; rows with no valid neighbours sum to 0 (their imputation
    contributes 0 — these correspond to a degenerate fold structure and should
    not occur in practice).

    Returns (W, bandwidth_used).
    """
    n = l_plus.shape[0]
    device = l_plus.device

    # Component-wise standardise L⁺ before forming distances.
    l_plus_s = _standardise(l_plus)

    # Pairwise squared distances on standardised L⁺.
    dists = torch.cdist(l_plus_s, l_plus_s, p=2)
    sq_dists = dists ** 2

    # Bandwidth: median heuristic on observed-row pairs, in the standardised space.
    delta_flat = delta_w.view(-1)
    if bandwidth is None:
        obs_mask = delta_flat > 0.5
        obs_l_plus_s = l_plus_s[obs_mask]
        bandwidth = _median_pairwise_distance(obs_l_plus_s)

    # Gaussian kernel in the standardised space.
    K = torch.exp(-sq_dists / (2.0 * bandwidth ** 2))

    # Fold assignment per row.
    fold_assignment = torch.full((n,), -1, dtype=torch.long, device=device)
    for k, idx in enumerate(fold_indices):
        fold_assignment[idx.to(device)] = k

    # Validity mask: j must be observed AND in a different fold than i.
    obs_col = (delta_flat > 0.5).to(K.dtype).unsqueeze(0)              # (1, n)
    diff_fold = (fold_assignment.unsqueeze(0) != fold_assignment.unsqueeze(1)).to(K.dtype)  # (n, n)
    valid = obs_col * diff_fold

    W = K * valid

    # Row-normalise. Rows with zero mass stay zero (degenerate; warned about elsewhere).
    row_sum = W.sum(dim=1, keepdim=True)
    row_sum_safe = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
    W = W / row_sum_safe

    return W, bandwidth


def imputed_residual(
    residual: torch.Tensor,
    delta_w: torch.Tensor,
    mar_weights: torch.Tensor,
) -> torch.Tensor:
    """r̃_i = δ_i · r_i + (1 − δ_i) · Σ_j W[i, j] · r_j.

    Vectorised: r_imputed = δ ⊙ r + (1 − δ) ⊙ (W @ r).
    """
    smoothed = mar_weights @ residual
    return delta_w * residual + (1.0 - delta_w) * smoothed
