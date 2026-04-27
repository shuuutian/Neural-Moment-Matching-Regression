"""MAR-aware demand DGP for the PCI/MAR-NMMR extension.

Wraps the upstream demand DGP (`demand_pv.py`) and applies a MAR mechanism
to the outcome proxy W. Three modes:

  - ``oracle``       : delta_w == 1 everywhere (no missingness)
  - ``mar_naive``    : drop rows with delta_w == 0 (observed-only subsample)
  - ``mar_modified`` : keep all rows; W is zeroed where delta_w == 0

The MAR mechanism is a logistic on the standardised L+ = (A, Z1, Z2, Y),
with the intercept calibrated by bisection so the marginal missing rate
hits the target. Mirrors `/Users/apple/DeepGMM/scenarios/demand_scenario.py`
so the three sibling repos share the same mechanism.
"""

from typing import Tuple

import numpy as np
from numpy.random import default_rng

from src.data.ate.data_class import PVTestDataSet
from src.data.ate.data_class_mar import PVTrainDataSetMAR
from src.data.ate.demand_pv import (
    cal_outcome,
    cal_structural,
    generatate_demand_core,
    psi,
)


VALID_MODES = ("oracle", "mar_naive", "mar_modified")
DEFAULT_MAR_ALPHA = np.array([1.6, 0.8, -0.8, 1.2], dtype=np.float64)


def _mar_delta(
    treatment: np.ndarray,
    treatment_proxy: np.ndarray,
    outcome: np.ndarray,
    missing_rate: float,
    seed: int,
    alpha: np.ndarray = DEFAULT_MAR_ALPHA,
) -> np.ndarray:
    """Generate the missingness indicator delta_w under MAR on L+ = (A, Z1, Z2, Y).

    The score is a linear combination of standardised L+ and is then squashed
    through a sigmoid. The intercept is found by bisection so the realised
    marginal observation probability matches ``1 - missing_rate``.

    Returns a (n, 1) array with values in {0.0, 1.0}.
    """
    l_plus = np.concatenate(
        [treatment[:, None], treatment_proxy, outcome[:, None]], axis=1
    )
    l_plus = (l_plus - l_plus.mean(0, keepdims=True)) / (
        l_plus.std(0, keepdims=True) + 1e-8
    )
    score = l_plus @ alpha.reshape(-1, 1)

    target_obs = 1.0 - missing_rate
    lo, hi = -15.0, 15.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        probs = 1.0 / (1.0 + np.exp(-(score + mid)))
        if probs.mean() > target_obs:
            hi = mid
        else:
            lo = mid
    intercept = 0.5 * (lo + hi)
    probs = 1.0 / (1.0 + np.exp(-(score + intercept)))

    rng = default_rng(seed=seed + 717)
    delta = (rng.uniform(size=probs.shape) < probs).astype(np.float64)
    return delta


def _generate_demand_mar_core(
    n_sample: int,
    mode: str,
    missing_rate: float,
    seed: int,
    Z_noise: float = 1.0,
    W_noise: float = 1.0,
) -> PVTrainDataSetMAR:
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {VALID_MODES}")

    rng = default_rng(seed=seed)
    demand, cost1, cost2, price, views, outcome_clean = generatate_demand_core(
        n_sample, rng, Z_noise, W_noise
    )
    outcome = (outcome_clean + rng.normal(0, 1.0, n_sample)).astype(float)

    treatment_proxy = np.c_[cost1, cost2]

    if mode == "oracle":
        delta = np.ones((n_sample, 1), dtype=np.float64)
        observed_w = views.copy()[:, np.newaxis]
    else:
        delta = _mar_delta(
            treatment=price,
            treatment_proxy=treatment_proxy,
            outcome=outcome,
            missing_rate=missing_rate,
            seed=seed,
        )
        observed_w = views.copy()[:, np.newaxis]
        observed_w[delta < 0.5] = 0.0

    data = PVTrainDataSetMAR(
        treatment=price[:, np.newaxis].astype(np.float64),
        treatment_proxy=treatment_proxy.astype(np.float64),
        outcome_proxy=observed_w.astype(np.float64),
        outcome=outcome[:, np.newaxis].astype(np.float64),
        backdoor=None,
        delta_w=delta.astype(np.float64),
    )

    if mode == "mar_naive":
        keep = data.delta_w[:, 0] > 0.5
        data = PVTrainDataSetMAR(
            treatment=data.treatment[keep],
            treatment_proxy=data.treatment_proxy[keep],
            outcome_proxy=data.outcome_proxy[keep],
            outcome=data.outcome[keep],
            backdoor=None,
            delta_w=data.delta_w[keep],
        )

    return data


def generate_train_demand_pv_mar(
    n_sample: int,
    mode: str = "mar_modified",
    missing_rate: float = 0.3,
    seed: int = 42,
    Z_noise: float = 1.0,
    W_noise: float = 1.0,
    **kwargs,
) -> PVTrainDataSetMAR:
    """Train-split MAR demand data. ``**kwargs`` swallows extra config knobs."""
    return _generate_demand_mar_core(
        n_sample=n_sample,
        mode=mode,
        missing_rate=missing_rate,
        seed=seed,
        Z_noise=Z_noise,
        W_noise=W_noise,
    )


def generate_test_demand_pv_mar(W_noise: float = 1.0, **kwargs) -> PVTestDataSet:
    """Test grid + structural ATE. Identical to the upstream test split.

    The test grid is on the treatment dimension only; missingness on W is a
    training-time concern. Test-time ATE evaluation marginalises over W.
    """
    price = np.linspace(10, 30, 10)
    treatment = np.array([cal_structural(p, W_noise) for p in price])
    return PVTestDataSet(
        structural=treatment[:, np.newaxis],
        treatment=price[:, np.newaxis],
    )
