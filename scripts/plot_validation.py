"""Plot four-role ATE bias distribution + mean ATE curves for one validation run.

Reads dumps produced by ``main.py <config> ate``: each dump's ``configs.json``
identifies which of the four MAR-PCI roles it represents, and ``one/*.pred.txt``
+ ``one/result.csv`` give the per-rep ATE curves and OOS-MSE.

Visual style mirrors DeepGMM's ``run_pci_compare.py``: histogram + KDE per role,
color-coded, with the structural-ATE reference and a vertical bias=0 line.

A dump can be passed as ``<path>`` (role auto-detected from configs.json) or
``<path>::<label>`` to override the role tag — useful for A/B diagnostics where
two dumps share the same "role" but differ in some other config knob.

Usage:
    uv run python scripts/plot_validation.py \\
        dumps/run_oracle_baseline dumps/run_baseline \\
        dumps/run_oracle_modified dumps/run_modified \\
        --out plots/oos --title-suffix "  test∈[10, 30]"
"""

import argparse
import json
import sys
from pathlib import Path

# Allow `from src.…` imports when invoked as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Color map mirrors DeepGMM's plotting palette; consistent across plots so
# the same role keeps the same color across runs.
ROLE_COLORS = {
    "oracle_baseline": "#2E86AB",
    "baseline":        "#A23B72",
    "oracle_modified": "#E07A5F",
    "modified":        "#B221E2",
}
ROLE_ORDER = ["baseline", "oracle_baseline", "oracle_modified", "modified"]

# Fallback palette for custom-labelled dumps that aren't one of the four roles.
EXTRA_COLORS = ["#3A7D44", "#D88C00", "#7E4E9B", "#1B4D89", "#A23B22"]


def role_from_config(cfg: dict) -> str:
    mode = cfg["data"]["mode"]
    use_mar = bool(cfg["model"].get("use_mar_modified", False))
    if mode == "oracle" and not use_mar:
        return "oracle_baseline"
    if mode == "mar_naive" and not use_mar:
        return "baseline"
    if mode == "oracle" and use_mar:
        return "oracle_modified"
    if mode == "mar_modified" and use_mar:
        return "modified"
    raise ValueError(f"Cannot infer role from mode={mode!r}, use_mar_modified={use_mar}")


def load_dump(dump_dir: Path, label_override: str | None = None) -> dict:
    cfg = json.loads((dump_dir / "configs.json").read_text())
    one = dump_dir / "one"
    losses = np.loadtxt(one / "result.csv")
    preds = np.array([np.loadtxt(f) for f in sorted(one.glob("*.pred.txt"),
                                                     key=lambda p: int(p.stem.split('.')[0]))])
    role = label_override if label_override else role_from_config(cfg)
    return dict(role=role, cfg=cfg, preds=preds, losses=losses)


def parse_dump_arg(spec: str) -> tuple[Path, str | None]:
    """Split ``path`` or ``path::label`` into (Path, label_or_None)."""
    if "::" in spec:
        path, label = spec.split("::", 1)
        return Path(path), label
    return Path(spec), None


def get_test_grid(cfg: dict):
    from src.data.ate.demand_pv_mar import generate_test_demand_pv_mar
    test = generate_test_demand_pv_mar(**cfg["data"])
    return test.treatment.flatten(), test.structural.flatten()


def ordered_roles(by_role: dict) -> list[str]:
    """Roles in canonical order; extras appended in insertion order."""
    canonical = [r for r in ROLE_ORDER if r in by_role]
    extras = [r for r in by_role.keys() if r not in canonical]
    return canonical + extras


def color_for(role: str, extras_index: dict[str, int]) -> str:
    if role in ROLE_COLORS:
        return ROLE_COLORS[role]
    return EXTRA_COLORS[extras_index[role] % len(EXTRA_COLORS)]


def plot_bias_distribution(by_role: dict, treats: np.ndarray, struct: np.ndarray,
                           out_path: Path, title_suffix: str):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 6))
    roles = ordered_roles(by_role)
    extras_index = {r: i for i, r in enumerate(r for r in roles if r not in ROLE_COLORS)}
    biases = {r: (by_role[r]["preds"] - struct).mean(axis=1) for r in roles}
    flat = np.concatenate(list(biases.values()))
    pad = 0.1 * max(1e-8, float(flat.max() - flat.min()))
    x_grid = np.linspace(float(flat.min() - pad), float(flat.max() + pad), 500)

    for role in roles:
        bias = biases[role]
        c = color_for(role, extras_index)
        ax.hist(bias, bins=20, density=True, alpha=0.22, color=c,
                edgecolor=c, linewidth=1.6, label=role)
        if bias.size >= 2 and bias.std() > 0:
            ax.plot(x_grid, gaussian_kde(bias)(x_grid), color=c, linewidth=2.2)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"per-rep mean bias  $\overline{\hat\beta(a) - \beta(a)}$")
    ax.set_ylabel("density")
    ax.set_title(f"ATE bias distribution{title_suffix}")
    ax.legend(title="role")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    plt.style.use("default")


def plot_ate_curves(by_role: dict, treats: np.ndarray, struct: np.ndarray,
                    out_path: Path, title_suffix: str):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 6))
    roles = ordered_roles(by_role)
    extras_index = {r: i for i, r in enumerate(r for r in roles if r not in ROLE_COLORS)}
    for role in roles:
        preds = by_role[role]["preds"]
        c = color_for(role, extras_index)
        mean_curve = preds.mean(axis=0)
        q25 = np.percentile(preds, 25, axis=0)
        q75 = np.percentile(preds, 75, axis=0)
        ax.plot(treats, mean_curve, color=c, linewidth=2, marker="o", label=role)
        ax.fill_between(treats, q25, q75, alpha=0.18, color=c)
    ax.plot(treats, struct, color="black", linewidth=2.5,
            linestyle="--", marker="x", label="structural")
    ax.set_xlabel("treatment a")
    ax.set_ylabel(r"$\hat\beta(a)$")
    ax.set_title(f"Mean ATE curve (band = IQR across reps){title_suffix}")
    ax.legend(title="role")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    plt.style.use("default")


def plot_bias_per_treatment(by_role: dict, treats: np.ndarray, struct: np.ndarray,
                            out_path: Path, title_suffix: str):
    """Per-a bias curves: shows where each role's systematic error concentrates.

    Mean bias ± 1 std across reps, plotted against treatment a. Diagnoses
    e.g. "modified is fine in [22, 28] but extrapolates badly below 16".
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 6))
    roles = ordered_roles(by_role)
    extras_index = {r: i for i, r in enumerate(r for r in roles if r not in ROLE_COLORS)}
    for role in roles:
        preds = by_role[role]["preds"]
        c = color_for(role, extras_index)
        bias_per_rep = preds - struct  # (n_rep, n_grid)
        mean_bias = bias_per_rep.mean(axis=0)
        std_bias = bias_per_rep.std(axis=0)
        ax.plot(treats, mean_bias, color=c, linewidth=2, marker="o", label=role)
        ax.fill_between(treats, mean_bias - std_bias, mean_bias + std_bias,
                        alpha=0.15, color=c)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("treatment a")
    ax.set_ylabel(r"bias  $\hat\beta(a) - \beta(a)$")
    ax.set_title(f"Bias per treatment (band = ±1 std across reps){title_suffix}")
    ax.legend(title="role")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    plt.style.use("default")


def print_summary(by_role: dict, treats: np.ndarray, struct: np.ndarray, label: str):
    roles = ordered_roles(by_role)
    name_w = max(18, max(len(r) for r in roles) + 2)
    print(f"\n=== {label} ===")
    header = f"{'role':{name_w}s}  {'MSE_mean':>10s}  {'MSE_med':>10s}  {'MSE_std':>10s}  {'biasL2':>8s}  {'mean_bias':>10s}"
    print(header)
    print("-" * len(header))
    for role in roles:
        losses = by_role[role]["losses"]
        preds = by_role[role]["preds"]
        bias_curve = preds.mean(axis=0) - struct
        bias_l2 = float(np.sqrt((bias_curve ** 2).mean()))
        mean_bias_per_rep = (preds - struct).mean(axis=1)
        print(f"{role:{name_w}s}  {losses.mean():10.2f}  {np.median(losses):10.2f}  "
              f"{losses.std():10.2f}  {bias_l2:8.2f}  {mean_bias_per_rep.mean():10.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dumps", nargs="+", type=str,
                    help="Dump dirs; ``path`` (auto-detect role) or ``path::label`` (override)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output dir for plots; created if needed")
    ap.add_argument("--title-suffix", default="",
                    help="Appended to plot titles (e.g. test range)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    by_role = {}
    for spec in args.dumps:
        path, label = parse_dump_arg(spec)
        info = load_dump(path, label_override=label)
        role = info["role"]
        if role in by_role:
            raise SystemExit(
                f"Duplicate role '{role}' in dumps; pass ``path::custom_label`` "
                f"on one of them to disambiguate"
            )
        by_role[role] = info

    # Use the first dump's data config to recover test grid + structural ATE
    # (all dumps in a single run should share the same test grid).
    treats, struct = get_test_grid(next(iter(by_role.values()))["cfg"])

    plot_bias_distribution(by_role, treats, struct, args.out / "bias_distribution.png",
                           args.title_suffix)
    plot_ate_curves(by_role, treats, struct, args.out / "ate_curves.png",
                    args.title_suffix)
    plot_bias_per_treatment(by_role, treats, struct, args.out / "bias_per_treatment.png",
                            args.title_suffix)
    print_summary(by_role, treats, struct, label=args.title_suffix.strip() or "summary")
    print(f"\nWrote plots to: {args.out}/")


if __name__ == "__main__":
    main()
