"""Plot four-role ATE bias distribution + mean ATE curves for one validation run.

Reads dumps produced by ``main.py <config> ate``: each dump's ``configs.json``
identifies which of the four MAR-PCI roles it represents, and ``one/*.pred.txt``
+ ``one/result.csv`` give the per-rep ATE curves and OOS-MSE.

Visual style mirrors DeepGMM's ``run_pci_compare.py``: histogram + KDE per role,
color-coded, with the structural-ATE reference and a vertical bias=0 line.

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


def load_dump(dump_dir: Path) -> dict:
    cfg = json.loads((dump_dir / "configs.json").read_text())
    one = dump_dir / "one"
    losses = np.loadtxt(one / "result.csv")
    preds = np.array([np.loadtxt(f) for f in sorted(one.glob("*.pred.txt"),
                                                     key=lambda p: int(p.stem.split('.')[0]))])
    return dict(role=role_from_config(cfg), cfg=cfg, preds=preds, losses=losses)


def get_test_grid(cfg: dict):
    from src.data.ate.demand_pv_mar import generate_test_demand_pv_mar
    test = generate_test_demand_pv_mar(**cfg["data"])
    return test.treatment.flatten(), test.structural.flatten()


def plot_bias_distribution(by_role: dict, treats: np.ndarray, struct: np.ndarray,
                           out_path: Path, title_suffix: str):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 6))
    roles = [r for r in ROLE_ORDER if r in by_role]
    biases = {r: (by_role[r]["preds"] - struct).mean(axis=1) for r in roles}
    flat = np.concatenate(list(biases.values()))
    pad = 0.1 * max(1e-8, float(flat.max() - flat.min()))
    x_grid = np.linspace(float(flat.min() - pad), float(flat.max() + pad), 500)

    for role in roles:
        bias = biases[role]
        c = ROLE_COLORS[role]
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
    roles = [r for r in ROLE_ORDER if r in by_role]
    for role in roles:
        preds = by_role[role]["preds"]
        c = ROLE_COLORS[role]
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


def print_summary(by_role: dict, treats: np.ndarray, struct: np.ndarray, label: str):
    roles = [r for r in ROLE_ORDER if r in by_role]
    print(f"\n=== {label} ===")
    header = f"{'role':18s}  {'MSE_mean':>10s}  {'MSE_med':>10s}  {'MSE_std':>10s}  {'biasL2':>8s}  {'mean_bias':>10s}"
    print(header)
    print("-" * len(header))
    for role in roles:
        losses = by_role[role]["losses"]
        preds = by_role[role]["preds"]
        bias_curve = preds.mean(axis=0) - struct
        bias_l2 = float(np.sqrt((bias_curve ** 2).mean()))
        mean_bias_per_rep = (preds - struct).mean(axis=1)
        print(f"{role:18s}  {losses.mean():10.2f}  {np.median(losses):10.2f}  "
              f"{losses.std():10.2f}  {bias_l2:8.2f}  {mean_bias_per_rep.mean():10.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dumps", nargs="+", type=Path,
                    help="One dump dir per role; role auto-detected from configs.json")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output dir for plots; created if needed")
    ap.add_argument("--title-suffix", default="",
                    help="Appended to plot titles (e.g. test range)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    by_role = {}
    for d in args.dumps:
        info = load_dump(d)
        role = info["role"]
        if role in by_role:
            raise SystemExit(f"Duplicate role '{role}' in dumps")
        by_role[role] = info

    # Use the first dump's data config to recover test grid + structural ATE
    # (all dumps in a single run should share the same test grid).
    treats, struct = get_test_grid(next(iter(by_role.values()))["cfg"])

    plot_bias_distribution(by_role, treats, struct, args.out / "bias_distribution.png",
                           args.title_suffix)
    plot_ate_curves(by_role, treats, struct, args.out / "ate_curves.png",
                    args.title_suffix)
    print_summary(by_role, treats, struct, label=args.title_suffix.strip() or "summary")
    print(f"\nWrote plots to: {args.out}/")


if __name__ == "__main__":
    main()
