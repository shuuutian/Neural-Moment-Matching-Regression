"""Plot per-rep training-history trajectories from MAR-PCI dumps.

Reads the per-rep ``<seed>.history.csv`` files written by the trainer when
``log_history: "True"`` is set in the model config. For each dump produces a
2-panel figure (causal-loss train+val on the left, observed-MSE train+val on
the right). When multiple dumps are passed they are overlaid on the same
axes so step-count sweeps can be read directly.

Each role's trajectories are drawn as faint per-rep lines plus a thick mean
line and an IQR band. Roles use the same color palette as
``plot_validation.py`` so the visual identity is consistent across plots.

Usage
-----
    uv run python scripts/plot_training_history.py \\
        dumps/run_oracle_baseline dumps/run_baseline \\
        dumps/run_oracle_modified dumps/run_modified \\
        --out plots/convergence/oos_1x.png \\
        --title-suffix "  Part B opener  test∈[10, 30]"

    # Sweep view: 4 roles × 3 step multiples in one grid:
    uv run python scripts/plot_training_history.py \\
        --grid \\
        dumps/<role>__1x dumps/<role>__2x dumps/<role>__5x \\
        --out plots/convergence/<role>_sweep.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `from src.…` imports when invoked as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.plot_validation import (
    ROLE_COLORS, ROLE_ORDER, EXTRA_COLORS,
    role_from_config, parse_dump_arg,
)


def load_history(dump_dir: Path, label_override: str | None = None) -> dict:
    """Return dict with per-rep histories stacked into (n_rep, n_epoch) arrays.

    Skipped if the dump has no ``*.history.csv`` files (i.e. it predates the
    log_history feature).
    """
    cfg_path = dump_dir / "configs.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found")
    cfg = json.loads(cfg_path.read_text())

    one = dump_dir / "one"
    files = sorted(one.glob("*.history.csv"),
                   key=lambda p: int(p.stem.split('.')[0]))
    if not files:
        raise FileNotFoundError(
            f"No *.history.csv files in {one}. "
            "Re-run with model.log_history='True' in the config."
        )

    # Stack per-rep dataframes; assumes all reps trained for same n_epochs.
    arrays = {col: [] for col in
              ('causal_loss_train', 'causal_loss_val',
               'obs_mse_train', 'obs_mse_val')}
    for f in files:
        df = pd.read_csv(f)
        for col in arrays:
            arrays[col].append(df[col].to_numpy())
    n_epochs = min(a.size for a in arrays['causal_loss_train'])
    stacked = {col: np.stack([a[:n_epochs] for a in arrays[col]], axis=0)
               for col in arrays}
    role = label_override if label_override else role_from_config(cfg)
    return dict(role=role, cfg=cfg, n_epochs=n_epochs, **stacked)


def _color_for(role: str, extras_index: dict[str, int]) -> str:
    if role in ROLE_COLORS:
        return ROLE_COLORS[role]
    return EXTRA_COLORS[extras_index[role] % len(EXTRA_COLORS)]


def _ordered_roles(by_role: dict) -> list[str]:
    canonical = [r for r in ROLE_ORDER if r in by_role]
    extras = [r for r in by_role.keys() if r not in canonical]
    return canonical + extras


def _draw_panel(ax, by_role, key_train, key_val, ylabel, log_y=False,
                show_per_rep=True, n_per_rep_max=10):
    """Draw mean ± IQR + per-rep faint lines onto a single axis."""
    roles = _ordered_roles(by_role)
    extras_index = {r: i for i, r in enumerate(r for r in roles if r not in ROLE_COLORS)}
    for role in roles:
        info = by_role[role]
        c = _color_for(role, extras_index)
        train = info[key_train]  # (n_rep, n_epoch)
        val = info[key_val]
        epochs = np.arange(info['n_epochs'])

        if show_per_rep:
            sample = np.linspace(0, train.shape[0] - 1,
                                 num=min(n_per_rep_max, train.shape[0]),
                                 dtype=int)
            for r in sample:
                ax.plot(epochs, train[r], color=c, alpha=0.10, linewidth=0.6)

        train_med = np.median(train, axis=0)
        train_q25 = np.percentile(train, 25, axis=0)
        train_q75 = np.percentile(train, 75, axis=0)
        val_med = np.median(val, axis=0)
        ax.plot(epochs, train_med, color=c, linewidth=2.2,
                label=f"{role} (train)")
        ax.fill_between(epochs, train_q25, train_q75, color=c, alpha=0.20)
        ax.plot(epochs, val_med, color=c, linewidth=1.6, linestyle="--",
                label=f"{role} (val)")

    if log_y:
        ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="best")


def plot_history(by_role: dict, out_path: Path, title_suffix: str,
                 log_y_causal: bool = False):
    """Two-panel: causal U-statistic loss + observed MSE."""
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    _draw_panel(axes[0], by_role, 'causal_loss_train', 'causal_loss_val',
                ylabel=r"causal loss  $\hat U$", log_y=log_y_causal)
    _draw_panel(axes[1], by_role, 'obs_mse_train', 'obs_mse_val',
                ylabel=r"observed MSE  $\|h(W,A)-Y\|^2$", log_y=False)
    axes[0].set_title("U-statistic loss (median ± IQR)")
    axes[1].set_title("Observed MSE (median ± IQR)")
    fig.suptitle(f"Training history{title_suffix}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    plt.style.use("default")


def print_tail_summary(by_role: dict, last_n: int = 30):
    """Print last-N-epoch slope and tail mean to assess convergence numerically.

    Convergence rule of thumb: if |tail slope| < 1% of tail mean per epoch
    over the last 30 epochs, the loss has plateaued. Prints both so the user
    can read off the gap.
    """
    print(f"\n=== Convergence tail summary (last {last_n} epochs) ===")
    name_w = max(18, max(len(r) for r in by_role) + 2)
    header = (f"{'role':{name_w}s}  {'tail_mean':>11s}  {'tail_slope':>11s}  "
              f"{'|slope|/mean':>13s}  verdict")
    print(header)
    print("-" * len(header))
    for role in _ordered_roles(by_role):
        info = by_role[role]
        tail = info['causal_loss_train'][:, -last_n:]      # (n_rep, last_n)
        per_rep_mean = tail.mean(axis=1).mean()
        # Slope per epoch via simple linear fit on the cross-rep mean curve.
        mean_curve = tail.mean(axis=0)
        if mean_curve.size >= 2:
            xs = np.arange(mean_curve.size)
            slope = float(np.polyfit(xs, mean_curve, 1)[0])
        else:
            slope = float('nan')
        ratio = abs(slope) / max(abs(per_rep_mean), 1e-12)
        verdict = "converged" if ratio < 0.01 else ("near" if ratio < 0.05 else "still moving")
        print(f"{role:{name_w}s}  {per_rep_mean:11.4g}  {slope:11.4g}  "
              f"{ratio:13.4g}  {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dumps", nargs="+", type=str,
                    help="Dump dirs; each must contain one/<seed>.history.csv")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output PNG path (parent dir created if needed)")
    ap.add_argument("--title-suffix", default="")
    ap.add_argument("--log-y", action="store_true",
                    help="symlog y-axis for the causal-loss panel")
    ap.add_argument("--tail-n", type=int, default=30,
                    help="Number of trailing epochs used for convergence summary")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    by_role = {}
    for spec in args.dumps:
        path, label = parse_dump_arg(spec)
        info = load_history(path, label_override=label)
        if info['role'] in by_role:
            raise SystemExit(
                f"Duplicate role '{info['role']}' in dumps; pass ``path::label`` "
                "on one of them to disambiguate"
            )
        by_role[info['role']] = info

    plot_history(by_role, args.out, args.title_suffix, log_y_causal=args.log_y)
    print_tail_summary(by_role, last_n=args.tail_n)
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
