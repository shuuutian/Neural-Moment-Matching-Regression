"""Compare two consistency sweeps (e.g. fixed-net vs growing-net) on shared axes.

Usage:
    uv run python scripts/plot_consistency_compare.py \\
        --fixed n=1000:dumpA n=2000:dumpA2 ... \\
        --growing n=1000:dumpB n=2000:dumpB2 ... \\
        --out plots/consistency_compare

Each ``n=N:dump_path`` argument is a (sample_size, dump_dir) pair.
Produces:
  - ``mean_abs_bias_vs_n.png``: overlaid mean |bias| curves
  - ``per_a_bias_compare.png``: small multiples, per-a bias curves at each n
  - prints summary tables to stdout
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt


def load_dump(dump_dir: Path):
    cfg_path = dump_dir / "configs.json"
    if not cfg_path.exists():
        cfg_path = dump_dir.parent / "configs.json"
    cfg = json.loads(cfg_path.read_text())
    leaf = dump_dir / "one" if (dump_dir / "one").is_dir() else dump_dir
    losses = np.loadtxt(leaf / "result.csv")
    preds = np.array([np.loadtxt(f) for f in sorted(leaf.glob("*.pred.txt"),
                                                     key=lambda p: int(p.stem.split('.')[0]))])
    return cfg, preds, losses


def get_test_grid(cfg):
    name = cfg["data"].get("name", "demand")
    if name == "demand_mar":
        from src.data.ate.demand_pv_mar import generate_test_demand_pv_mar as g
    else:
        from src.data.ate.demand_pv import generate_test_demand_pv as g
    test = g(**cfg["data"])
    return test.treatment.flatten(), test.structural.flatten()


def parse_specs(specs):
    runs = []
    for spec in specs:
        n_part, path = spec.split(":", 1)
        n = int(n_part.split("=", 1)[1])
        cfg, preds, _ = load_dump(Path(path))
        runs.append((n, cfg, preds))
    runs.sort(key=lambda r: r[0])
    return runs


def summarize(runs, treats, struct):
    n_arr = np.array([r[0] for r in runs])
    n_a = len(treats)
    bias = np.zeros((len(runs), n_a))
    stderr = np.zeros((len(runs), n_a))
    for i, (_, _, preds) in enumerate(runs):
        bias[i] = preds.mean(axis=0) - struct
        stderr[i] = preds.std(axis=0)
    return n_arr, bias, stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixed", nargs="+", required=True,
                    help="Specs for fixed-net runs (e.g. n=1000:dumps/...)")
    ap.add_argument("--growing", nargs="+", required=True,
                    help="Specs for growing-net runs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    fixed_runs = parse_specs(args.fixed)
    grow_runs = parse_specs(args.growing)
    treats, struct = get_test_grid(fixed_runs[0][1])

    fn, fb, fs = summarize(fixed_runs, treats, struct)
    gn, gb, gs = summarize(grow_runs, treats, struct)

    # --- text summary ---
    print(f'\n{"n":>7s}  {"|bias|.mean (fixed)":>21s}  {"|bias|.mean (growing)":>22s}  {"std.mean (fixed)":>17s}  {"std.mean (growing)":>19s}')
    aligned_n = sorted(set(fn.tolist()) | set(gn.tolist()))
    for n in aligned_n:
        fi = np.where(fn == n)[0]
        gi = np.where(gn == n)[0]
        fb_v = np.abs(fb[fi[0]]).mean() if len(fi) else float("nan")
        gb_v = np.abs(gb[gi[0]]).mean() if len(gi) else float("nan")
        fs_v = fs[fi[0]].mean() if len(fi) else float("nan")
        gs_v = gs[gi[0]].mean() if len(gi) else float("nan")
        print(f'{n:7d}  {fb_v:21.3f}  {gb_v:22.3f}  {fs_v:17.3f}  {gs_v:19.3f}')

    # --- plot 1: mean |bias| vs n ---
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fn, np.abs(fb).mean(axis=1), marker="o", linewidth=2,
            color="#1f77b4", label="fixed (d=4, w=80)")
    ax.plot(gn, np.abs(gb).mean(axis=1), marker="s", linewidth=2,
            color="#d62728", label=r"growing (w $\propto$ $\sqrt{n}$)")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("sample size n  (log scale)")
    ax.set_ylabel(r"mean $|\hat\beta(a) - \beta(a)|$ across 10 grid points")
    ax.set_title("Consistency probe: bias floor vs. growing network capacity")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.out / "mean_abs_bias_vs_n.png", dpi=160)
    plt.close(fig)

    # --- plot 2: per-a bias profile, panel per n ---
    aligned = sorted(set(fn.tolist()) & set(gn.tolist()))
    n_panels = len(aligned)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.4 * n_panels, 4.0),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]
    for j, n in enumerate(aligned):
        fi = np.where(fn == n)[0][0]
        gi = np.where(gn == n)[0][0]
        ax = axes[j]
        ax.plot(treats, fb[fi], marker="o", color="#1f77b4", label="fixed")
        ax.plot(treats, gb[gi], marker="s", color="#d62728", label="growing")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("a")
        ax.set_title(f"n = {n}")
        if j == 0:
            ax.set_ylabel(r"mean bias $\overline{\hat\beta(a) - \beta(a)}$")
        if j == n_panels - 1:
            ax.legend(loc="best")
    fig.suptitle("Per-a bias profile at each sample size")
    fig.tight_layout()
    fig.savefig(args.out / "per_a_bias_compare.png", dpi=160)
    plt.close(fig)

    plt.style.use("default")
    print(f"\nWrote plots to: {args.out}/")


if __name__ == "__main__":
    main()
