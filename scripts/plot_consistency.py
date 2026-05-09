"""Plot per-a bias and rep-variance vs sample size n.

Usage:
    uv run python scripts/plot_consistency.py n=1000:dump1 n=2000:dump2 ... \\
        --out plots/consistency

Each ``n=N:dump_dir`` argument names a sample size and the dump it produced.
The script loads ``one/*.pred.txt`` and ``one/result.csv`` per dump, derives the
test grid from the dump's ``configs.json``, and produces:

  - ``bias_vs_n.png``: one curve per a-value, mean bias across reps as f(n)
  - ``stderr_vs_n.png``: rep-to-rep std of beta_hat(a) as f(n)
  - ``per_a_bias_grid.png``: per-a bias curves overlaid by n
  - prints a summary table to stdout
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt


def load_dump(dump_dir: Path):
    # configs.json may be at this level (single-config dump) or one parent up
    # (grid-search dump where dump_dir is e.g. .../nmmr_xxx/n_sample:1000).
    cfg_path = dump_dir / "configs.json"
    if not cfg_path.exists():
        cfg_path = dump_dir.parent / "configs.json"
    cfg = json.loads(cfg_path.read_text())

    # Predictions live in dump_dir/one (when neither data nor model has a list)
    # or directly in dump_dir (when data has a list and model is "one").
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("specs", nargs="+",
                    help="Each: ``n=N:dump_path`` (e.g. ``n=5000:dumps/nmmr_...``)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    runs = []
    for spec in args.specs:
        n_part, path = spec.split(":", 1)
        n = int(n_part.split("=", 1)[1])
        cfg, preds, losses = load_dump(Path(path))
        runs.append((n, cfg, preds, losses))
    runs.sort(key=lambda r: r[0])

    # All runs should share the same test grid; use the first.
    treats, struct = get_test_grid(runs[0][1])
    n_a = len(treats)

    # bias[i, k] = mean bias at a=treats[k] when n = runs[i][0]
    bias = np.zeros((len(runs), n_a))
    stderr = np.zeros((len(runs), n_a))     # SD across reps (variance proxy)
    mse = np.zeros(len(runs))
    n_arr = np.array([r[0] for r in runs])
    for i, (n, cfg, preds, losses) in enumerate(runs):
        bias[i] = preds.mean(axis=0) - struct
        stderr[i] = preds.std(axis=0)
        mse[i] = losses.mean()

    # Print summary
    print(f'{"n":>7s}  {"reps":>5s}  {"MSE_mean":>10s}  {"|bias|.mean":>12s}  {"bias_max":>9s}  {"std_avg":>9s}')
    for i, (n, _, preds, _) in enumerate(runs):
        print(f'{n:7d}  {preds.shape[0]:5d}  {mse[i]:10.2f}  {np.abs(bias[i]).mean():12.3f}  {np.max(np.abs(bias[i])):9.3f}  {stderr[i].mean():9.3f}')
    print()
    print('Per-a mean bias (rows = n, cols = a):')
    print('  ' + ' '.join(f'{a:7.2f}' for a in treats))
    for i, (n, _, _, _) in enumerate(runs):
        print(f'n={n:5d}  ' + ' '.join(f'{b:+7.2f}' for b in bias[i]))

    # Plot 1: bias vs n, one curve per a
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis(np.linspace(0, 1, n_a))
    for k in range(n_a):
        ax.plot(n_arr, bias[:, k], marker="o", color=cmap[k], label=f"a={treats[k]:.1f}")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("sample size n  (log scale)")
    ax.set_ylabel(r"mean bias  $\overline{\hat\beta(a) - \beta(a)}$")
    ax.set_title("Bias vs sample size — one curve per treatment value")
    ax.legend(title="a", fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(args.out / "bias_vs_n.png", dpi=160)
    plt.close(fig)

    # Plot 2: rep-std (variance proxy) vs n, one curve per a
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in range(n_a):
        ax.plot(n_arr, stderr[:, k], marker="o", color=cmap[k], label=f"a={treats[k]:.1f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample size n  (log scale)")
    ax.set_ylabel(r"rep std of $\hat\beta(a)$  (log scale)")
    ax.set_title("Variance vs sample size — one curve per treatment value")
    ax.legend(title="a", fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(args.out / "stderr_vs_n.png", dpi=160)
    plt.close(fig)

    # Plot 3: per-a bias overlay, color by n
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.plasma(np.linspace(0, 0.9, len(runs)))
    for i, (n, _, _, _) in enumerate(runs):
        ax.plot(treats, bias[i], marker="o", linewidth=2, color=cmap[i], label=f"n={n}")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("treatment a")
    ax.set_ylabel(r"mean bias  $\overline{\hat\beta(a) - \beta(a)}$")
    ax.set_title("Per-a bias curves at each sample size")
    ax.legend(title="n", loc="best")
    fig.tight_layout()
    fig.savefig(args.out / "per_a_bias_grid.png", dpi=160)
    plt.close(fig)

    plt.style.use("default")
    print(f"\nWrote plots to: {args.out}/")


if __name__ == "__main__":
    main()
