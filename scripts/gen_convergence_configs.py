"""Generate the 4-role × 3-multiple convergence sweep config grid.

Each role's "1×" config matches the existing Pass 4 schedule (150 total SGD
steps). 2× and 5× double / quintuple n_epochs while keeping batch_size
fixed, so the only thing that changes between them is the optimizer's
training budget.

All configs set ``log_history: "True"`` so the trainer persists per-epoch
loss to ``<dump>/<seed>.history.csv``. Test grid is the default OOS
[10, 30] × 10 (matching the Part B opener entry in RESEARCH_DIARY.md).

Outputs:
    configs/mar_pci_configs/convergence_sweep/<role>_<mult>x.json   (×12)
"""
from __future__ import annotations
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "configs/mar_pci_configs/convergence_sweep"

# Per-role base config: data.mode, model.use_mar_modified, batch_size, base n_epochs.
# Base n_epochs corresponds to the existing Pass 4 schedule (= 1× multiple).
ROLES = {
    "baseline":        dict(data_mode="mar_naive",   use_mar=False, batch=1000, base_epochs=30),
    "oracle_baseline": dict(data_mode="oracle",      use_mar=False, batch=1000, base_epochs=30),
    "oracle_modified": dict(data_mode="oracle",      use_mar=True,  batch=5000, base_epochs=150),
    "modified":        dict(data_mode="mar_modified", use_mar=True, batch=5000, base_epochs=150),
}

MULTIPLES = [1, 2, 5]


def build_config(role: str, mult: int) -> dict:
    spec = ROLES[role]
    cfg = {
        "n_repeat": 100,
        "data": {
            "name": "demand_mar",
            "n_sample": 5000,
            "mode": spec["data_mode"],
            "missing_rate": 0.3,
            "Z_noise": 1,
            "W_noise": 1,
        },
        "model": {
            "name": "nmmr",
            "n_epochs": spec["base_epochs"] * mult,
            "batch_size": spec["batch"],
            "log_metrics": "False",
            "log_history": "True",
            "learning_rate": 0.003,
            "l2_penalty": 3e-06,
            "loss_name": "U_statistic",
            "network_depth": 4,
            "network_width": 80,
            "use_mar_modified": spec["use_mar"],
        },
    }
    if spec["use_mar"]:
        cfg["model"]["n_folds"] = 5
    return cfg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for role in ROLES:
        for mult in MULTIPLES:
            cfg = build_config(role, mult)
            path = OUT_DIR / f"{role}_{mult}x.json"
            path.write_text(json.dumps(cfg, indent=4) + "\n")
            print(f"wrote {path.relative_to(OUT_DIR.parents[2])}  "
                  f"(n_epochs={cfg['model']['n_epochs']}, "
                  f"batch={cfg['model']['batch_size']})")


if __name__ == "__main__":
    main()
