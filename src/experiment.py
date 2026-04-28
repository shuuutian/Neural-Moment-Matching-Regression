from typing import Dict, Any, Optional
from pathlib import Path
import os
import numpy as np
import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils import grid_search_dict

logger = logging.getLogger()


def _run_repeats(run_func, env_param, mdl_param, one_mdl_dump_dir, n_repeat, num_cpus, verbose):
    """Run ``n_repeat`` calls of ``run_func`` over distinct seeds.

    Parallel iff ``num_cpus > 1`` AND ``n_repeat > 1``. Workers are spawned
    fresh (so each re-imports torch with the BLAS-thread cap below applied)
    and write per-seed pred files at distinct paths, so writes are
    parallel-safe for the NMMR demand path. Other model paths in this repo
    have not been audited for parallel safety — opt in via ``-t > 1`` only
    for NMMR-style models that write per-seed dump files.
    """
    if num_cpus > 1 and n_repeat > 1:
        # Cap per-worker BLAS threads so 12 cores aren't oversubscribed
        # 12-ways. ProcessPoolExecutor on macOS defaults to "spawn", so
        # each worker re-imports torch and picks up these env vars.
        torch_threads = max(1, (os.cpu_count() or 1) // num_cpus)
        os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))
        results: list = [None] * n_repeat
        with ProcessPoolExecutor(max_workers=num_cpus) as ex:
            futures = {
                ex.submit(run_func, env_param, mdl_param, one_mdl_dump_dir, idx, verbose): idx
                for idx in range(n_repeat)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        return [r for r in results if r is not None]
    return [run_func(env_param, mdl_param, one_mdl_dump_dir, idx, verbose)
            for idx in range(n_repeat)]


def get_run_func(mdl_name: str):
    # Lazy imports so running NMMR (the only model needed for MAR-PCI) does not
    # require jax / tensorflow / etc., which the legacy baselines pull in.
    if mdl_name == "kpv":
        from src.models.kernelPV.model import kpv_experiments
        return kpv_experiments
    elif mdl_name == "dfpv":
        from src.models.DFPV.trainer import dfpv_experiments
        return dfpv_experiments
    elif mdl_name == "dfpv_cnn":
        from src.models.DFPV_CNN.trainer import dfpv_cnn_experiments
        return dfpv_cnn_experiments
    elif mdl_name == "pmmr":
        from src.models.PMMR.model import pmmr_experiments
        return pmmr_experiments
    elif mdl_name == "cevae":
        from src.models.CEVAE.trainer import cevae_experiments
        return cevae_experiments
    elif mdl_name == "nmmr":
        from src.models.NMMR.NMMR_experiments import NMMR_experiment
        return NMMR_experiment
    elif mdl_name in ["linear_regression_AY", "linear_regression_AWZY", "linear_regression_AY2",
                      "linear_regression_AWZY2", "linear_regression_AWY", "linear_regression_AWY2"]:
        from src.models.linear_regression.linear_reg_experiments import linear_reg_demand_experiment
        return linear_reg_demand_experiment
    elif mdl_name == "naive_neural_net_AY" or mdl_name == "naive_neural_net_AWZY" or mdl_name == "naive_neural_net_AWY":
        from src.models.naive_neural_net.naive_nn_experiments import naive_nn_experiment
        return naive_nn_experiment
    elif mdl_name == "twoSLS":
        from src.models.twoSLS.twoSLS_experiments import twoSLS_experiment
        return twoSLS_experiment
    else:
        raise ValueError(f"name {mdl_name} is not known")


def experiments(configs: Dict[str, Any],
                dump_dir: Path,
                num_cpus: int):

    data_config = configs["data"]
    model_config = configs["model"]
    n_repeat: int = configs["n_repeat"]

    if num_cpus <= 1 and n_repeat <= 1:
        verbose: int = 2
    else:
        verbose: int = 0

    run_func = get_run_func(model_config["name"])
    for dump_name, env_param in grid_search_dict(data_config):
        one_dump_dir = dump_dir.joinpath(dump_name)
        os.mkdir(one_dump_dir)
        for mdl_dump_name, mdl_param in grid_search_dict(model_config):
            if mdl_dump_name != "one":
                one_mdl_dump_dir = one_dump_dir.joinpath(mdl_dump_name)
                os.mkdir(one_mdl_dump_dir)
            else:
                one_mdl_dump_dir = one_dump_dir

            if model_config.get("log_metrics", False) == "True":
                test_losses = []
                train_metrics_ls = []
                for idx in range(n_repeat):
                    test_loss, train_metrics = run_func(env_param, mdl_param, one_mdl_dump_dir, idx, verbose)
                    train_metrics['rep_ID'] = idx
                    train_metrics_ls.append(train_metrics)
                    if test_loss is not None:
                        test_losses.append(test_loss)

                if test_losses:
                    np.savetxt(one_mdl_dump_dir.joinpath("result.csv"), np.array(test_losses))
                metrics_df = pd.concat(train_metrics_ls).reset_index()
                metrics_df.rename(columns={'index': 'epoch_num'}, inplace=True)
                metrics_df.to_csv(one_mdl_dump_dir.joinpath("train_metrics.csv"), index=False)
            else:
                test_losses = _run_repeats(run_func, env_param, mdl_param, one_mdl_dump_dir,
                                           n_repeat, num_cpus, verbose)
                if test_losses:
                    np.savetxt(one_mdl_dump_dir.joinpath("result.csv"), np.array(test_losses))
