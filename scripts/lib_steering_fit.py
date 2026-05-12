# FIT THE STEERING MODELS

import numpy as np
import pandas as pd
from lib_grid import clip_params, grid_params, local_params, smooth_values, unique_params
from lib_steering_model import BATCH_SIZE, OBJECTIVE, param_values
from lib_steering_runs import run_errors

def evaluate_params(runs, spec, params, stage):
    params = unique_params(clip_params(spec, params))
    if len(params) > BATCH_SIZE:
        frames = [evaluate_params(runs, spec, params[start : start + BATCH_SIZE], stage) for start in range(0, len(params), BATCH_SIZE)]
        return pd.concat(frames, ignore_index=True)
    mean_errors = []
    median_errors = []
    for run in runs:
        run_mean, run_median = run_errors(run, spec, params)
        mean_errors.append(run_mean)
        median_errors.append(run_median)
    mean_matrix = np.vstack(mean_errors).T
    median_matrix = np.vstack(median_errors).T
    t_values, k_values, n_values = param_values(spec, params)
    return pd.DataFrame(
        {
            "model_key": spec["model_key"], "model_label": spec["model_label"], "stage": stage,
            "T_s": t_values, "K": k_values, "N": n_values,
            "run_equal_mean_error_m": mean_matrix.mean(axis=1),
            "median_run_mean_error_m": np.median(mean_matrix, axis=1),
            "q25_run_median_error_m": np.quantile(median_matrix, 0.25, axis=1),
            "median_run_median_error_m": np.median(median_matrix, axis=1),
            "q75_run_median_error_m": np.quantile(median_matrix, 0.75, axis=1),
        }
    )


def first_starts(grid, spec):
    if len(spec["param_names"]) == 1:
        param = spec["param_names"][0]
        line = grid.sort_values(param).copy()
        line["smooth"] = smooth_values(line[OBJECTIVE].to_numpy(dtype=float), 2)
        starts = [grid.sort_values(OBJECTIVE).iloc[0][list(spec["param_names"])].to_numpy(dtype=float)]
        starts.extend(line.sort_values("smooth").head(spec["r1_starts"])[list(spec["param_names"])].to_numpy(dtype=float))
        return unique_params(np.asarray(starts, dtype=float))
    starts = [grid.sort_values(OBJECTIVE).iloc[0][list(spec["param_names"])].to_numpy(dtype=float)]
    starts.extend(grid.sort_values(OBJECTIVE).head(spec["r1_starts"])[list(spec["param_names"])].to_numpy(dtype=float))
    return unique_params(np.asarray(starts, dtype=float))


def fit_model(runs, spec):
    grid = evaluate_params(runs, spec, grid_params(spec), "grid")
    refine_1 = evaluate_params(runs, spec, local_params(spec, first_starts(grid, spec), spec["r1_width"], spec["r1_points"]), "refine_1")
    all_evals = pd.concat([grid, refine_1], ignore_index=True)
    starts = all_evals.sort_values(OBJECTIVE).head(spec["r2_starts"])[list(spec["param_names"])].to_numpy(dtype=float)
    refine_2 = evaluate_params(runs, spec, local_params(spec, starts, spec["r2_width"], spec["r2_points"]), "refine_2")
    return pd.concat([grid, refine_1, refine_2], ignore_index=True)

def selected_summary(evaluations):
    rows = [group.sort_values(OBJECTIVE).iloc[0].to_dict() for _, group in evaluations.groupby("model_key", sort=False)]
    return pd.DataFrame(rows).sort_values(OBJECTIVE).reset_index(drop=True)
