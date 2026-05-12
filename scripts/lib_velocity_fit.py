# RANDOM PARAMETER SEARCH FOR VELOCITY MODELS

import numpy as np
import pandas as pd

from lib_grid import clip_params, unique_params
from lib_random_search import local_random_params, random_params, separated_starts
from lib_velocity_anchors import anchor_params
from lib_velocity_model import OBJECTIVE, PARAMETER_NAMES, parameter_values
from lib_velocity_runs import run_metrics


RANDOM_SEARCH_SAMPLES = 260
REFINE_1_STARTS = 5
REFINE_2_STARTS = 3
REFINE_1_SAMPLES = 90
REFINE_2_SAMPLES = 90
REFINE_1_WIDTH = 0.20
REFINE_2_WIDTH = 0.06
START_SEPARATION = 0.12


def evaluate_params(runs, spec, params, stage):
    params = unique_params(clip_params(spec, params))
    run_mean = []
    run_median = []
    for run in runs:
        mean_error, median_error = run_metrics(run, spec, params)
        run_mean.append(mean_error)
        run_median.append(median_error)
    mean_matrix = np.vstack(run_mean).T
    median_matrix = np.vstack(run_median).T
    values = parameter_values(spec, params)
    row = {
        "model_key": spec["model_key"], "model_label": spec["model_label"], "stage": stage,
        "run_equal_mean_speed_mae_mps": mean_matrix.mean(axis=1),
        "q25_run_mean_speed_mae_mps": np.quantile(mean_matrix, 0.25, axis=1),
        "median_run_mean_speed_mae_mps": np.median(mean_matrix, axis=1),
        "q75_run_mean_speed_mae_mps": np.quantile(mean_matrix, 0.75, axis=1),
        "median_run_median_speed_ae_mps": np.median(median_matrix, axis=1),
    }
    for name in PARAMETER_NAMES:
        row[name] = values[name]
    return pd.DataFrame(row)


def fit_model(runs, spec, rng, extra_anchors):
    initial = unique_params(np.vstack([anchor_params(spec), extra_anchors, random_params(spec, rng, RANDOM_SEARCH_SAMPLES)]))
    initial_evals = evaluate_params(runs, spec, initial, "random")
    starts = separated_starts(initial_evals, spec, OBJECTIVE, REFINE_1_STARTS, START_SEPARATION)
    refine_1 = evaluate_params(runs, spec, local_random_params(spec, starts, rng, REFINE_1_WIDTH, REFINE_1_SAMPLES), "refine_1")
    so_far = pd.concat([initial_evals, refine_1], ignore_index=True)
    starts = separated_starts(so_far, spec, OBJECTIVE, REFINE_2_STARTS, START_SEPARATION)
    refine_2 = evaluate_params(runs, spec, local_random_params(spec, starts, rng, REFINE_2_WIDTH, REFINE_2_SAMPLES), "refine_2")
    return pd.concat([initial_evals, refine_1, refine_2], ignore_index=True)


def selected_summary(evaluations):
    rows = [group.sort_values(OBJECTIVE).iloc[0].to_dict() for _, group in evaluations.groupby("model_key", sort=False)]
    return pd.DataFrame(rows).sort_values(OBJECTIVE).reset_index(drop=True)
