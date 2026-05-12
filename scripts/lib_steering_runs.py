# COMPUTE STEERING MODEL ERRORS

import numpy as np
import pandas as pd

from lib_paths import AA_SAMPLES
from lib_steering_model import SUBSTEPS, command, param_values
from lib_math import wrap_to_pi


def load_runs():
    samples = pd.read_csv(AA_SAMPLES).sort_values(["run", "t"]).reset_index(drop=True)
    runs = []
    for run, run_df in samples.groupby("run", sort=True):
        t = run_df["t"].to_numpy(dtype=float)
        runs.append(
            {
                "run": run,
                "dt": np.diff(t),
                "x_L": run_df["x_L"].to_numpy(dtype=float),
                "y_L": run_df["y_L"].to_numpy(dtype=float),
                "x_P": run_df["x_P"].to_numpy(dtype=float),
                "y_P": run_df["y_P"].to_numpy(dtype=float),
                "vx_P": run_df["vx_P"].to_numpy(dtype=float),
                "vy_P": run_df["vy_P"].to_numpy(dtype=float),
                "speed_L": run_df["speed_L"].to_numpy(dtype=float),
                "psi_L": np.unwrap(run_df["psi_L"].to_numpy(dtype=float)),
                "n_samples": int(len(run_df)),
            }
        )
    return runs


def run_errors(run, spec, params):
    params = np.asarray(params, dtype=float)
    t_values, k_values, n_values = param_values(spec, params)
    x = np.full(len(params), run["x_L"][0])
    y = np.full(len(params), run["y_L"][0])
    psi = np.full(len(params), run["psi_L"][0])
    errors = np.empty((run["n_samples"], len(params)), dtype=float)
    for frame in range(run["n_samples"]):
        errors[frame] = np.hypot(run["x_L"][frame] - x, run["y_L"][frame] - y)
        if frame == run["n_samples"] - 1:
            continue
        dt = run["dt"][frame] / SUBSTEPS
        for substep in range(SUBSTEPS):
            speed_l, psi_dot = command(run, frame, substep / SUBSTEPS, x, y, psi, spec, t_values, k_values, n_values)
            x = x + dt * speed_l * np.cos(psi)
            y = y + dt * speed_l * np.sin(psi)
            psi = wrap_to_pi(psi + dt * psi_dot)
    return errors.mean(axis=0), np.median(errors, axis=0)
