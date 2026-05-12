# COMPUTE VELOCITY MODEL ERRORS

import numpy as np
import pandas as pd

from lib_paths import AA_SAMPLES
from lib_velocity_model import CHASE_GATE_SPEED_MPS, SUBSTEPS, acceleration, parameter_values


def load_runs():
    samples = pd.read_csv(AA_SAMPLES).sort_values(["run", "t"]).reset_index(drop=True)
    runs = []
    for run, run_df in samples.groupby("run", sort=True):
        speed = run_df["speed_L"].to_numpy(dtype=float)
        hits = np.flatnonzero(speed >= CHASE_GATE_SPEED_MPS)
        start = int(hits[0]) if hits.size else 0
        run_df = run_df.iloc[start:].reset_index(drop=True)
        t = run_df["t"].to_numpy(dtype=float)
        runs.append(
            {
                "run": run,
                "dt": np.diff(t),
                "speed_L": run_df["speed_L"].to_numpy(dtype=float),
                "psi_dot": run_df["psi_dot"].to_numpy(dtype=float),
                "R": run_df["R"].to_numpy(dtype=float),
                "n_samples": int(len(run_df)),
            }
        )
    return runs


def interp(run, key, frame, fraction):
    return run[key][frame] + fraction * (run[key][frame + 1] - run[key][frame])


def run_metrics(run, spec, params):
    params = np.asarray(params, dtype=float)
    values = parameter_values(spec, params)
    v = np.full(len(params), run["speed_L"][0], dtype=float)
    errors = np.empty((run["n_samples"], len(params)), dtype=float)
    for frame in range(run["n_samples"]):
        errors[frame] = np.abs(run["speed_L"][frame] - v)
        if frame == run["n_samples"] - 1:
            continue
        dt = run["dt"][frame] / SUBSTEPS
        for substep in range(SUBSTEPS):
            fraction = substep / SUBSTEPS
            dv = acceleration(spec, values, v, interp(run, "R", frame, fraction), interp(run, "psi_dot", frame, fraction))
            v = np.maximum(0.0, v + dt * dv)
    return errors.mean(axis=0), np.median(errors, axis=0)
