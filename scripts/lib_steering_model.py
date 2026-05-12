# SIMULATE THE STEERING MODELS WITH (SUBSTEP) EULER

import numpy as np

from lib_math import wrap_to_pi


R_EPS = 1e-9
SUBSTEPS = 5
OBJECTIVE = "median_run_median_error_m"
BASIN_MARGIN_M = 0.05
BATCH_SIZE = 512

MODEL_SPECS = [
    {"model_key": "pp", "model_label": "PP", "param_names": ("K",), "bounds": ((1.0, 60.0),), "grid_points": (101,), "r1_starts": 4, "r2_starts": 2, "r1_points": 201, "r2_points": 201, "r1_width": (3.0,), "r2_width": (0.75,)},
    {"model_key": "pn", "model_label": "PN", "param_names": ("N",), "bounds": ((-1.0, 2.0),), "grid_points": (101,), "r1_starts": 4, "r2_starts": 2, "r1_points": 201, "r2_points": 201, "r1_width": (0.75,), "r2_width": (0.2,)},
    {"model_key": "pp_pn", "model_label": "PP+PN", "param_names": ("T_s", "N"), "bounds": ((0.1, 4.0), (0.5, 3.0)), "grid_points": (71, 71), "r1_starts": 6, "r2_starts": 3, "r1_points": 41, "r2_points": 31, "r1_width": (0.25, 0.35), "r2_width": (0.08, 0.12)},
]


def param_values(spec, params):
    params = np.asarray(params, dtype=float)
    t_values = np.full(len(params), np.nan)
    k_values = np.zeros(len(params))
    n_values = np.zeros(len(params))
    if spec["model_key"] == "pp":
        k_values = params[:, 0]
    elif spec["model_key"] == "pn":
        n_values = params[:, 0]
    else:
        t_values = params[:, 0]
        n_values = params[:, 1]
        k_values = n_values / t_values
    return t_values, k_values, n_values


def command(run, frame, fraction, x, y, psi, spec, t_values, k_values, n_values):
    x_p = run["x_P"][frame] + fraction * (run["x_P"][frame + 1] - run["x_P"][frame])
    y_p = run["y_P"][frame] + fraction * (run["y_P"][frame + 1] - run["y_P"][frame])
    vx_p = run["vx_P"][frame] + fraction * (run["vx_P"][frame + 1] - run["vx_P"][frame])
    vy_p = run["vy_P"][frame] + fraction * (run["vy_P"][frame + 1] - run["vy_P"][frame])
    speed_l = run["speed_L"][frame] + fraction * (run["speed_L"][frame + 1] - run["speed_L"][frame])
    rx = x_p - x
    ry = y_p - y
    theta = wrap_to_pi(np.arctan2(ry, rx) - psi)
    lambda_dot = (rx * (vy_p - speed_l * np.sin(psi)) - ry * (vx_p - speed_l * np.cos(psi))) / np.maximum(rx * rx + ry * ry, R_EPS)
    if spec["model_key"] == "pp":
        return speed_l, k_values * theta
    if spec["model_key"] == "pn":
        return speed_l, n_values * lambda_dot
    return speed_l, n_values * (lambda_dot + theta / t_values)
