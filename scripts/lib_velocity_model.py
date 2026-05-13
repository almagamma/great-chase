# DEFINE THE VELOCITY MODELS

import numpy as np


SUBSTEPS = 5
CHASE_GATE_SPEED_MPS = 3.0
OBJECTIVE = "median_run_mean_speed_mae_mps"
PROFILE_POINTS = 70
PROFILE_MARGIN_MPS = 0.05
PARAMETER_NAMES = ("k", "vstar", "bpsi", "bR_linear", "bR_gaussian", "bR_switch", "s", "t")

MODEL_SPECS = [
    {"model_key": "m0_constant", "model_label": "M0 constant target", "terms": (), "param_names": ("k", "vstar"), "bounds": ((0.1, 5.0), (1.0, 20.0))},
    {"model_key": "m1_lateral", "model_label": "M1 lateral demand", "terms": ("psi",), "param_names": ("k", "vstar", "bpsi"), "bounds": ((0.1, 5.0), (1.0, 20.0), (-2.0, 0.0))},
    {"model_key": "m2_linear", "model_label": "M2 lateral + linear R", "terms": ("psi", "linear"), "param_names": ("k", "vstar", "bpsi", "bR_linear"), "bounds": ((0.1, 5.0), (1.0, 20.0), (-2.0, 0.0), (-0.5, 0.5))},
    {"model_key": "m3_gaussian", "model_label": "M3 lateral + Gaussian R", "terms": ("psi", "gaussian"), "param_names": ("k", "vstar", "bpsi", "bR_gaussian", "s"), "bounds": ((0.1, 5.0), (1.0, 20.0), (-2.0, 0.0), (-5.0, 5.0), (0.5, 50.0))},
    {"model_key": "m4_switch", "model_label": "M4 lateral + switch R", "terms": ("psi", "switch"), "param_names": ("k", "vstar", "bpsi", "bR_switch", "s", "t"), "bounds": ((0.1, 5.0), (1.0, 20.0), (-2.0, 0.0), (-5.0, 5.0), (0.5, 10.0), (0.1, 5.0))},
]


def parameter_values(spec, params):
    values = {name: params[:, i].astype(float) for i, name in enumerate(spec["param_names"])}
    full = {name: np.full(len(params), np.nan, dtype=float) for name in PARAMETER_NAMES}
    for name, value in values.items():
        full[name] = value
    return full


def acceleration(spec, values, v, R, psi_dot):
    target = values["vstar"].copy()
    if "psi" in spec["terms"]:
        target = target + values["bpsi"] * v * np.abs(psi_dot)
    if "linear" in spec["terms"]:
        target = target + values["bR_linear"] * R
    if "gaussian" in spec["terms"]:
        target = target + values["bR_gaussian"] * np.exp(-(R * R) / (values["s"] * values["s"]))
    if "switch" in spec["terms"]:
        scale = values["t"] + values["s"]
        target = target + values["bR_switch"] * np.exp(-(R * R) / (scale * scale)) * ((R / values["t"]) ** 2 - 1.0)
    return values["k"] * (target - v)
