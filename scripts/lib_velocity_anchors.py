# SET THE INITIAL ANCHOR VALUES FOR THE VELOCITY PARAMETER SEARCH

import numpy as np

from lib_grid import clip_params


def anchor_params(spec):
    if spec["model_key"] == "m0_constant":
        rows = [(0.35, 8.0), (0.50, 9.5), (0.75, 10.0), (1.00, 9.0), (1.25, 12.0)]
    elif spec["model_key"] == "m1_lateral":
        rows = [(0.35, 8.0, -0.2), (0.50, 10.0, -0.5), (0.75, 9.5, -0.8), (1.00, 9.0, -1.0), (1.50, 12.0, -0.8)]
    elif spec["model_key"] == "m2_linear":
        rows = [(0.50, 10.0, -0.5, 0.00), (0.75, 9.5, -0.8, 0.00), (0.75, 8.5, -0.8, 0.05), (0.75, 8.0, -0.8, 0.10), (1.00, 9.0, -1.0, -0.05)]
    elif spec["model_key"] == "m3_gaussian":
        rows = [(0.50, 10.0, -0.5, 0.0, 5.0), (0.75, 9.5, -0.8, 2.0, 5.0), (0.75, 9.5, -0.8, -2.0, 5.0), (0.75, 10.0, -0.8, 4.0, 10.0), (1.00, 9.0, -1.0, -4.0, 10.0)]
    else:
        rows = [(0.50, 10.0, -0.5, 0.0, 5.0, 1.0), (0.75, 9.5, -0.8, 2.0, 5.0, 1.0), (0.75, 9.5, -0.8, -2.0, 5.0, 1.0), (0.75, 10.0, -0.8, 4.0, 10.0, 1.5), (1.00, 9.0, -1.0, -4.0, 10.0, 1.5)]
    midpoint = tuple((lo + hi) / 2.0 for lo, hi in spec["bounds"])
    return clip_params(spec, np.array([*rows, midpoint], dtype=float))


def nested_anchors(spec, previous_best):
    rows = []
    for row in previous_best:
        bpsi = 0.0 if np.isnan(row["bpsi"]) else row["bpsi"]
        if spec["model_key"] == "m1_lateral":
            rows.append((row["k"], row["vstar"], 0.0))
        elif spec["model_key"] == "m2_linear":
            rows.append((row["k"], row["vstar"], bpsi, 0.0))
        elif spec["model_key"] == "m3_gaussian":
            rows.append((row["k"], row["vstar"], bpsi, 0.0, 20.0))
        elif spec["model_key"] == "m4_switch":
            rows.append((row["k"], row["vstar"], bpsi, 0.0, 5.0, 1.5))
    if not rows:
        return np.empty((0, len(spec["param_names"])), dtype=float)
    return clip_params(spec, np.asarray(rows, dtype=float))
