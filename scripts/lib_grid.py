# STUFF FOR GRID PARAMETER SEARCH

import numpy as np


def mesh(values):
    grids = np.meshgrid(*values, indexing="ij")
    return np.column_stack([grid.ravel() for grid in grids])

def grid_params(spec):
    return mesh([np.linspace(lo, hi, n) for (lo, hi), n in zip(spec["bounds"], spec["grid_points"])])

def clip_params(spec, params):
    params = np.asarray(params, dtype=float).copy()
    for i, (lo, hi) in enumerate(spec["bounds"]):
        params[:, i] = np.clip(params[:, i], lo, hi)
    return params


def unique_params(params):
    rounded = np.round(np.asarray(params, dtype=float), 10)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return np.asarray(params, dtype=float)[np.sort(idx)]

def local_params(spec, starts, widths, n_points):
    all_params = []
    for center in np.asarray(starts, dtype=float):
        values = [np.linspace(center[i] - widths[i], center[i] + widths[i], n_points) for i in range(len(widths))]
        all_params.append(mesh(values))
    return unique_params(clip_params(spec, np.vstack(all_params)))

def smooth_values(values, radius=2):
    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        out[i] = np.median(values[max(0, i - radius) : min(len(values), i + radius + 1)])
    return out
