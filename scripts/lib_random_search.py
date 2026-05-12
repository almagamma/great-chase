# RANDOM PARAMETER SEARCH

import numpy as np

from lib_grid import clip_params, unique_params


def random_params(spec, rng, n):
    return np.column_stack([rng.uniform(lo, hi, size=n) for lo, hi in spec["bounds"]])


def local_random_params(spec, starts, rng, width_fraction, samples_per_start):
    span = np.array([hi - lo for lo, hi in spec["bounds"]], dtype=float)
    all_params = [starts]
    for center in starts:
        lo = center - width_fraction * span
        hi = center + width_fraction * span
        all_params.append(np.column_stack([rng.uniform(lo[i], hi[i], size=samples_per_start) for i in range(len(center))]))
    return unique_params(clip_params(spec, np.vstack(all_params)))


def separated_starts(evaluations, spec, objective, n_starts, separation_fraction):
    span = np.array([hi - lo for lo, hi in spec["bounds"]], dtype=float)
    selected = []
    for _, row in evaluations.sort_values(objective).iterrows():
        current = np.array([float(row[name]) for name in spec["param_names"]], dtype=float)
        if all(np.sqrt(np.sum(((current - old) / (separation_fraction * span)) ** 2)) >= 1.0 for old in selected):
            selected.append(current)
        if len(selected) == n_starts:
            break
    return np.asarray(selected, dtype=float)
