# RANDOM STUFF

import numpy as np
import pandas as pd
from scipy import stats


def wrap_to_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def finite_diff(y, t):
    out = np.zeros_like(y, dtype=float)
    out[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
    out[0] = (y[1] - y[0]) / (t[1] - t[0])
    out[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    return out


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def sign_flip_pvalue(differences, draws, rng):
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(differences.mean()))
    signs = rng.choice([-1.0, 1.0], size=(draws, len(differences)))
    sampled = np.abs((signs * differences).mean(axis=1))
    return float((np.count_nonzero(sampled >= observed) + 1) / (draws + 1))


def paired_tests(rows, family_columns, rng, draws):
    tests = pd.DataFrame(rows)
    if tests.empty:
        return tests
    tests["wilcoxon_p_holm"] = np.nan
    tests["permutation_p_holm"] = np.nan
    for _, index in tests.groupby(family_columns, sort=False).groups.items():
        tests.loc[index, "wilcoxon_p_holm"] = holm_adjust(tests.loc[index, "wilcoxon_p"].to_numpy(dtype=float))
        tests.loc[index, "permutation_p_holm"] = holm_adjust(tests.loc[index, "permutation_p"].to_numpy(dtype=float))
    return tests
