# STUFF FOR THE BUTTERWORTH FILTER & CUTOFF FREQ SELECTION
import numpy as np
from scipy.signal import butter, filtfilt


CUTOFF_GRID_HZ = np.arange(1.0, 60.0, 2.0)


def whiteness_score(residual, max_lag=10):
    residual = residual - residual.mean()
    denom = float(np.dot(residual, residual))
    if denom == 0.0:
        return 0.0
    lags = range(1, min(max_lag, len(residual) - 1) + 1)
    acf = [np.dot(residual[:-lag], residual[lag:]) / denom for lag in lags]
    return float(np.sum(np.square(acf)))


def choose_cutoff(y, sample_rate_hz):
    scores = []
    usable = CUTOFF_GRID_HZ[CUTOFF_GRID_HZ < 0.45 * sample_rate_hz]
    for cutoff_hz in usable:
        b, a = butter(2, cutoff_hz, btype="low", fs=sample_rate_hz)
        y_smooth = filtfilt(b, a, y)
        scores.append((cutoff_hz, whiteness_score(y - y_smooth)))
    return min(scores, key=lambda pair: (pair[1], pair[0]))


def smooth_coordinate(y, cutoff_hz, sample_rate_hz):
    b, a = butter(2, cutoff_hz, btype="low", fs=sample_rate_hz)
    return filtfilt(b, a, y)
