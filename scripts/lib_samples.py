# COMPUTE THE KINEMATIC QUANTITIES

import numpy as np
import pandas as pd

from lib_filter import choose_cutoff, smooth_coordinate
from lib_math import finite_diff, wrap_to_pi


def load_raw_run(path):
    columns = ["frame_60", "time_s", "k0_x", "k0_y", "k1_x", "k1_y"]
    df = pd.read_csv(path, usecols=columns)
    return df.sort_values("time_s").drop_duplicates("time_s").reset_index(drop=True)


def prepare_run(row, raw_dir, n_runs):
    raw = load_raw_run(raw_dir / row["raw_csv"])
    t = raw["time_s"].to_numpy(dtype=float)
    sample_rate_hz = float(1.0 / np.median(np.diff(t)))
    x_l_raw = raw["k0_x"].to_numpy(dtype=float)
    y_l_raw = raw["k0_y"].to_numpy(dtype=float)
    x_p_raw = raw["k1_x"].to_numpy(dtype=float)
    y_p_raw = raw["k1_y"].to_numpy(dtype=float)

    cutoff_x_l, score_x_l = choose_cutoff(x_l_raw, sample_rate_hz)
    cutoff_y_l, score_y_l = choose_cutoff(y_l_raw, sample_rate_hz)
    cutoff_x_p, score_x_p = choose_cutoff(x_p_raw, sample_rate_hz)
    cutoff_y_p, score_y_p = choose_cutoff(y_p_raw, sample_rate_hz)
    cutoff_hz = float(np.median([cutoff_x_l, cutoff_y_l, cutoff_x_p, cutoff_y_p]))

    x_l = smooth_coordinate(x_l_raw, cutoff_hz, sample_rate_hz)
    y_l = smooth_coordinate(y_l_raw, cutoff_hz, sample_rate_hz)
    x_p = smooth_coordinate(x_p_raw, cutoff_hz, sample_rate_hz)
    y_p = smooth_coordinate(y_p_raw, cutoff_hz, sample_rate_hz)
    vx_l = finite_diff(x_l, t)
    vy_l = finite_diff(y_l, t)
    ax_l = finite_diff(vx_l, t)
    ay_l = finite_diff(vy_l, t)
    vx_p = finite_diff(x_p, t)
    vy_p = finite_diff(y_p, t)

    r_x = x_p - x_l
    r_y = y_p - y_l
    R = np.sqrt(r_x * r_x + r_y * r_y)
    speed_l = np.sqrt(vx_l * vx_l + vy_l * vy_l)
    lambda_los = np.arctan2(r_y, r_x)
    psi_l = np.arctan2(vy_l, vx_l)
    r_hat_x = r_x / R
    r_hat_y = r_y / R
    t_hat_x = vx_l / speed_l
    t_hat_y = vy_l / speed_l

    derived = pd.DataFrame(
        {
            "run": row["run"], "caught": row["caught"], "fence": row["fence"], "lure": row["lure"],
            "frame_60": raw["frame_60"] if "frame_60" in raw.columns else np.arange(len(raw)),
            "t": t, "x_L": x_l, "y_L": y_l, "x_P": x_p, "y_P": y_p,
            "vx_L": vx_l, "vy_L": vy_l, "ax_L": ax_l, "ay_L": ay_l, "vx_P": vx_p, "vy_P": vy_p,
            "speed_L": speed_l, "R": R, "R_dot": finite_diff(R, t), "lambda": wrap_to_pi(lambda_los),
            "lambda_dot": finite_diff(np.unwrap(lambda_los), t), "psi_L": wrap_to_pi(psi_l),
            "psi_dot": finite_diff(np.unwrap(psi_l), t), "theta": wrap_to_pi(lambda_los - psi_l),
            "theta_dot": finite_diff(np.unwrap(wrap_to_pi(lambda_los - psi_l)), t),
            "a_r": ax_l * r_hat_x + ay_l * r_hat_y,
            "a_theta": ax_l * (-r_hat_y) + ay_l * r_hat_x,
            "a_parallel": ax_l * t_hat_x + ay_l * t_hat_y,
            "a_perp": ax_l * (-t_hat_y) + ay_l * t_hat_x,
            "cutoff_hz": cutoff_hz, "sample_rate_hz": sample_rate_hz,
        }
    ).iloc[1:-1].reset_index(drop=True)
    derived["run_samples"] = len(derived)
    derived["frame_weight"] = 1.0 / (n_runs * len(derived))

    filter_row = {
        "run": row["run"], "raw_csv": row["raw_csv"], "n_raw_rows": len(raw),
        "n_retained_rows": len(derived), "sample_rate_hz": sample_rate_hz,
        "cutoff_x_L_hz": cutoff_x_l, "cutoff_y_L_hz": cutoff_y_l,
        "cutoff_x_P_hz": cutoff_x_p, "cutoff_y_P_hz": cutoff_y_p, "cutoff_hz": cutoff_hz,
        "score_x_L": score_x_l, "score_y_L": score_y_l, "score_x_P": score_x_p, "score_y_P": score_y_p,
    }
    return derived, filter_row
