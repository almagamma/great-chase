#CHOOSE CSV PATHS AND NAMES

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

AA_MANIFEST = SCRIPT_DIR / "aa_manifest.csv"
AA_SAMPLES = SCRIPT_DIR / "aa_samples.csv"
AA_FILTERS = SCRIPT_DIR / "aa_filters.csv"

BB_PSD = SCRIPT_DIR / "bb_psd.csv"
BB_STRIDES = SCRIPT_DIR / "bb_strides.csv"
BB_WINDOWS = SCRIPT_DIR / "bb_stride_windows.csv"

CC_SPEED_BINS = SCRIPT_DIR / "cc_speed_bins.csv"

EE_EVALUATIONS = SCRIPT_DIR / "ee_steering_evaluations.csv"
EE_RUN_ERRORS = SCRIPT_DIR / "ee_steering_run_errors.csv"

FF_EVALUATIONS = SCRIPT_DIR / "ff_velocity_evaluations.csv"
FF_RUN_ERRORS = SCRIPT_DIR / "ff_velocity_run_errors.csv"
