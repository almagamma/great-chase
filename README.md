Run `nb_basic.ipynb`. The first cell sets the input paths and run list:

- `RAW_DIR`: folder containing the raw tracking CSVs.
- `RUN_LIST_PATH`: CSV listing the runs to include.

The notebook writes analysis CSVs into its own directory.
It also displays small summary tables in-notebook, which are not saved as CSVs.

## Folders

- data/: raw tracking data CSVs for each run.
- info/: CSVs containing lists of runs and metadata. runs.csv is all runs. If desired, analysis can be restricted to a subset of runs by using a different run list. For example, onerun.csv lists one run.
- scripts/: the notebook, helper scripts, and generated CSVs.

## Notebook

`nb_basic.ipynb` executes all the scripts. Run it from start to finish.
The main output tables are named by prefix:

- `aa_*`: manifest, filtered samples, and filtering diagnostics.
- `bb_*`: power spectra, stride-period estimates, and stride-window data.
- `cc_*`: speed-binned stride-window data.
- `ee_*`: steering model evaluations and run-wise errors.
- `ff_*`: velocity model evaluations and run-wise errors.

## Scripts

- `lib_paths.py`: records output paths for CSVs written by the notebook (adjust as desired).
- `lib_math.py`: simple numerics and stats functions.
- `lib_filter.py`: coordinate smoothing and cutoff selection.
- `lib_samples.py`: raw data import and compute derived kinematics.
- `lib_grid.py`: simple functions used for grid search to optimize steering model params.
- `lib_random_search.py`: functions used for random-search optimization of velocity model params.
- `lib_steering_model.py`: steering model definitions and commands.
- `lib_steering_runs.py`: steering simulation and run-wise error evaluation.
- `lib_steering_fit.py`: steering parameter search and selected-fit summary.
- `lib_velocity_model.py`: velocity model definitions.
- `lib_velocity_runs.py`: velocity simulation and run-wise error evaluation.
- `lib_velocity_anchors.py`: simple velocity search anchor points.
- `lib_velocity_fit.py`: velocity parameter search and selected-fit summary.
