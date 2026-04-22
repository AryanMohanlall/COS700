# COS700 Feature Selection Benchmark

This project benchmarks feature-selection algorithms on the ransomware dataset.

## What this project does

It compares these feature-selection methods:

- `info_gain`
- `variance_threshold`
- `chi2`
- `forward_selection`
- `backward_elimination`
- `l1`

Each selector can be evaluated with these models:

- `random_forest`
- `svm`
- `xgboost`

Each selector-model combination is scored using:

- `fitness_score`: mean cross-validated `F1` score
- `time`: average selection time
- `amount_of_features_chosen`

## Project structure

- `Datasets/` contains the CSV datasets
- `pyML/benchmark_feature_selection.py` is the main script
- `outputs/` contains generated CSV, SVG, and report files

## Before you run

Open PowerShell in the project root:

```powershell
cd C:\Users\Aryan\Documents\Personal\COS700
```

Activate the virtual environment:

```powershell
pyML\.venv\Scripts\Activate.ps1
```

If packages are missing, install them:

```powershell
python -m pip install pandas numpy scikit-learn
```

If you want to use `xgboost`, also install:

```powershell
python -m pip install xgboost
```

## Main command

Run the feature-selection benchmark with:

```powershell
pyML\.venv\Scripts\python.exe pyML\benchmark_feature_selection.py `
  --csv Datasets\Ransomware.csv `
  --output-dir outputs\feature_selection_final `
  --selectors info_gain variance_threshold chi2 forward_selection backward_elimination l1 `
  --models random_forest svm xgboost `
  --k-values 5 10 20 `
  --folds 3 `
  --sample-size 500
```

## What the command means

- `--csv`: dataset to use
- `--output-dir`: where results will be saved
- `--selectors`: feature-selection algorithms to test
- `--models`: models used to score each selector
- `--k-values`: number of top features to keep
- `--folds`: cross-validation folds
- `--sample-size`: smaller sample for faster testing

## Output files

After the run, check the output folder for:

- `*_benchmark_results.csv`: main spreadsheet-friendly results
- `*_benchmark_results.xlsx`: Excel version of the main results
- `*_fitness_score.svg`: bar chart of top results
- `*_selected_feature_frequencies.csv`: how often features were selected
- `*_benchmark_report.md`: readable summary report
- `*_run_metadata.json`: run settings and dataset details

## Main results columns

The results CSV contains:

- `time`
- `amount_of_features_chosen`
- `model`
- `selection_algorithm`
- `fitness_score`

## Faster run option

If you want a quicker run, skip the slowest method:

```powershell
pyML\.venv\Scripts\python.exe pyML\benchmark_feature_selection.py `
  --csv Datasets\Ransomware.csv `
  --output-dir outputs\feature_selection_fast `
  --selectors info_gain variance_threshold chi2 forward_selection l1 `
  --models random_forest svm `
  --k-values 5 10 20 `
  --folds 3 `
  --sample-size 500
```

`backward_elimination` is the slowest selector, so it may take much longer than the others.

## Run backward elimination separately

```powershell
pyML\.venv\Scripts\python.exe pyML\benchmark_feature_selection.py `
  --csv Datasets\Ransomware.csv `
  --output-dir outputs\feature_selection_backward_only `
  --selectors backward_elimination `
  --models random_forest svm `
  --k-values 5 10 20 `
  --folds 3 `
  --sample-size 500
```
