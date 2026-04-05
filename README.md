# COS700

Feature-selection benchmark tooling for the ransomware datasets in [`Datasets`](C:\Users\Aryan\Documents\Personal\COS700\Datasets).

## What The Benchmark Does

The benchmark is designed for the main research question:

"Which feature-selection method finds a small, stable, high-value subset of features for ransomware detection?"

It:

- loads either ransomware CSV in this repo
- auto-detects the label column
- converts the task to binary detection: ransomware vs benign
- drops likely identifier or leakage columns
- keeps numeric features
- benchmarks multiple feature-selection methods inside cross-validation
- evaluates the selected features with fixed classifiers
- exports both model results and selected-feature frequencies

## Supported Datasets

- [`Datasets/Ransomware.csv`](\Datasets\Ransomware.csv)
  - pipe-delimited PE malware feature dataset
  - label column: `legitimate`
  - mapped to `ransomware=1`, `benign=0`
- [`Datasets/Android_Ransomeware.csv`](\Datasets\Android_Ransomeware.csv)
  - comma-delimited Android/network-flow dataset
  - label column: `Label`
  - mapped to `ransomware=1`, `Benign=0`

## Python Dependencies

Create and activate a virtual environment first:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

Then install the required packages:

```powershell
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn
```

If you want to compare against XGBoost as well, install:

```powershell
python -m pip install xgboost
```

## Main Script

Use [`benchmark_feature_selection.py`](C:\Users\Aryan\Documents\Personal\COS700\benchmark_feature_selection.py).

## Example Runs

Run against the PE ransomware dataset:

```powershell
python benchmark_feature_selection.py --csv "Datasets\Ransomware.csv" --output-dir "outputs\ransomware_pe"
```

Run against the Android ransomware dataset:

```powershell
python benchmark_feature_selection.py --csv "Datasets\Android_Ransomeware.csv" --output-dir "outputs\android"
```

Use a stratified sample for quicker iteration:

```powershell
python benchmark_feature_selection.py --csv "Datasets\Android_Ransomeware.csv" --output-dir "outputs\android_sample" --sample-size 50000
```

Compare multiple classifiers on the same feature selectors:

```powershell
python benchmark_feature_selection.py --csv "Datasets\Ransomware.csv" --output-dir "outputs\compare_models" --sample-size 20000 --folds 3 --selectors baseline mutual_info f_classif l1 --k-values 10 20 --classifiers logistic random_forest svm knn xgboost
```

Include the more expensive wrapper method:

```powershell
python benchmark_feature_selection.py --csv "Datasets\Ransomware.csv" --output-dir "outputs\ransomware_with_rfe" --selectors baseline mutual_info f_classif l1 random_forest rfe
```

## Output Files

Each run writes:

- `benchmark_results.csv`
  - mean and standard deviation of F1, precision, recall, ROC-AUC, PR-AUC, runtime, and selected-feature count
- `selected_feature_frequencies.csv`
  - how often each feature was selected across folds for each configuration
- `run_metadata.json`
  - dataset info, dropped columns, and benchmark settings

## Suggested Research Workflow

1. Run a smaller sampled benchmark first to narrow the selector list.
2. Re-run the strongest selectors on the full dataset.
3. Compare:
   - F1
   - recall
   - PR-AUC
   - number of selected features
   - stability of selected features
4. Use `selected_feature_frequencies.csv` to identify the most consistently valuable ransomware features.
5. In your report, discuss both predictive quality and feature stability.
