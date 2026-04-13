# COS700 Ransomware Feature-Selection Benchmark

## Setup

From the repository root:

```powershell
python -m venv pyML\.venv
pyML\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
pyML\.venv\Scripts\Activate.ps1
```

Optional XGBoost support:

```powershell
python -m pip install xgboost
```

You can also run the script directly with the virtualenv interpreter:

```powershell
pyML\.venv\Scripts\python.exe pyML\benchmark_feature_selection.py --help
```

## Usage

Run the PE ransomware dataset:

```powershell
python pyML\benchmark_feature_selection.py --csv "Datasets\Ransomware.csv" --output-dir "outputs\ransomware_pe"
```

Run the Android ransomware dataset:

```powershell
python pyML\benchmark_feature_selection.py --csv "Datasets\Android_Ransomeware.csv" --output-dir "outputs\android"
```

Run a quicker sampled experiment:

```powershell
python pyML\benchmark_feature_selection.py --csv "Datasets\Android_Ransomeware.csv" --output-dir "outputs\android_sample" --sample-size 50000
```

Compare selected feature methods and models:

```powershell
python pyML\benchmark_feature_selection.py --csv "Datasets\Ransomware.csv" --output-dir "outputs\compare_models" --sample-size 20000 --folds 3 --selectors baseline mutual_info f_classif l1 random_forest --k-values 10 20 --classifiers logistic random_forest svm knn
```

Include XGBoost after installing it:

```powershell
python pyML\benchmark_feature_selection.py --csv "Datasets\Ransomware.csv" --output-dir "outputs\xgboost_compare" --sample-size 20000 --folds 3 --selectors baseline f_classif l1 random_forest --k-values 10 20 --classifiers logistic random_forest xgboost
```

Include the more expensive RFE wrapper method:

```powershell
python pyML\benchmark_feature_selection.py --csv "Datasets\Ransomware.csv" --output-dir "outputs\ransomware_with_rfe" --selectors baseline mutual_info f_classif l1 random_forest rfe
```

## Command Options

- `--csv`
  - required path to the input dataset
- `--output-dir`
  - output folder for reports, charts, CSVs, and metadata
- `--selectors`
  - feature-selection methods to benchmark
  - supported: `baseline`, `mutual_info`, `f_classif`, `l1`, `random_forest`, `rfe`
- `--classifiers`
  - classifiers used to evaluate selected features
  - supported: `logistic`, `random_forest`, `svm`, `knn`, `xgboost`
- `--k-values`
  - feature subset sizes to test, for example `10 20 30 50`
- `--folds`
  - number of stratified cross-validation folds
- `--sample-size`
  - optional stratified sample size for faster test runs
- `--benign-label`
  - label value treated as benign for multi-class datasets
- `--target-column`
  - manual target-column override if auto-detection is wrong
- `--random-state`
  - random seed for reproducible splits and sampling

## Output Files

Each run writes files using the input dataset name as a prefix. For `Ransomware.csv`, outputs look like:

- `ransomware_benchmark_report.md`
  - start here; readable report with a recommendation, leaderboard, baseline comparison, and stable features
- `ransomware_leaderboard_f1.svg`
  - visual chart of the top selector/model combinations by F1 score
- `ransomware_top_features.svg`
  - visual chart of the most consistently selected features for the best configuration
- `ransomware_benchmark_results.csv`
  - metrics for every selector/classifier/k combination
- `ransomware_selected_feature_frequencies.csv`
  - how often each feature was selected across folds
- `ransomware_run_metadata.json`
  - dataset details, dropped columns, selected settings, and class balance
- `ransomware_benchmark_summary.txt`
  - console-style text summary

To make a PDF, open `*_benchmark_report.md` in an editor or browser that renders Markdown, then print or export it as a PDF. The SVG charts are saved in the same output folder and referenced by the report.
