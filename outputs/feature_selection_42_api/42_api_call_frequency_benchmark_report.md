# Feature Selection Benchmark Report: 42_API_CALL_FREQUENCY.csv

## Summary

Current front-runner: rfe + xgboost (k=10) with fitness_score=0.9851, stability=0.5018, and about 10.0 selected features.

## Dataset

- Rows evaluated: **3,083**
- Usable numeric features: **279**
- Class 0 count: **570**
- Class 1 count: **2,513**
- Folds: **3**
- Selectors: **variance_threshold, chi2, info_gain, rfe, l1**
- Models: **random_forest, svm, xgboost**
- k values: **10, 20, 30**

## Top Results

| Configuration | Stability | Fitness score | Features | Time |
| --- | --- | --- | --- | --- |
| rfe + xgboost (k=10) | 0.5018 | 0.9851 | 10.0 | 6.69 |
| info_gain + xgboost (k=10) | 0.5018 | 0.9849 | 10.0 | 5.72 |
| info_gain + xgboost (k=30) | 0.8762 | 0.9845 | 30.0 | 5.98 |
| l1 + xgboost (k=30) | 0.4180 | 0.9845 | 30.0 | 6.45 |
| rfe + random_forest (k=10) | 0.5018 | 0.9843 | 10.0 | 6.63 |
| rfe + xgboost (k=20) | 0.7943 | 0.9843 | 20.0 | 288.30 |
| info_gain + xgboost (k=20) | 0.8182 | 0.9843 | 20.0 | 5.93 |
| chi2 + xgboost (k=10) | 0.6744 | 0.9841 | 10.0 | 0.29 |
| rfe + xgboost (k=30) | 0.8762 | 0.9841 | 30.0 | 6.11 |
| chi2 + xgboost (k=20) | 0.8470 | 0.9837 | 20.0 | 0.25 |

## Stable Features

Best configuration: rfe + xgboost (k=10)
- CoInitializeEx: selected in 3 fold(s)
- CreateDirectoryW: selected in 3 fold(s)
- DrawTextExW: selected in 3 fold(s)
- NtOpenKey: selected in 3 fold(s)
- NtOpenSection: selected in 3 fold(s)
- NtQueryDirectoryFile: selected in 3 fold(s)
- GetFileSizeEx: selected in 2 fold(s)
- LoadStringW: selected in 2 fold(s)
- CoCreateInstance: selected in 1 fold(s)
- NtOpenProcess: selected in 1 fold(s)
