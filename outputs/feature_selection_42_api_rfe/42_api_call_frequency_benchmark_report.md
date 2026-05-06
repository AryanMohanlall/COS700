# Feature Selection Benchmark Report: 42_API_CALL_FREQUENCY.csv

## Summary

Highest fitness_score: chi2 + xgboost (k=30) at 0.9846. If you want a leaner subset with nearly the same fitness_score, use rfe + xgboost (k=10) at 0.9840 with about 10.0 features.

## Dataset

- Rows evaluated: **2,000**
- Usable numeric features: **279**
- Class 0 count: **370**
- Class 1 count: **1,630**
- Folds: **3**
- Selectors: **variance_threshold, chi2, info_gain, rfe, l1**
- Models: **random_forest, svm, xgboost**
- k values: **10, 20, 30**

## Top Results

| Configuration | Stability | Fitness score | Features | Time |
| --- | --- | --- | --- | --- |
| chi2 + xgboost (k=30) | 0.6383 | 0.9846 | 30.0 | 0.06 |
| variance_threshold + xgboost (k=30) | 0.5817 | 0.9840 | 30.0 | 0.04 |
| rfe + xgboost (k=20) | 0.6949 | 0.9840 | 20.0 | 0.86 |
| chi2 + xgboost (k=20) | 0.6908 | 0.9840 | 20.0 | 0.05 |
| rfe + xgboost (k=10) | 0.5079 | 0.9840 | 10.0 | 0.89 |
| info_gain + xgboost (k=20) | 0.7702 | 0.9840 | 20.0 | 0.83 |
| l1 + xgboost (k=30) | 0.3346 | 0.9840 | 30.0 | 0.91 |
| variance_threshold + xgboost (k=20) | 0.5432 | 0.9837 | 20.0 | 0.04 |
| chi2 + xgboost (k=10) | 0.7172 | 0.9837 | 10.0 | 0.04 |
| info_gain + xgboost (k=10) | 0.5385 | 0.9837 | 10.0 | 0.83 |

## Stable Features

Best configuration: chi2 + xgboost (k=30)
- CreateDirectoryW: selected in 3 fold(s)
- CryptCreateHash: selected in 3 fold(s)
- CryptHashData: selected in 3 fold(s)
- DrawTextExW: selected in 3 fold(s)
- GetFileAttributesW: selected in 3 fold(s)
- GetSystemMetrics: selected in 3 fold(s)
- NtDelayExecution: selected in 3 fold(s)
- NtQuerySystemInformation: selected in 3 fold(s)
- NtReadFile: selected in 3 fold(s)
- NtReadVirtualMemory: selected in 3 fold(s)
