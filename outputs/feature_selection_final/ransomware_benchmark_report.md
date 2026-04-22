# Feature Selection Benchmark Report: Ransomware.csv

## Summary

Highest fitness_score: l1 (k=10) at 0.9792

## Dataset

- Rows evaluated: **1,000**
- Usable numeric features: **54**
- Class 0 count: **299**
- Class 1 count: **701**
- Folds: **3**
- Selectors: **info_gain, variance_threshold, chi2, forward_selection, backward_elimination, l1**
- k values: **5, 10, 20**

## Top Results

| Configuration | Stability | Fitness score | Features | Time |
| --- | --- | --- | --- | --- |
| l1 (k=10) | 0.5812 | 0.9792 | 10.0 | 0.11 |
| backward_elimination (k=20) | 0.8207 | 0.9785 | 20.0 | 0.27 |
| forward_selection (k=20) | 0.8207 | 0.9778 | 20.0 | 1.83 |
| backward_elimination (k=10) | 0.5385 | 0.9777 | 10.0 | 1.36 |
| info_gain (k=20) | 0.7918 | 0.9771 | 20.0 | 0.10 |
| l1 (k=20) | 0.4828 | 0.9763 | 20.0 | 0.10 |
| forward_selection (k=10) | 0.4484 | 0.9736 | 10.0 | 1.24 |
| forward_selection (k=5) | 0.4286 | 0.9730 | 5.0 | 0.76 |
| info_gain (k=10) | 0.5385 | 0.9714 | 10.0 | 0.10 |
| backward_elimination (k=5) | 0.5079 | 0.9704 | 5.0 | 1.65 |

## Stable Features

Best configuration: l1 (k=10)
- DllCharacteristics: selected in 3 fold(s)
- MajorOperatingSystemVersion: selected in 3 fold(s)
- SectionsMaxEntropy: selected in 3 fold(s)
- SizeOfOptionalHeader: selected in 3 fold(s)
- SizeOfStackReserve: selected in 3 fold(s)
- Subsystem: selected in 3 fold(s)
- VersionInformationSize: selected in 3 fold(s)
- MinorLinkerVersion: selected in 2 fold(s)
- ExportNb: selected in 1 fold(s)
- ImportsNb: selected in 1 fold(s)
