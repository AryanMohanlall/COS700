# Feature Selection Benchmark Report: Ransomware.csv

## Summary

Highest fitness_score: l1 (k=10) at 0.9799. If you want a leaner subset with nearly the same fitness_score, use backward_elimination (k=5) at 0.9741 with about 5.0 features.

## Dataset

- Rows evaluated: **500**
- Usable numeric features: **54**
- Class 0 count: **150**
- Class 1 count: **350**
- Folds: **3**
- Selectors: **info_gain, variance_threshold, chi2, forward_selection, backward_elimination, l1**
- k values: **5, 10, 20**

## Top Results

| Configuration | Stability | Fitness score | Features | Time |
| --- | --- | --- | --- | --- |
| l1 (k=10) | 0.7172 | 0.9799 | 10.0 | 0.56 |
| forward_selection (k=10) | 0.4057 | 0.9771 | 10.0 | 1.20 |
| backward_elimination (k=5) | 0.4286 | 0.9741 | 5.0 | 1.51 |
| l1 (k=20) | 0.4462 | 0.9731 | 20.0 | 0.53 |
| info_gain (k=10) | 0.5873 | 0.9687 | 10.0 | 0.07 |
| info_gain (k=20) | 0.6686 | 0.9685 | 20.0 | 0.07 |
| forward_selection (k=20) | 0.7172 | 0.9685 | 20.0 | 1.76 |
| backward_elimination (k=20) | 0.6686 | 0.9671 | 20.0 | 1.69 |
| backward_elimination (k=10) | 0.3968 | 0.9626 | 10.0 | 1.49 |
| forward_selection (k=5) | 0.2037 | 0.9605 | 5.0 | 0.72 |

## Stable Features

Best configuration: l1 (k=10)
- DllCharacteristics: selected in 3 fold(s)
- MajorOperatingSystemVersion: selected in 3 fold(s)
- SectionsMaxEntropy: selected in 3 fold(s)
- SizeOfOptionalHeader: selected in 3 fold(s)
- SizeOfStackReserve: selected in 3 fold(s)
- Subsystem: selected in 3 fold(s)
- VersionInformationSize: selected in 3 fold(s)
- ImportsNbDLL: selected in 2 fold(s)
- Machine: selected in 2 fold(s)
- MinorLinkerVersion: selected in 2 fold(s)
