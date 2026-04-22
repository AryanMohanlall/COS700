# Feature Selection Benchmark Report: Ransomware.csv

## Summary

Highest fitness_score: variance_threshold + random_forest (k=20) at 0.9871. 

## Dataset

- Rows evaluated: **500**
- Usable numeric features: **54**
- Class 0 count: **150**
- Class 1 count: **350**
- Folds: **3**
- Selectors: **info_gain, variance_threshold, chi2, l1**
- Models: **random_forest, svm**
- k values: **5, 10, 20**

## Top Results

| Configuration | Stability | Fitness score | Features | Time |
| --- | --- | --- | --- | --- |
| variance_threshold + random_forest (k=20) | 0.9365 | 0.9871 | 20.0 | 0.03 |
| chi2 + random_forest (k=20) | 0.7943 | 0.9870 | 20.0 | 0.02 |
| variance_threshold + random_forest (k=10) | 0.5079 | 0.9856 | 10.0 | 0.03 |
| l1 + random_forest (k=10) | 0.7172 | 0.9830 | 10.0 | 0.14 |
| chi2 + random_forest (k=10) | 0.6239 | 0.9827 | 10.0 | 0.01 |
| l1 + random_forest (k=20) | 0.4462 | 0.9816 | 20.0 | 0.12 |
| info_gain + random_forest (k=10) | 0.5873 | 0.9816 | 10.0 | 0.31 |
| chi2 + random_forest (k=5) | 0.7778 | 0.9814 | 5.0 | 0.01 |
| info_gain + random_forest (k=20) | 0.6686 | 0.9802 | 20.0 | 0.43 |
| info_gain + svm (k=5) | 0.7778 | 0.9799 | 5.0 | 0.43 |

## Stable Features

Best configuration: variance_threshold + random_forest (k=20)
- AddressOfEntryPoint: selected in 3 fold(s)
- BaseOfCode: selected in 3 fold(s)
- BaseOfData: selected in 3 fold(s)
- CheckSum: selected in 3 fold(s)
- ImageBase: selected in 3 fold(s)
- ResourcesMaxSize: selected in 3 fold(s)
- ResourcesMeanSize: selected in 3 fold(s)
- SectionMaxRawsize: selected in 3 fold(s)
- SectionMaxVirtualsize: selected in 3 fold(s)
- SectionsMeanRawsize: selected in 3 fold(s)
