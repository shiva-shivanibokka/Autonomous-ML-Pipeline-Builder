Datasets: 15; seeds: [0, 1, 2]; metric: test ROC AUC (macro OvR for multiclass).

| dataset | logreg | lgbm | xgb | flaml11 | sys_fixed |
|---|---|---|---|---|---|
| balance-scale (11) | 0.972 ± 0.011 | 0.918 ± 0.009 | 0.919 ± 0.006 | 0.948 ± 0.015 | 0.908 ± 0.006 |
| mfeat-morphological (18) | 0.966 ± 0.004 | 0.958 ± 0.004 | 0.959 ± 0.004 | 0.966 ± 0.002 | 0.964 ± 0.003 |
| mfeat-zernike (22) | 0.983 ± 0.001 | 0.965 ± 0.003 | 0.966 ± 0.003 | 0.976 ± 0.002 | 0.975 ± 0.003 |
| tic-tac-toe (50) | 0.993 ± 0.006 | 0.999 ± 0.001 | 0.999 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| vowel (307) | 0.964 ± 0.004 | 0.998 ± 0.001 | 0.997 ± 0.001 | 0.997 ± 0.002 | 0.999 ± 0.000 |
| analcatdata_dmft (469) | 0.594 ± 0.030 | 0.554 ± 0.030 | 0.547 ± 0.029 | 0.577 ± 0.021 | 0.571 ± 0.015 |
| pc4 (1049) | 0.891 ± 0.010 | 0.944 ± 0.003 | 0.942 ± 0.001 | 0.925 ± 0.009 | 0.948 ± 0.007 |
| pc3 (1050) | 0.784 ± 0.056 | 0.842 ± 0.025 | 0.829 ± 0.022 | 0.824 ± 0.009 | 0.826 ± 0.042 |
| kc2 (1063) | 0.819 ± 0.039 | 0.799 ± 0.029 | 0.787 ± 0.034 | 0.831 ± 0.035 | 0.825 ± 0.038 |
| pc1 (1068) | 0.821 ± 0.042 | 0.881 ± 0.033 | 0.863 ± 0.059 | 0.880 ± 0.012 | 0.867 ± 0.048 |
| banknote-authentication (1462) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| cylinder-bands (6332) | 0.773 ± 0.030 | 0.912 ± 0.034 | 0.912 ± 0.033 | 0.874 ± 0.053 | 0.917 ± 0.033 |
| dresses-sales (23381) | 0.629 ± 0.084 | 0.596 ± 0.036 | 0.576 ± 0.009 | 0.598 ± 0.041 | 0.617 ± 0.056 |
| car (40975) | 0.989 ± 0.003 | 1.000 ± 0.001 | 0.999 ± 0.001 | 0.999 ± 0.001 | 1.000 ± 0.000 |
| segment (40984) | 0.979 ± 0.004 | 0.995 ± 0.001 | 0.994 ± 0.001 | 0.995 ± 0.001 | 0.995 ± 0.001 |
| **mean AUC (fail = 0.5)** | 0.8771 | 0.8907 | 0.8860 | 0.8928 | 0.8941 |
| **mean AUC, sys_llm failures -> sys_fixed fallback** | 0.8771 | 0.8907 | 0.8860 | 0.8928 | 0.8941 |
| **mean AUC, failures excluded (15 datasets where every arm completed every seed)** | 0.8771 | 0.8907 | 0.8860 | 0.8928 | 0.8941 |
| **average rank among all 5 arms (1 = best; failed run = 0.5)** | 3.47 | 3.17 | 3.83 | 2.40 | 2.13 |
| failed runs | 0/45 | 0/45 | 0/45 | 0/45 | 0/45 |
| median wall time per run (s) | 0.0 | 0.2 | 0.1 | 22.2 | 10.6 |

Failure modes: completed = datasets where both arms completed every seed; imputed = failed run scores 0.5; fallback_fixed = failed sys_llm run takes sys_fixed's score. Holm is within family (llm / non_llm) and mode.

| comparison | failures | n datasets | mean ΔAUC | 95% bootstrap CI | W/L/T | Wilcoxon p | Holm p |
|---|---|---|---|---|---|---|---|
| sys_fixed − flaml11 | completed | 15 | +0.0013 | [-0.0074, +0.0102] | 8/7/0 | 0.89 | 0.89 |
| sys_fixed − flaml11 | imputed | 15 | +0.0013 | [-0.0074, +0.0102] | 8/7/0 | 0.89 | 0.89 |
| sys_fixed − lgbm | completed | 15 | +0.0034 | [-0.0024, +0.0093] | 12/3/0 | 0.135 | 0.406 |
| sys_fixed − lgbm | imputed | 15 | +0.0034 | [-0.0024, +0.0093] | 12/3/0 | 0.135 | 0.406 |
| sys_fixed − xgb | completed | 15 | +0.0080 | [+0.0014, +0.0157] | 13/2/0 | 0.0151 | 0.0603 |
| sys_fixed − xgb | imputed | 15 | +0.0080 | [+0.0014, +0.0157] | 13/2/0 | 0.0151 | 0.0603 |
| sys_fixed − logreg | completed | 15 | +0.0170 | [-0.0036, +0.0411] | 10/5/0 | 0.208 | 0.416 |
| sys_fixed − logreg | imputed | 15 | +0.0170 | [-0.0036, +0.0411] | 10/5/0 | 0.208 | 0.416 |
