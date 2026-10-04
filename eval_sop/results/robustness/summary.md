Datasets: 16; seeds: [0, 1, 2]; metric: test ROC AUC (macro OvR for multiclass).

| dataset | logreg | lgbm | xgb | flaml11 | sys_fixed |
|---|---|---|---|---|---|
| balance-scale (11) | 0.972 ± 0.011 | 0.918 ± 0.009 | 0.919 ± 0.006 | 0.948 ± 0.015 | 0.908 ± 0.006 |
| mfeat-morphological (18) | 0.966 ± 0.004 | 0.958 ± 0.004 | 0.959 ± 0.004 | 0.966 ± 0.002 | 0.964 ± 0.003 |
| mfeat-zernike (22) | 0.983 ± 0.001 | 0.965 ± 0.003 | 0.966 ± 0.003 | 0.976 ± 0.002 | 0.975 ± 0.003 |
| splice (46) | 0.990 ± 0.002 | 0.996 ± 0.000 | 0.996 ± 0.000 | 0.995 ± 0.000 | 0.996 ± 0.001 |
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
| **mean AUC (fail = 0.5)** | 0.8842 | 0.8973 | 0.8929 | 0.8992 | 0.9004 |
| **mean AUC, sys_llm failures -> sys_fixed fallback** | 0.8842 | 0.8973 | 0.8929 | 0.8992 | 0.9004 |
| **mean AUC, failures excluded (16 datasets where every arm completed every seed)** | 0.8842 | 0.8973 | 0.8929 | 0.8992 | 0.9004 |
| **average rank among all 5 arms (1 = best; failed run = 0.5)** | 3.56 | 3.09 | 3.66 | 2.50 | 2.19 |
| candidate models that errored inside completed runs | none | none | none | none | xgboost on 4 ds (12 runs) |
| failed runs | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 |
| median wall time per run (s) | 0.1 | 0.2 | 0.2 | 23.2 | 11.0 |

Failure modes: completed = datasets where both arms completed every seed; imputed = failed run scores 0.5; fallback_fixed = failed sys_llm run takes sys_fixed's score. Holm is within family (llm / non_llm) and mode.

| comparison | failures | n datasets | mean ΔAUC | 95% bootstrap CI | W/L/T | Wilcoxon p | Holm p |
|---|---|---|---|---|---|---|---|
| sys_fixed − flaml11 | completed | 16 | +0.0013 | [-0.0071, +0.0095] | 9/7/0 | 0.821 | 0.821 |
| sys_fixed − flaml11 | imputed | 16 | +0.0013 | [-0.0071, +0.0095] | 9/7/0 | 0.821 | 0.821 |
| sys_fixed − lgbm | completed | 16 | +0.0032 | [-0.0022, +0.0087] | 12/4/0 | 0.13 | 0.389 |
| sys_fixed − lgbm | imputed | 16 | +0.0032 | [-0.0022, +0.0087] | 12/4/0 | 0.13 | 0.389 |
| sys_fixed − xgb | completed | 16 | +0.0075 | [+0.0014, +0.0149] | 13/3/0 | 0.0182 | 0.073 |
| sys_fixed − xgb | imputed | 16 | +0.0075 | [+0.0014, +0.0149] | 13/3/0 | 0.0182 | 0.073 |
| sys_fixed − logreg | completed | 16 | +0.0162 | [-0.0036, +0.0387] | 11/5/0 | 0.175 | 0.389 |
| sys_fixed − logreg | imputed | 16 | +0.0162 | [-0.0036, +0.0387] | 11/5/0 | 0.175 | 0.389 |
