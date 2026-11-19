# Does tuning help? Optuna search for XGBoost

Search: TPE (seed 0); trial 0 = default recipe; no pruning, objective = validation AUPRC of a model trained on the fit block, early-stopped on the validation block (test steps never read); trains on steps 1-29, validates on 30-34. Both recipes then run the standard protocol (seeds [0, 1, 2], refit on 1-34, thresholds from validation only) and are scored once on test steps 35-49. Threads: 12 (arms suffixed `_Nt`: N threads). Wall time 186.3 min, of which about two hours was a manual suspension of the process; active compute was about 57 min (searches 26.3 min).

## XGBoost, local + aggregated (165) (`xgb_all`)

100 trials in 10.1 min. Validation AUPRC (search seed): default 0.9310 (50 trees), best 0.9929 (trial 93, 203 trees).

Tuned parameters: max_depth=10, learning_rate=0.04329, min_child_weight=3.362, subsample=0.672, colsample_bytree=0.3047, reg_lambda=0.8815, gamma=1.048, scale_pos_weight=1.256

| Recipe | Val AUPRC (stage 1) | Trees | F1 at 0.5 | Precision | Recall | AUPRC | Cost at 0.5 | Cost, rolling-origin thr. | Cost, original thr. | F1 steps 43+ |
|---|---|---|---|---|---|---|---|---|---|---|
| default | 0.9289 | 50, 5, 52 | 0.526 ± 0.455 | 0.586 | 0.476 | 0.777 ± 0.006 | 0.3444 | 0.2029 | 0.2213 | 0.021 |
| tuned | 0.9912 | 203, 161, 285 | 0.813 ± 0.001 | 0.926 | 0.725 | 0.804 ± 0.001 | 0.1827 | 0.1855 | 0.1840 | 0.029 |
| default_2t | 0.9806 | 452, 780, 565 | 0.817 ± 0.002 | 0.931 | 0.728 | 0.800 ± 0.001 | 0.1805 | 0.1878 | 0.1966 | 0.029 |
| tuned_2t | 0.9906 | 255, 198, 223 | 0.809 ± 0.004 | 0.916 | 0.725 | 0.802 ± 0.002 | 0.1833 | 0.1885 | 0.1868 | 0.029 |

Differences against `default` (12 threads), 95% paired time-step bootstrap interval (lower cost is better):

| Recipe | F1 at 0.5 | AUPRC | cost at 0.5 | cost, rolling-origin threshold | cost, original threshold |
|---|---|---|---|---|---|
| tuned - default | +0.283 [+0.232, +0.310] | +0.027 [+0.016, +0.043] | -0.1616 [-0.2481, -0.0756] | -0.0174 [-0.0265, -0.0093] | -0.0374 [-0.0478, -0.0294] |
| tuned_2t - default | +0.280 [+0.228, +0.308] | +0.025 [+0.014, +0.042] | -0.1611 [-0.2474, -0.0745] | -0.0143 [-0.0224, -0.0067] | -0.0345 [-0.0455, -0.0256] |

Differences against `default_2t` (2 threads), 95% paired time-step bootstrap interval (lower cost is better):

| Recipe | F1 at 0.5 | AUPRC | cost at 0.5 | cost, rolling-origin threshold | cost, original threshold |
|---|---|---|---|---|---|
| tuned - default_2t | -0.004 [-0.009, +0.002] | +0.004 [-0.001, +0.010] | +0.0023 [+0.0001, +0.0051] | -0.0023 [-0.0072, +0.0024] | -0.0127 [-0.0199, -0.0053] |
| tuned_2t - default_2t | -0.008 [-0.012, -0.004] | +0.002 [-0.004, +0.009] | +0.0028 [+0.0005, +0.0058] | +0.0007 [-0.0025, +0.0038] | -0.0098 [-0.0160, -0.0036] |

## XGBoost, 165 + engineered graph features (`xgb_all_graph`)

100 trials in 8.6 min. Validation AUPRC (search seed): default 0.9834 (163 trees), best 0.9917 (trial 98, 84 trees).

Tuned parameters: max_depth=10, learning_rate=0.2439, min_child_weight=1.84, subsample=0.6679, colsample_bytree=0.5756, reg_lambda=0.5705, gamma=1.118, scale_pos_weight=1.069

| Recipe | Val AUPRC (stage 1) | Trees | F1 at 0.5 | Precision | Recall | AUPRC | Cost at 0.5 | Cost, rolling-origin thr. | Cost, original thr. | F1 steps 43+ |
|---|---|---|---|---|---|---|---|---|---|---|
| default | 0.9801 | 163, 249, 105 | 0.808 ± 0.009 | 0.917 | 0.721 | 0.810 ± 0.005 | 0.1852 | 0.1874 | 0.1952 | 0.035 |
| tuned | 0.9856 | 84, 26, 16 | 0.799 ± 0.012 | 0.894 | 0.723 | 0.798 ± 0.001 | 0.1858 | 0.1955 | 0.1989 | 0.034 |
| default_2t | 0.9761 | 365, 53, 376 | 0.810 ± 0.017 | 0.932 | 0.716 | 0.803 ± 0.014 | 0.1878 | 0.1864 | 0.1962 | 0.023 |
| tuned_2t | 0.9889 | 20, 41, 27 | 0.797 ± 0.012 | 0.891 | 0.721 | 0.798 ± 0.004 | 0.1873 | 0.2002 | 0.1909 | 0.025 |

Differences against `default` (12 threads), 95% paired time-step bootstrap interval (lower cost is better):

| Recipe | F1 at 0.5 | AUPRC | cost at 0.5 | cost, rolling-origin threshold | cost, original threshold |
|---|---|---|---|---|---|
| tuned - default | -0.009 [-0.017, -0.002] | -0.012 [-0.023, -0.006] | +0.0005 [-0.0013, +0.0025] | +0.0081 [+0.0037, +0.0128] | +0.0037 [+0.0003, +0.0078] |
| tuned_2t - default | -0.011 [-0.021, -0.005] | -0.012 [-0.023, -0.005] | +0.0021 [-0.0018, +0.0054] | +0.0128 [+0.0080, +0.0174] | -0.0043 [-0.0115, +0.0040] |

Differences against `default_2t` (2 threads), 95% paired time-step bootstrap interval (lower cost is better):

| Recipe | F1 at 0.5 | AUPRC | cost at 0.5 | cost, rolling-origin threshold | cost, original threshold |
|---|---|---|---|---|---|
| tuned - default_2t | -0.011 [-0.020, -0.003] | -0.005 [-0.011, -0.003] | -0.0020 [-0.0054, +0.0014] | +0.0091 [+0.0045, +0.0143] | +0.0027 [-0.0018, +0.0071] |
| tuned_2t - default_2t | -0.014 [-0.024, -0.006] | -0.005 [-0.013, -0.001] | -0.0005 [-0.0037, +0.0031] | +0.0139 [+0.0090, +0.0193] | -0.0053 [-0.0129, +0.0030] |

## XGBoost, local features (93) (`xgb_local`)

100 trials in 7.6 min. Validation AUPRC (search seed): default 0.9887 (298 trees), best 0.9901 (trial 44, 180 trees).

Tuned parameters: max_depth=7, learning_rate=0.04934, min_child_weight=0.7832, subsample=0.7607, colsample_bytree=0.7201, reg_lambda=0.3639, gamma=0.8748, scale_pos_weight=4.614

| Recipe | Val AUPRC (stage 1) | Trees | F1 at 0.5 | Precision | Recall | AUPRC | Cost at 0.5 | Cost, rolling-origin thr. | Cost, original thr. | F1 steps 43+ |
|---|---|---|---|---|---|---|---|---|---|---|
| default | 0.9874 | 298, 217, 104 | 0.774 ± 0.007 | 0.865 | 0.701 | 0.785 ± 0.002 | 0.2017 | 0.2018 | 0.2110 | 0.029 |
| tuned | 0.9891 | 180, 291, 230 | 0.740 ± 0.011 | 0.748 | 0.733 | 0.789 ± 0.001 | 0.1893 | 0.2002 | 0.2016 | 0.039 |
| default_2t | 0.9880 | 374, 280, 185 | 0.775 ± 0.004 | 0.860 | 0.706 | 0.787 ± 0.001 | 0.1986 | 0.2101 | 0.2148 | 0.029 |
| tuned_2t | 0.9889 | 237, 198, 281 | 0.738 ± 0.006 | 0.742 | 0.735 | 0.790 ± 0.002 | 0.1890 | 0.2009 | 0.2037 | 0.041 |

Differences against `default` (12 threads), 95% paired time-step bootstrap interval (lower cost is better):

| Recipe | F1 at 0.5 | AUPRC | cost at 0.5 | cost, rolling-origin threshold | cost, original threshold |
|---|---|---|---|---|---|
| tuned - default | -0.034 [-0.058, -0.016] | +0.004 [+0.002, +0.008] | -0.0124 [-0.0248, +0.0000] | -0.0016 [-0.0049, +0.0014] | -0.0094 [-0.0154, -0.0042] |
| tuned_2t - default | -0.036 [-0.060, -0.018] | +0.005 [+0.002, +0.009] | -0.0127 [-0.0252, -0.0001] | -0.0009 [-0.0042, +0.0018] | -0.0073 [-0.0122, -0.0035] |

Differences against `default_2t` (2 threads), 95% paired time-step bootstrap interval (lower cost is better):

| Recipe | F1 at 0.5 | AUPRC | cost at 0.5 | cost, rolling-origin threshold | cost, original threshold |
|---|---|---|---|---|---|
| tuned - default_2t | -0.035 [-0.056, -0.019] | +0.002 [+0.001, +0.005] | -0.0093 [-0.0194, +0.0006] | -0.0099 [-0.0168, -0.0034] | -0.0132 [-0.0195, -0.0070] |
| tuned_2t - default_2t | -0.038 [-0.058, -0.022] | +0.003 [+0.000, +0.006] | -0.0096 [-0.0197, +0.0007] | -0.0093 [-0.0154, -0.0036] | -0.0111 [-0.0157, -0.0070] |
