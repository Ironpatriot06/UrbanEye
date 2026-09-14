# Baseline models

Generated 2026-09-14T17:59:13.044692+00:00

evidence of dataset readiness, not production models. Every model here is an out-of-the-box gradient-boosted tree with default-ish settings.

| Protocol | |
|---|---|
| split_column_requested | split_global |
| split_rationale | these baselines pool every city into one model, so they are scored on split_global, whose train/val/test are ordered on a single wall clock. On the per-source `split` column a pooled model is evaluated partly on the past. |
| predictors | taken from each task manifest — never 'all columns except the target' |
| preprocessing | ordinal encoding fitted on TRAIN only; the 250 most frequent train levels are kept and the tail folded into __OTHER__; unseen levels map to -1; NULL is an explicit __MISSING__ level |
| evaluation | val and test scored separately, never merged |

## resolution

Trained on 400,000 rows in 4.8s. Naive baseline: median resolution time of the TRAIN split.

**val**

| Metric | Model | Naive |
|---|---|---|
| n | 141,349.0000 | 141,349.0000 |
| MAE | 734.1477 | 866.6402 |
| median_AE | 84.7449 | 108.9690 |
| RMSE | 3,135.3575 | 3,683.9514 |
| RMSE_log1p | 1.8063 | 2.4294 |

**test**

| Metric | Model | Naive |
|---|---|---|
| n | 141,349.0000 | 141,349.0000 |
| MAE | 876.4706 | 1,034.7635 |
| median_AE | 127.7391 | 115.7171 |
| RMSE | 3,131.4855 | 3,684.0877 |
| RMSE_log1p | 1.8071 | 2.4026 |

> Compare model MAE against naive MAE. The target spans four orders of magnitude, so median absolute error is the more readable number and MAE is dominated by the tail. Per-city numbers matter more than the pooled one: the three cities have different operating regimes, and Chicago's validation window is the first COVID wave.


## sla

Trained on 13,209 rows in 0.6s. Naive baseline: predict the TRAIN positive rate for every row.

**val**

| Metric | Model | Naive |
|---|---|---|
| n | 3,131.0000 | 3,131.0000 |
| positive_rate | 0.0779 | 0.0779 |
| PR_AUC | 0.2563 | 0.0779 |
| precision@0.5 | 0.5366 | 0.0000 |
| recall@0.5 | 0.0902 | 0.0000 |
| f1@0.5 | 0.1544 | 0.0000 |
| brier | 0.0678 | 0.0730 |
| ROC_AUC | 0.7402 | 0.5000 |

**test**

| Metric | Model | Naive |
|---|---|---|
| n | 3,873.0000 | 3,873.0000 |
| positive_rate | 0.1038 | 0.1038 |
| PR_AUC | 0.2544 | 0.1038 |
| precision@0.5 | 0.5556 | 0.0000 |
| recall@0.5 | 0.0871 | 0.0000 |
| f1@0.5 | 0.1505 | 0.0000 |
| brier | 0.0898 | 0.0931 |
| ROC_AUC | 0.6839 | 0.5000 |

> PR-AUC against the positive rate is the number that matters; ROC-AUC flatters an imbalanced problem. Remember the population: NYC, Jan-Mar 2010, only the complaint types that carry a Due Date. Nothing here transfers to another city's SLA policy.


## hotspot

Trained on 47,092 rows in 1.4s. Naive baseline: None.

**val**

| Metric | Model | Naive |
|---|---|---|
| n | 10,000.0000 | — |
| MAE | 7.8966 | — |
| median_AE | 3.9752 | — |
| RMSE | 14.4789 | — |
| RMSE_log1p | 0.7395 | — |
| poisson_deviance | 7.3930 | — |

**test**

| Metric | Model | Naive |
|---|---|---|
| n | 10,000.0000 | — |
| MAE | 10.5371 | — |
| median_AE | 4.9499 | — |
| RMSE | 32.3692 | — |
| RMSE_log1p | 0.7050 | — |
| poisson_deviance | 14.0186 | — |

> The ROLLING 4-WEEK MEAN is the primary benchmark. Persistence is retained because it is the simplest thing an operator would try, but it is the weaker of the two and quoting only the improvement over it overstates the model. A smoothed trailing mean is what any competent forecaster would reach for first on a weekly count panel, so that is the bar. Poisson deviance is the loss-aligned metric for a count target; MAE is included because it is the one an operator can read.


## duplicate

Trained on 324,501 rows in 1.7s. Naive baseline: predict the TRAIN positive rate for every pair.

**val**

| Metric | Model | Naive |
|---|---|---|
| n | 69,276.0000 | 69,276.0000 |
| positive_rate | 0.5022 | 0.5022 |
| PR_AUC | 0.9883 | 0.5022 |
| precision@0.5 | 0.9616 | 0.5022 |
| recall@0.5 | 0.9477 | 1.0000 |
| f1@0.5 | 0.9546 | 0.6686 |
| brier | 0.0384 | 0.2500 |
| ROC_AUC | 0.9842 | 0.5000 |

**test**

| Metric | Model | Naive |
|---|---|---|
| n | 61,683.0000 | 61,683.0000 |
| positive_rate | 0.5641 | 0.5641 |
| PR_AUC | 0.9978 | 0.5641 |
| precision@0.5 | 0.9843 | 0.5641 |
| recall@0.5 | 0.9893 | 1.0000 |
| f1@0.5 | 0.9868 | 0.7213 |
| brier | 0.0135 | 0.2500 |
| ROC_AUC | 0.9964 | 0.5000 |

> TREAT THESE NUMBERS AS AN UPPER BOUND. The negatives are constructed, and two of the four rules define the negative by distance or time gap — the same quantities the model is given as features. The per-strategy breakdown shows how much of the pooled score comes from the easy negatives. The 1:1 class balance is also a sampling choice; the deployment prevalence of duplicate candidate pairs is far lower, so the operating threshold must be re-derived against it.


## priority

**Not trained.** REFUSED BY DESIGN. There is no priority ground truth in any public 311 dataset. priority_baseline is a deterministic function of config/priority_config.yaml, so a supervised model fitted to it would reproduce the YAML and any accuracy quoted from it would be circular. The policy score stands as the benchmark a future model must beat, once operator-assigned labels exist.
