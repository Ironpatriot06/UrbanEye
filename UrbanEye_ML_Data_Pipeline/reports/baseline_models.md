# Baseline models

Generated 2026-09-14T12:41:04.973171+00:00

evidence of dataset readiness, not production models. Every model here is an out-of-the-box gradient-boosted tree with default-ish settings.

| Protocol | |
|---|---|
| predictors | taken from each task manifest — never 'all columns except the target' |
| preprocessing | ordinal encoding fitted on TRAIN only; the 250 most frequent train levels are kept and the tail folded into __OTHER__; unseen levels map to -1; NULL is an explicit __MISSING__ level |
| evaluation | val and test scored separately, never merged |

## resolution

Trained on 400,000 rows in 3.9s. Naive baseline: median resolution time of the TRAIN split.

**val**

| Metric | Model | Naive |
|---|---|---|
| n | 153,980.0000 | 153,980.0000 |
| MAE | 681.8983 | 856.2373 |
| median_AE | 47.0120 | 91.7615 |
| RMSE | 2,858.0283 | 3,475.2351 |
| RMSE_log1p | 1.7851 | 2.7595 |

**test**

| Metric | Model | Naive |
|---|---|---|
| n | 153,989.0000 | 153,989.0000 |
| MAE | 733.5452 | 896.9345 |
| median_AE | 84.7188 | 92.3229 |
| RMSE | 2,825.6546 | 3,428.2750 |
| RMSE_log1p | 1.8812 | 2.6443 |

> Compare model MAE against naive MAE. The target spans four orders of magnitude, so median absolute error is the more readable number and MAE is dominated by the tail. Per-city numbers matter more than the pooled one: the three cities have different operating regimes, and Chicago's validation window is the first COVID wave.


## sla

Trained on 13,053 rows in 0.5s. Naive baseline: predict the TRAIN positive rate for every row.

**val**

| Metric | Model | Naive |
|---|---|---|
| n | 3,223.0000 | 3,223.0000 |
| positive_rate | 0.0838 | 0.0838 |
| PR_AUC | 0.3491 | 0.0838 |
| precision@0.5 | 0.7742 | 0.0000 |
| recall@0.5 | 0.1778 | 0.0000 |
| f1@0.5 | 0.2892 | 0.0000 |
| brier | 0.0673 | 0.0775 |
| ROC_AUC | 0.7656 | 0.5000 |

**test**

| Metric | Model | Naive |
|---|---|---|
| n | 3,948.0000 | 3,948.0000 |
| positive_rate | 0.1031 | 0.1031 |
| PR_AUC | 0.2602 | 0.1031 |
| precision@0.5 | 0.4343 | 0.0000 |
| recall@0.5 | 0.1057 | 0.0000 |
| f1@0.5 | 0.1700 | 0.0000 |
| brier | 0.0890 | 0.0925 |
| ROC_AUC | 0.6785 | 0.5000 |

> PR-AUC against the positive rate is the number that matters; ROC-AUC flatters an imbalanced problem. Remember the population: NYC, Jan-Mar 2010, only the complaint types that carry a Due Date. Nothing here transfers to another city's SLA policy.


## hotspot

Trained on 47,694 rows in 1.1s. Naive baseline: future_incident_count = this week's incident_count (persistence).

**val**

| Metric | Model | Naive |
|---|---|---|
| n | 9,949.0000 | 9,949.0000 |
| MAE | 7.6157 | 7.8138 |
| median_AE | 3.9728 | 4.0000 |
| RMSE | 15.5428 | 15.3775 |
| RMSE_log1p | 0.7253 | 0.7943 |
| poisson_deviance | 6.4189 | 15.9098 |

**test**

| Metric | Model | Naive |
|---|---|---|
| n | 9,449.0000 | 9,449.0000 |
| MAE | 10.9247 | 12.8205 |
| median_AE | 4.7040 | 5.0000 |
| RMSE | 36.1530 | 39.3664 |
| RMSE_log1p | 0.6902 | 0.8182 |
| poisson_deviance | 14.8455 | 23.3906 |

> Persistence is a strong baseline on a weekly panel. A model that does not beat it is not yet worth deploying. Poisson deviance is the right loss-aligned metric for a count target; MAE is included because it is the one an operator can read.


## duplicate

Trained on 324,501 rows in 1.4s. Naive baseline: predict the TRAIN positive rate for every pair.

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
