# SLA breach model — sla_breach

Trained 2026-09-14T19:53:59.339025+00:00 · seed 20260915 · scikit-learn 1.6.1

**Dataset** `sla_ml` · **split column** `split` ({'train': 13209, 'test': 3873, 'val': 3131}) · **features** 17

**Selected configuration:** `A_default` (chosen on validation PR_AUC) · **threshold 0.1666** (max F1 on VALIDATION, frozen before test is scored)

> Accuracy is NOT reported as a headline. With a ~10% positive rate, always predicting 'no breach' scores ~90% accuracy and catches nothing. PR-AUC against the base rate is the number that means something.

## Metrics

| Fold | PR-AUC | ROC-AUC | precision | recall | F1 | Brier | positive rate |
|---|---|---|---|---|---|---|---|
| validation — model | 0.2786 | 0.7554 | 0.1989 | 0.4303 | 0.2720 | 0.0661 | 0.0779 |
| validation — baseline | 0.0779 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0730 | 0.0779 |
| **test — model** | 0.2489 | 0.6829 | 0.2204 | 0.3607 | 0.2736 | 0.0903 | 0.1038 |
| **test — baseline** | 0.1038 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0931 | 0.1038 |

Baseline: predict the TRAINING positive rate for every row (0.1120).

PR-AUC lift over the base rate on test: **x2.40**

## Confusion matrix (test, at the frozen threshold)

| | predicted no breach | predicted breach |
|---|---|---|
| **actual no breach** | 2,958 | 513 |
| **actual breach** | 257 | 145 |

## Calibration (test)

| Predicted probability | n | mean predicted | observed rate |
|---|---|---|---|
| [0.0,0.1) | 2,747 | 0.0342 | 0.0732 |
| [0.1,0.2) | 604 | 0.1431 | 0.1175 |
| [0.2,0.3) | 242 | 0.2374 | 0.1736 |
| [0.3,0.4) | 131 | 0.3451 | 0.2901 |
| [0.4,0.5) | 71 | 0.4525 | 0.1831 |
| [0.5,0.6) | 42 | 0.5483 | 0.3333 |
| [0.6,0.7) | 10 | 0.6472 | 0.3000 |
| [0.7,0.8) | 19 | 0.7537 | 0.7368 |
| [0.8,0.9) | 6 | 0.8286 | 1.0000 |
| [0.9,1.0) | 1 | 0.9337 | 0.0000 |

## Configurations tried

| Config | Question | val PR-AUC | val ROC-AUC | val F1 |
|---|---|---|---|---|
| **A_default** | is there any signal beyond the base rate? | 0.2786 | 0.7554 | 0.2720 |
| B_balanced | does correcting the ~10% class imbalance help? | 0.2632 | 0.7362 | 0.2593 |
| C_slow_regularised | does a slower, more regularised fit generalise better on 13k rows? | 0.2575 | 0.7452 | 0.2787 |

## Class balance

- train 0.1120 · val 0.0779 · test 0.1038
- handled by class_weight inside the fit where selected; the data is never resampled, so val and test keep the prevalence a deployment would actually see

## Features

`category`, `subcategory`, `department`, `zone_key`, `report_channel`, `hour`, `day_of_week`, `month`, `is_weekend`, `is_night`, `sla_target_hours`, `nearby_similar_incidents_24h`, `nearby_similar_incidents_7d`, `nearby_similar_incidents_30d`, `local_incident_density`, `category_incident_density`, `hours_since_previous_similar_incident`

## Caveat

> NYC only, January-March 2010, and only the complaint types that carry a Due Date. This is a research model for that slice, not a general SLA-risk model, and must never be presented as one.

## Notes

- split column: split — single-source table (source_dataset=nyc311); 'split_global' degenerates to an empty val/test
