# Hotspot model — future_incident_count

Trained 2026-09-14T19:54:03.739119+00:00 · seed 20260915 · scikit-learn 1.6.1

**Dataset** `hotspot_ml` · **split column** `split_global` ({'train': 47092, 'val': 10000, 'test': 10000}) · **features** 13

**Selected configuration:** `C_log1p_squared` (chosen on validation MAE)

> **VERDICT: the model LOSES to the rolling 4-week mean on validation (-6.87%) and beats it only on test (+5.24%) — not a stable improvement.**

> **Recommendation:** ship the rolling 4-week mean; the model has not demonstrated a real edge.

## Validation

| Predictor | MAE | RMSE | median_AE | poisson_deviance | R2 |
|---|---|---|---|---|---|
| **model** | 7.4571 | 14.2200 | 3.5195 | 7.2054 | 0.5499 |
| **rolling_4w_mean** | 6.9779 | 13.0125 | 3.5000 | 6.1391 | 0.6231 |
| rolling_4w_mean_lagged | 7.3226 | 13.6797 | 3.7500 | 6.6948 | 0.5834 |
| persistence | 7.9257 | 15.1287 | 4.0000 | 14.2341 | 0.4905 |

| Compared with | model MAE | baseline MAE | improvement | model wins |
|---|---|---|---|---|
| persistence | 7.457 | 7.926 | +5.91% | yes |
| rolling_4w_mean | 7.457 | 6.978 | -6.87% | **no** |
| rolling_4w_mean_lagged | 7.457 | 7.323 | -1.84% | **no** |

## Test

| Predictor | MAE | RMSE | median_AE | poisson_deviance | R2 |
|---|---|---|---|---|---|
| **model** | 10.3145 | 32.4134 | 4.3704 | 14.7785 | 0.3077 |
| **rolling_4w_mean** | 10.8850 | 31.6485 | 5.0000 | 14.4754 | 0.3400 |
| rolling_4w_mean_lagged | 11.7622 | 33.3829 | 5.2500 | 16.5558 | 0.2657 |
| persistence | 12.6231 | 37.1046 | 5.0000 | 23.8779 | 0.0928 |

| Compared with | model MAE | baseline MAE | improvement | model wins |
|---|---|---|---|---|
| persistence | 10.314 | 12.623 | +18.29% | yes |
| rolling_4w_mean | 10.314 | 10.885 | +5.24% | yes |
| rolling_4w_mean_lagged | 10.314 | 11.762 | +12.31% | yes |

## Baseline definitions

- `persistence` — future_incident_count = this week's incident_count
- `rolling_4w_mean` — mean weekly count over weeks t-3..t (PRIMARY benchmark)
- `rolling_4w_mean_lagged` — mean weekly count over weeks t-4..t-1

## Configurations tried

| Config | Question | val MAE | val RMSE | val Poisson dev. |
|---|---|---|---|---|
| A_poisson | the loss that matches a count target | 7.757 | 14.174 | 7.109 |
| B_poisson_regularised | the model loses to a 4-week mean despite having that mean as a feature — is it over-fitting the panel? | 7.730 | 14.247 | 6.775 |
| **C_log1p_squared** | does modelling log1p(count) behave better than a Poisson loss here? | 7.457 | 14.220 | 7.205 |

## Leakage

> Every predictor is a trailing quantity ending at or before week t, while the target is week t+1. The rolling baselines are computed on the full sorted panel so that history is not truncated at a fold boundary. tests/test_pipeline.py asserts the rolling baseline does not move when every future value is rewritten.

## Features

`city`, `zone_key`, `zone_type`, `category`, `month`, `year`, `week_of_year`, `incident_count`, `previous_period_count`, `rolling_4w_count`, `rolling_12w_count`, `rolling_4w_mean`, `trend_4w`

## Notes

- split column: split_global — multi-source table (3 sources) — pooled evaluation on one wall clock
