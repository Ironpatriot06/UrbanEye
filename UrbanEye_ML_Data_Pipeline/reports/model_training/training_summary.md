# Model training summary

Generated 2026-09-14T19:54:03.859581+00:00 · seed 20260915 · scikit-learn 1.6.1 · 43.6s total

## Trained tasks

| Task | Target | Split | train/val/test | Headline (test) | Baseline (test) |
|---|---|---|---|---|---|
| `resolution` | `resolution_time_hours` | `split_global` | 659,627/141,349/141,349 | MAE 896.5 · median AE 133.3 | MAE 1,034.8 · median AE 115.5 |
| `sla` | `sla_breach` | `split` | 13,209/3,131/3,873 | PR-AUC 0.2489 · R 0.361 | base rate 0.1038 (lift x2.40) |
| `hotspot` | `future_incident_count` | `split_global` | 47,092/10,000/10,000 | MAE 10.314 | rolling 4w 10.885 · persistence 12.623 |

## Verdicts

- **resolution** — the model beats the median baseline on MAE but LOSES on median absolute error — it wins on the tail and loses in the middle.
- **sla** — PR-AUC 0.2489 against a base rate of 0.1038, a lift of x2.40. Real signal on a narrow population.
- **hotspot** — the model LOSES to the rolling 4-week mean on validation (-6.87%) and beats it only on test (+5.24%) — not a stable improvement. _Ship the rolling 4-week mean; the model has not demonstrated a real edge._

## Not trained, and why

- **duplicate** — NOT READY — the labels are genuine but the evaluation set is not. In the realistic candidate population (same category, <=200 m, <=7 days) the test fold contains 23,480 positives and zero negatives, and a depth-2 decision tree scores PR-AUC 0.967. Any headline metric measures the negative-sampling rule. Needs adjudicated hard negatives from real operations.
- **priority** — NOT READY and NOT A SUPERVISED TASK — no public 311 dataset records an operational priority. priority_baseline is a deterministic function of config/priority_config.yaml, so training on it would reproduce the YAML. Needs operator-assigned priorities and the override flag.

## Artifacts

| Task | Model | Preprocessor | Metadata |
|---|---|---|---|
| `resolution` | `/Users/ratishkapoor/Desktop/UrbanEye/UrbanEye_ML_Data_Pipeline/models/resolution/model.joblib` | `/Users/ratishkapoor/Desktop/UrbanEye/UrbanEye_ML_Data_Pipeline/models/resolution/preprocessor.joblib` | `/Users/ratishkapoor/Desktop/UrbanEye/UrbanEye_ML_Data_Pipeline/models/resolution/metadata.json` |
| `sla` | `/Users/ratishkapoor/Desktop/UrbanEye/UrbanEye_ML_Data_Pipeline/models/sla/model.joblib` | `/Users/ratishkapoor/Desktop/UrbanEye/UrbanEye_ML_Data_Pipeline/models/sla/preprocessor.joblib` | `/Users/ratishkapoor/Desktop/UrbanEye/UrbanEye_ML_Data_Pipeline/models/sla/metadata.json` |
| `hotspot` | `/Users/ratishkapoor/Desktop/UrbanEye/UrbanEye_ML_Data_Pipeline/models/hotspot/model.joblib` | `/Users/ratishkapoor/Desktop/UrbanEye/UrbanEye_ML_Data_Pipeline/models/hotspot/preprocessor.joblib` | `/Users/ratishkapoor/Desktop/UrbanEye/UrbanEye_ML_Data_Pipeline/models/hotspot/metadata.json` |

Model artifacts are generated files and are excluded from version control by `.gitignore`, in line with the repository's existing policy for `data/`.
