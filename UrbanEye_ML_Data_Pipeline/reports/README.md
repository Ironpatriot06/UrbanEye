# Reports

Empty until you run the pipeline. Generated here:

| File | Produced by |
|---|---|
| `data_quality.json` / `.md` | `validation/data_quality_report.py` |
| `provenance_report.json` | `validation/validate_provenance.py` |
| `leakage_report.json` | `validation/validate_leakage.py` |
| `output_validation.json` | `validation/validate_outputs.py` |
| `mapping_validation.json` | `validation/validate_mappings.py` |
| `dataset_status.md` / `.json` | `validation/dataset_status.py` |
| `priority_summary.json` | `preprocess/build_priority_features.py` |
| `time_features_summary.json` | `preprocess/build_time_features.py` |
| `geo_features_summary.json` | `preprocess/build_geo_features.py` |
| `ml_dataset_summary.json` | `features/build_ml_dataset.py` |
| `duplicate_pairs_summary.json` | `preprocess/build_duplicate_pairs.py` |
| `pipeline_audit.json` / `.md` | `validation/audit_pipeline.py` — **the stage-by-stage audit** |
| `task_dataset_audit.json` / `.md` | `validation/audit_task_datasets.py` — per-task feature/target audit + 12 leakage routes |
| `statistical_audit.json` / `.md` | `validation/audit_statistics.py` — imbalance, drift, degenerate columns, each classified |
| `category_mapping_candidates.csv` / `.md` / `.json` | `validation/report_category_candidates.py` — ranked proposals, **applied to nothing** |
| `task_datasets_index.json` | `features/build_task_datasets.py` |
| `baseline_models.json` / `.md` | `baselines/run_baselines.py` — optional, not part of the pipeline |
| `model_training/<task>_results.json` / `.md` | `train/train_<task>.py` — per-task training + evaluation |
| `model_training/training_summary.json` / `.md` | `train/run_all.py` — cross-task summary, including the tasks deliberately not trained |
| `unmapped_categories.csv` | the preprocessors — **your work queue for extending category_mapping.csv** |
| `pipeline_run.json` | `run_pipeline.py` |

For an example of what these look like, see `examples/fixture_run/reports/`
(synthetic data).
