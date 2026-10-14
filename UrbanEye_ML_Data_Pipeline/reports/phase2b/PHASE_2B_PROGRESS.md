# Phase 2B — progress checkpoint

**STATUS (2026-10-13): COMPLETE.** All "Remaining" items below are done. See `PHASE_2B_REPORT.md`
(final report), `final_gates.json`, `error_analysis.{json,md}`, `models/{resolution,sla}_v2/`,
`models/hotspot_v2/metadata.json`. Tests: 111/111. One correction to the findings below: on panel C
the hotspot GBM beats the trained EWMA by ~4% in 5/5 folds (not "≈"); the EWMA decision rests on the
37–52% transfer loss.

Paths are relative to `UrbanEye_ML_Data_Pipeline/`. No backend, frontend, DB schema or Edge AI
changes have been made. Original v1 artifacts (`models/{resolution,sla,hotspot}/`,
`reports/model_training/`) are untouched. `scripts/phase2b/verify_v1_reproducible.py` confirms all
three still reproduce their Phase 2A test metrics exactly (`reports/phase2b/v1_reproduction.json`).

## Done

- Phase 2A report recovered from an old scratchpad and saved to `reports/phase2a/`.
- **Code** (`scripts/phase2b/`):
  - `common.py`: paths, environment capture, ExperimentLog, UrbanEye policy constants, mapping, metrics, bootstrap.
  - `features.py`: leak-free workload, recent-resolution, recent-breach-rate and spatial-grid features.
  - `harness.py`: rolling-origin folds with label availability, LOCO folds, lookup baselines.
  - `build_datasets.py`, `domain_alignment.py`, `audit_urbaneye_db.py`.
  - `resolution_experiments.py`, `sla_experiments.py`, `hotspot_experiments.py`.
  - `inference.py`: contracts. Resolution and SLA predictors load `models/<task>_v2`. The hotspot predictor is pure EWMA.
  - `verify_v1_reproducible.py`.
- **Config**: `config/urbaneye_category_mapping.csv`. It maps raw publisher categories to UrbanEye categories with EXACT, STRONG or PARTIAL confidence. FIRE_HAZARD and OTHER are unsupported. Garbage Cart Maintenance, hydrants and basement water are explicitly UNSUPPORTED.
- **Data** (gitignored, rebuild in about 1 minute with `build_datasets.py`), in `data/processed/phase2b/`:
  - `incident_features_v2.parquet` (1.9M rows)
  - `resolution_v2.parquet` (725,624 rows)
  - `sla_v2.parquet` (272,007 rows; breach rate 51% under UrbanEye SLAs)
  - hotspot panels A and C
- **Tests**: `tests/test_phase2b.py` has 14 passing tests (brute-force checks, future-perturbation leakage checks, mapping, folds, backend-policy drift). The original 90 tests passed at the start.
- **Reports so far**: `reports/phase2b/urbaneye_db_audit.json`, `domain_alignment.{json,md}`, `hotspot_track_a.json`, `experiment_log_*.json`, `v1_reproduction.json`.

## Key findings so far

- **Data**:
  - The Chicago artifact extends through February 2019, so the clean cut is 2019-03-01. For hotspot, use 2019-05-27 so that 12-week windows are clean.
  - Zombie "open" records required a 90-day cap on the backlog.
- **UrbanEye DB**:
  - 6 test incidents and 1 resolved, after 0.15 s. It is not trainable.
  - An SLA of 12 h exists, so admins override SLAs.
  - Coordinates are in Bengaluru and Chennai.
- **Domain**:
  - The same category resolves at very different speeds by city (STREETLIGHT median: Chicago 554 h, NYC 42 h).
  - In LOCO, category medians transfer worse than a single global median.
- **Resolution** (rolling origin, 6 folds, UE rows, hours MAE):
  - Category median: 592.9.
  - Best deployable ML, F4b (workload + spatial, no month, l1-log): **581.7**. That is about 2% better, ahead in 5 of 6 folds, with better median AE and log error.
  - The Phase 2A recipe, which is not deployable, scores 574.9.
  - The 250 m density features add nothing.
- **SLA** (UrbanEye target):
  - The best baseline is the recent violation rate: PR-AUC 0.849, Brier 0.161.
  - S4 (hist + workload + spatial) scores **PR-AUC 0.879, Brier 0.152**. It wins PR-AUC in 6 of 6 folds.
  - Workload alone hurts; spatial helps. S4b and S7 (multi-horizon + spatial) are being evaluated.
- **Hotspot**:
  - Within Chicago, ML (absolute loss) beats the trained EWMA by 2.4% MAE but has worse deviance.
  - Transfer from Chicago to SF: ML is **37% worse** than EWMA.
  - On the deployable 1 km grid, ML ≈ EWMA.
  - The existing v1 model is not significantly better than EWMA.
  - **Decision: EWMA (non-ML) proceeds; no hotspot ML artifact is saved.**

## Remaining

1. Finish the experiment stages. Each run is cached and skips if already recorded.
   - `python scripts/phase2b/resolution_experiments.py --stage {models,loco,label_policy,tune,intervals,track_a}`
   - `python scripts/phase2b/sla_experiments.py --stage {screen,models,loco}`
2. Write `scripts/phase2b/train_final.py` with explicit pass/fail gates:
   - **Resolution**: beats the best baseline on mean MAE, wins at least 4 of 6 folds, median AE not worse, and the pooled bootstrap CI excludes 0. Train point, q10 and q90 models. Save `models/resolution_v2/` with the metadata listed in the prompt.
   - **SLA**: PR-AUC and Brier both beat the recent violation rate. Prefer S7 (multi-horizon) if it ties S4, because it supports SLA overrides. Choose the threshold on the 2020-05 fold and test it on 2020-07. Save `models/sla_v2/` with `decision_threshold` and `risk_bands`.
   - **Hotspot**: `models/hotspot_v2/metadata.json` only. Set `ewma_alpha` (tuned on panel C) and `min_history_weeks` = 4.
3. Add inference-contract tests to `tests/test_phase2b.py` (skip if the artifacts are absent). Run the full ML test suite.
4. Run error analysis: by duration bin, by category, by fold, plus permutation importance for the final models.
5. Write `reports/phase2b/PHASE_2B_REPORT.md` using the 15-section structure from the prompt, and answer the 7 final questions.
6. Re-run the checksum check. The before-snapshot was taken at session start (147 files). Then confirm that `git diff` touches nothing outside `UrbanEye_ML_Data_Pipeline/`.
