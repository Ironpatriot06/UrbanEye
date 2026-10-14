"""
UrbanEye+ ML inference layer (Phase 2C).

What lives here
---------------
  registry.py   loads the serving bundle once per process, verifies it, reports readiness
  encoding.py   the exported FeaturePreprocessor (JSON) re-implemented
  areas.py      service areas: the "city" the history features are computed within
  features.py   leak-free history features, ported from the ML pipeline's
                scripts/phase2b/features.py and checked against its output
  predictors.py resolution-time and SLA-breach predictions, with explicit fallbacks
  hotspot.py    next-week counts by EWMA (alpha 0.3) — a statistic, NOT an ML model
  service.py    database access, orchestration and prediction logging

Ground rules
------------
* Nothing is trained here.  The boosters were trained offline (Phase 2B) on US
  311 data (Chicago, San Francisco, New York City) up to 2020-08-28, and are
  loaded read-only from a SHA-256-verified bundle.
* Every prediction is computed AS OF the incident's created_at, from other
  incidents reported before it and closures recorded before it, so it is
  reproducible and cannot see the incident's own outcome.
* Readiness (are the artifacts loaded?) and availability (can this incident be
  predicted, and how?) are reported separately.
* Every result names its method (ML_MODEL or a named baseline) and status:
    AVAILABLE     model output, every input inside the evaluated regime
    EXPERIMENTAL  model output, but local history is missing or out of range
    FALLBACK      a documented non-ML baseline, with the reason the model was not used
    UNAVAILABLE   neither a valid model result nor an applicable baseline
  `local_reliability_established` is False for all of them: no prediction has
  yet been evaluated against UrbanEye outcomes.
* Prediction failures never raise into incident workflows.
"""
