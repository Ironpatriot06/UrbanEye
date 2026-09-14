# UrbanEye+ — Priority Methodology

How priority is produced today, why it is not yet a learned model, and what has
to happen before it can be.

---

## 1. The production flow

```
CITIZEN
  category, description, latitude, longitude, timestamp, optional image
        |
        v
POSTGIS / GEO FEATURE ENGINE          <- deterministic, not ML
  near_school + distance_to_nearest_school_m
  near_hospital, near_metro, near_major_road, near_intersection
  metro_city, urban_area, road_class
  nearby_similar_incidents_24h / 7d / 30d, local_incident_density
        |
        v
PRIORITY
  (a) BASELINE RULE ENGINE  -> P1..P4 + score + reasons   <- available now
  (b) ML PRIORITY MODEL     -> P1..P4                     <- blocked, no labels
        |
        v
SLA BUSINESS RULE                     <- configuration, not ML
  P1 -> 30h   P2 -> 50h   P3 -> 72h   P4 -> 90h
        |
        v
RESOLUTION observed -> feeds the resolution-time / SLA-risk model
```

The citizen never supplies proximity or road-class information. They report what
they see; the system derives the context.

---

## 2. The baseline engine

Everything lives in `config/priority_config.yaml`. Nothing is hard-coded in
Python.

### Scoring

```
score = sum(weight_f * signal_f  for f in AVAILABLE features)
        / sum(weight_f           for f in AVAILABLE features)
```

Signals are normalised to `[0, 1]`:

- **Proximity** — graded, not binary: `signal = clip(1 - distance/threshold, 0, 1)`.
  An incident 20 m from a school scores higher than one 190 m away, even though
  both are "near a school" under a 200 m threshold.
- **Counts** — `signal = clip(value / saturation, 0, 1)`.
- **Booleans** — 1.0 or 0.0.
- **Category** — a hand-authored base score, 0.10 (graffiti) to 1.00 (open manhole).

Score bands map to `P1 >= 0.75`, `P2 >= 0.55`, `P3 >= 0.30`, `P4` otherwise.

### The one absolute rule

`OPEN_MANHOLE` goes straight to P1 regardless of context. An uncovered shaft in a
walking or driving surface is an immediate fall hazard whether or not a school
happens to be nearby.

### Missing features are not zeros

`null_handling.mode: strict_available_only`. The denominator only includes
features that were actually present.

This matters more than it sounds. "We don't know whether a school is nearby" and
"there is definitely no school nearby" are different statements, and only the
second should lower a priority. Treating NULL as `false` would systematically
under-prioritise every incident in an area with incomplete POI coverage.

### Confidence is a coverage statement, not a probability

Each row publishes two auditable numbers:

| Column | Meaning |
|---|---|
| `priority_features_available` | how many of the 14 weighted inputs were non-null |
| `priority_feature_coverage` | the share of TOTAL policy weight those inputs carry |

`priority_confidence` is a deterministic tier over those two numbers, defined in
`priority_config.yaml -> confidence_tiers` and evaluated top-down:

| Tier | Requires | On the US corpus |
|---|---|---|
| HIGH | >= 8 inputs and >= 70% of policy weight | unreachable — needs the POI layer |
| MEDIUM | >= 3 inputs and >= 30% of policy weight | every row with a usable coordinate (category + the two density terms = 36% of weight) |
| LOW | anything less | rows with no usable coordinate |

A row hit by an absolute category rule reports HIGH with coverage 1.0: no other
input could change the outcome, so nothing is missing from it.

Nothing about the tier is inferred from the incident. The same inputs always
produce the same tier, and both inputs to the decision travel with the row so a
reader never has to trust the label.

### Worked examples

Verified by running `build_priority_features.py --explain`:

| Input | Result |
|---|---|
| `POTHOLE` + school 50 m + metro 300 m + major road 15 m + intersection 12 m + metro city + 37 nearby in 30d | **P2**, score 0.623, confidence MEDIUM (7/14 inputs, 73% of weight) |
| `POTHOLE` + school 1800 m + metro 4000 m + major road 900 m + intersection 600 m + 1 nearby | **P4**, score 0.241 |
| `OPEN_MANHOLE`, no context at all | **P1**, score 1.0, absolute rule |

Same category, opposite ends of the scale. Priority is a property of the
situation, not of the category label.

### Explanation

Every row carries `priority_reasons`, e.g.:

```
category POTHOLE (0.17) | near a school (0.09) | on or beside a major road (0.07)
| near an intersection (0.05) | in a metropolitan city (0.04)
```

`explanation.expose_to_citizen: false`, `expose_to_admin: true`. The citizen sees
**Priority: P2**. An admin can see why.

---

## 3. Why priority is not an ML target yet

`priority_baseline` is a deterministic function of a YAML file. A model trained
on it learns the YAML. It would score well and know nothing that the rules did
not already encode, and it could never outperform them.

A learned model needs `priority_label` — a genuine operational priority. **No
public 311 dataset contains one.** I inspected the live column metadata for SF,
Boston, Chicago and NYC: none has a priority, urgency, severity or risk field.

The pipeline therefore keeps three fields strictly apart:

| Field | Provenance | Value today |
|---|---|---|
| `priority_baseline` | DERIVED | P1–P4 from the engine |
| `priority_label` | **NULL** | empty — enforced |
| `priority_label_source` | **NULL** | empty — enforced |
| `is_ground_truth` | DERIVED | `False`, always |

This is enforced, not documented. `validate_provenance.py` check **ACTIVE-5**
attempts to copy `priority_baseline` into `priority_label` and asserts the guard
rejects it — because that is exactly the shortcut someone would take to
manufacture a training target.

### What SLA evidence does and does not justify

Boston publishes `TARGET_DT` and NYC publishes `Due Date`. The pipeline derives
`sla_target_hours` from both. This is **reference evidence about how those cities
behave**.

It does **not** justify UrbanEye+'s own SLA values, and
`priority_config.yaml` sets `sla_reference.use_for_priority_assignment: false`
for a reason: a US city's SLA window encodes its own staffing and budget.
Importing it as an Indian priority would launder one municipality's operational
constraints into another's safety policy.

The SLA hours (P1 30h, P2 50h, P3 72h, P4 90h) are **product configuration** you
supplied. They are stored in config and changeable without touching code.

---

## 4. The config ships as DRAFT

`config/priority_config.yaml` has `status: DRAFT` and `approved_by: null`.
`build_priority_features.py` **refuses to run** without `--accept-draft-config`
(exit code 3).

This is deliberate friction. The weights are a starting template, not evidence.
Before they become policy someone must decide:

1. Which categories carry `public_safety_flag` in Indian cities.
2. What "near" means in metres for Indian urban density — 200 m from a school is
   a very different exposure in Chennai than in suburban Boston.
3. Whether severity is required for P1, or whether category alone can trigger it.
4. Whether `nearby_similar_incidents` should escalate (persistent problem) or
   de-escalate (noisy neighbourhood). The template escalates.
5. Whether streetlight faults should escalate at night — currently they do not,
   and the personal-safety dimension is uncaptured.

When settled, set `status: APPROVED` with `approved_by` and `approved_at`.

---

## 5. Behaviour on the US data

Every POI, road-network and area-context input is NULL, because no public US 311
dataset contains one and this repository ships no reference geodata. What IS
available is the category term and the two backward-looking density terms, so on
the current corpus:

- the score is driven by category **plus local repeat-report pressure**
- `priority_confidence` is MEDIUM for the 1,870,394 rows with a usable
  coordinate and LOW for the 29,606 without one
- 36% of the policy's total weight is evaluable; the other 64% waits on PostGIS

That is meaningfully better than a category lookup — `audit_pipeline.py` PR-7
asserts the score varies *within* a category — but it is still a long way from
the policy as designed. `reports/priority_summary.json` and
`reports/pipeline_audit.json` state the coverage explicitly so nobody reads the
distribution as a finished result.

**Earlier versions of this pipeline reported LOW for 100% of rows.** That was
not a policy statement: `build_geo_features.py` had never been run (its density
loop did not finish on 1.9M rows), so the density columns were absent and the
score genuinely did collapse to the category term. The builder is now vectorised
and the features exist.

### Using the score as a ranking

The score is normalised over the weights of the features that were actually
available, so **two rows scored on different input sets are not on the same
scale**:

| Inputs available | Rows | Score range |
|---|---|---|
| 3 of 14 — category + both density terms | 1,870,394 | 0.083 – 0.875 |
| 1 of 14 — category only (no usable coordinate) | 29,606 | 0.100 – 0.850 |

Rank within a stratum, or display `priority_feature_coverage` beside the score.
One ordered queue mixing the two compares numbers computed differently.

`priority_features.parquet` also carries an `in_scope` flag: 864,956 of the
1.9M scored rows are UNMAPPED or OUT_OF_SCOPE and are scored with the `OTHER`
base score. They are retained for auditability and are not UrbanEye incidents —
filter on `in_scope` before using the ranking operationally.

### Candidate inputs a policy review should consider

`hours_since_previous_similar_incident` is now computed and is not yet a policy
input. Adding it would let the engine distinguish "five reports this month" from
"five reports in the last hour", which is a genuine urgency signal. Whether it
should escalate, and by how much, is question 4 in the list above — a policy
decision, so the pipeline computes the feature and leaves the weight alone.

---

## 6. Unblocking the ML priority model

Log what operators actually do:

```
incident_id, category, reported_at, latitude, longitude,
priority_assigned_by_operator,      <- the genuine label
priority_baseline_at_time,          <- what the engine suggested
operator_overrode,                  <- boolean: the most informative column
override_reason,
resolution_time_hours, sla_met
```

**The override flag is the valuable part.** Rows where an operator disagreed with
the baseline are precisely where the rules are wrong, and they are the training
signal that lets a model beat the rules rather than imitate them.

A few thousand such rows gives you a legitimate target. Then:

- set `priority_label` and `priority_label_source = OPERATOR_ASSIGNED`
- the provenance guard's NULL rule for these fields must be lifted **explicitly**,
  by moving them from `NULL` to `SOURCE` in `scripts/utils/schema.py`, with a
  comment recording where the labels came from
- train, and evaluate against the baseline as the benchmark to beat

Until then, the baseline engine is the production path — and being explainable,
it is defensible to a municipal officer in a way a model would not be.
