# UrbanEye+ — ML Scope

What is a machine-learning problem here, what is deliberately not, and why.

---

## The scope decision that drives everything

**The citizen selects the category in the app.** That single product fact removes
three models that an earlier research phase had identified:

| Removed model | Why |
|---|---|
| Image → category | The category is already known. Predicting it would be re-deriving an input. |
| Text → category | Same. |
| Image → severity | Severity is not shown in the citizen UI and has no public labels. Out of scope twice over. |

What remains is **tabular and geospatial**. No image dataset is needed, and none
is downloaded.

---

## Component map

| Component | Type | Status today |
|---|---|---|
| Category | **Citizen-provided** — not ML | Available |
| PostGIS geospatial enrichment | **Deterministic computation** — not ML | Interface defined; India layer not yet plugged in |
| Priority (baseline) | **Deterministic rule engine** — not ML | Implemented, explainable, config-driven |
| Priority (learned) | **ML** | **Blocked — no genuine labels exist** |
| Resolution time | **ML** | Conditionally ready — see `ML_READINESS.md` §2 |
| SLA risk | **ML** | Conditionally ready — NYC 2010 only, `ML_READINESS.md` §3 |
| Hotspot / incident risk | **ML or statistical** | Conditionally ready — beats persistence but **not** a 4-week moving average, `ML_READINESS.md` §4 |
| Duplicate detection | **ML or algorithmic** | Labels ready, **evaluation is not** — `ML_READINESS.md` §5 |
| SLA assignment from priority | **Business rule** — not ML | Config: P1 30h, P2 50h, P3 72h, P4 90h |
| Severity | **Not predicted** | Out of scope; NULL everywhere |

A recurring failure mode in projects like this is routing every decision through
a model because a dataset happens to exist. Three of the components above are
deterministic and should stay that way: PostGIS is geometry, SLA assignment is a
product decision, and the priority baseline needs to be explainable to a
municipal officer who will ask "why is this P1?".

---

## 1. Priority

### Baseline engine — implemented, not ML

A weighted score over category and geospatial context, defined entirely in
`config/priority_config.yaml`. Produces `priority_baseline`, `priority_score`,
`priority_reasons`, `priority_confidence`, `priority_features_available` and
`priority_feature_coverage`. Fully explainable, and the last two make the
confidence tier itself auditable rather than asserted.

On the current US corpus 36% of the policy's weight is evaluable (the category
term plus the two backward-looking density terms); the remaining 64% is POI,
road-network and area context, which no public US 311 dataset provides. Every
row says so in `priority_feature_coverage`.

### Learned model — blocked, and the reason matters

`priority_baseline` is a deterministic function of a YAML file. **Training a
model to predict it would produce a model that has learned the YAML** — high
accuracy, zero information, and no ability to outperform the rules it copied.

A learned priority model needs `priority_label`: a *genuine* operational
priority. No public 311 dataset contains one. I checked the column metadata of
all four directly; none has a priority, urgency or severity field.

So the pipeline prepares the **feature side** of that dataset and leaves the
target NULL:

```
priority_baseline     = P2          DERIVED, is_ground_truth = False
priority_label        = NULL        genuine label — none exists yet
priority_label_source = NULL
```

**Unblocking it:** log the priority that UrbanEye+ operators actually assign
(and override) in production, plus the outcome. After a few thousand real
Indian incidents with operator-assigned or outcome-validated priorities, you have
a legitimate target. Details in `PRIORITY_METHODOLOGY.md`.

---

## 2. Resolution time / SLA risk — ML, ready

| | |
|---|---|
| Targets | `resolution_time_hours` (regression), `sla_breach` (binary) |
| Features | category, subcategory, city, department, zone, report channel, time-of-report, backward-looking density |
| Data | All four cities; `sla_breach` only where a target timestamp exists (Boston, NYC) |
| Suggested start | Gradient boosting on `log1p(resolution_time_hours)` |

Two honest caveats the builder records in its `.meta.json`:

- **Instant-closure artifact.** 8.2% of the old target was a transactional write
  rather than a service duration (Chicago closes thousands of cases at exactly
  5-7 seconds; NYC's sub-minute population is entirely at 0 s). Those rows keep
  their status and closed_at and carry `resolution_instant_closure = True`, but
  their duration is NULL — it is below what the source timestamps can express.
- **Right-censoring.** Open cases have no duration and are excluded, which biases
  the sample toward faster resolutions. Survival analysis is the correct
  treatment if long-running cases matter.
- **`department` timing.** It is assigned at 311 intake, so it is known at report
  time. If your municipality assigns it later, drop it.

---

## 3. Hotspot / incident risk — ML or statistical, ready

| | |
|---|---|
| Unit | city × zone × category × ISO week |
| Targets | `future_incident_count` (t+1), `future_incident_flag` |
| Features | lagged counts, 4/12-week rolling sums, trend, seasonality |

Next-period count was chosen over a "risk score" because it is **directly
observable** — you can score it against reality with MAE. A risk target would
need a definition we would have to invent.

We are explicitly **not** building traffic or footfall prediction.

> **Zones do not transfer.** `zone_id` is a US ward / community area / analysis
> neighbourhood / community board. Model per city or include `city` as a feature;
> never pool zones across cities as if comparable. This dataset teaches temporal
> dynamics, not geography.

---

## 4. Duplicate detection — ML or algorithmic, ready

| | |
|---|---|
| Target | `same_incident` |
| Positives | Chicago `PARENT_SR_NUMBER` — a real municipal determination |
| Negatives | Four documented blocking strategies |
| Features | distance, time delta, category match, text similarity |

Negative sampling avoids the trap of trivially easy pairs. The hard negatives
(N2: same category, within 200 m, more than 90 days apart) are the ones that stop
the model degenerating into a proximity detector.

`image_similarity` is NULL: Chicago, the only source with duplicate labels,
publishes no photographs. There is no public data with which to supervise
image-based duplicate matching.

Recommended production shape: **blocking then scoring** — candidate generation by
category + radius + time window, then a classifier on the pair features.

---

## What can be trained today vs what cannot

**Can train now:** resolution time, SLA breach, hotspot risk, duplicate detection.

**Cannot train now:** priority. Blocked on `priority_label`, not on data volume,
feature engineering or compute. No amount of US 311 data unblocks it.

**Will never be trained here:** category (citizen-provided), severity (out of
scope), PostGIS enrichment (deterministic), SLA assignment (business rule).

---

## India vs US

The US 311 data supplies **behavioural** patterns: how categories relate to
resolution time, how incidents cluster in time, what a duplicate looks like.

It does **not** supply Indian geographic context. Every POI feature is NULL and
stays NULL until an Indian geospatial layer is connected. Raw latitude/longitude
is never used as a model feature — only semantic features derived from it. See
`GEOSPATIAL_FEATURES.md`.
