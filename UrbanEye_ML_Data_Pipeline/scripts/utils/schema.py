"""
Canonical UrbanEye+ schemas, with provenance attached to every field.

Provenance is enforced, not merely documented. `finalise_incidents()` refuses to
emit a frame in which a NULL-provenance field has been populated. That is the
mechanism that stops severity, occurred_at, response_time and priority_label
from quietly acquiring fabricated values three refactors from now.

The most important distinction in this file:

    priority_baseline   DERIVED by the configurable weighted engine.
                        Explainable. NOT ground truth.
    priority_label      A genuine operational priority assigned by a human or a
                        municipal system. NULL-provenance: no public 311 dataset
                        contains one, so populating it is a provenance violation.
"""
from __future__ import annotations

import pandas as pd

SOURCE = "SOURCE"    # verbatim or normalised from the publisher
DERIVED = "DERIVED"  # computed here from SOURCE fields
SYSTEM = "SYSTEM"    # produced later by the UrbanEye+ backend (India PostGIS)
NULL = "NULL"        # genuinely unavailable in every configured source

INCIDENT_SCHEMA: dict[str, tuple[str, str, str]] = {
    "incident_id":           ("string",  DERIVED, "f'{source_dataset}:{source_incident_id}' — globally unique"),
    "source_dataset":        ("string",  DERIVED, "which publisher this row came from"),
    "source_incident_id":    ("string",  SOURCE,  "CaseID / CASE_ENQUIRY_ID / SR_NUMBER / Unique Key"),
    "city":                  ("string",  SOURCE,  "publisher's city — never rewritten to an Indian city"),
    "category":              ("string",  DERIVED, "source category mapped via config/category_mapping.csv"),
    "subcategory":           ("string",  SOURCE,  "publisher's second-level label; NULL for Chicago"),
    "source_category_raw":   ("string",  SOURCE,  "retained so mapping decisions stay auditable"),
    "description":           ("string",  SOURCE,  "AGENCY-written text (resolution notes / case titles). NOT citizen text. Forbidden as a report-time feature."),
    "latitude":              ("float64", SOURCE,  "publisher WGS84. US coordinates stay US coordinates."),
    "longitude":             ("float64", SOURCE,  ""),
    "address":               ("string",  SOURCE,  ""),
    "zone_id":               ("string",  SOURCE,  "ward / community area / analysis neighbourhood / community board. A US admin unit, NOT an Indian zone."),
    "zone_type":             ("string",  DERIVED, "names which administrative unit zone_id refers to"),
    "reported_at":           ("datetime64[ns, UTC]", SOURCE, "when the citizen reported it"),
    "occurred_at":           ("datetime64[ns, UTC]", NULL, "NO configured source records when the problem began. Always NULL. Never derived from reported_at."),
    "closed_at":             ("datetime64[ns, UTC]", SOURCE, "POST-RESOLUTION. Never a feature."),
    "status":                ("string",  SOURCE,  "normalised open/closed/cancelled/other. POST-RESOLUTION."),
    "department":            ("string",  SOURCE,  "responsible agency / owner department"),

    "priority_baseline":     ("string",  DERIVED, "P1-P4 from the weighted engine in config/priority_config.yaml. NOT ground truth."),
    "priority_score":        ("float64", DERIVED, "normalised 0-1 score behind priority_baseline"),
    "priority_reasons":      ("string",  DERIVED, "pipe-separated explanation of the top contributing features"),
    "priority_confidence":   ("string",  DERIVED, "HIGH/MEDIUM/LOW — a deterministic tier over how much of the policy was evaluable, per priority_config.confidence_tiers. Not a probability."),
    "priority_features_available": ("Int64", DERIVED, "how many weighted inputs were non-null for this row"),
    "priority_feature_coverage": ("float64", DERIVED, "share of TOTAL policy weight those inputs carry, 0-1. Published so priority_confidence can always be audited."),
    "priority_method":       ("string",  DERIVED, "priority_config.yaml@<config_version>"),
    "priority_label":        ("string",  NULL,    "GENUINE operational priority. No public 311 dataset provides one. Always NULL here."),
    "priority_label_source": ("string",  NULL,    "provenance of a genuine label. Always NULL until real labels exist."),
    "is_ground_truth":       ("boolean", DERIVED, "always False for priority in this pipeline"),

    "sla_target_hours":      ("float64", DERIVED, "Boston TARGET_DT-OPEN_DT; NYC Due Date-Created Date; NULL for SF/Chicago"),
    "sla_met":               ("boolean", DERIVED, "resolution_time_hours <= sla_target_hours, only when both valid. POST-RESOLUTION."),
    "sla_hours_policy":      ("float64", DERIVED, "UrbanEye+ business-rule SLA for priority_baseline, from priority_config.sla_hours. Product config, not evidence."),

    "response_time_hours":   ("float64", NULL,    "NO configured source records a first-response timestamp. Always NULL."),
    "resolution_time_hours": ("float64", DERIVED, "closed_at - reported_at. NULL when still open. POST-RESOLUTION — target only, never a feature."),

    "is_duplicate":          ("boolean", SOURCE,  "Chicago DUPLICATE flag; weak text signal from SF Status Notes; NULL elsewhere"),
    "parent_incident_id":    ("string",  SOURCE,  "Chicago PARENT_SR_NUMBER only"),

    "report_channel":        ("string",  SOURCE,  "phone / web / mobile / social / other"),
    "image_url":             ("string",  SOURCE,  "SF Media URL or Boston SubmittedPhoto. METADATA ONLY — no image is ever downloaded."),
    "image_url_after":       ("string",  SOURCE,  "Boston ClosedPhoto only. Metadata only."),

    "severity":              ("Int64",   NULL,    "Out of current product scope AND unavailable. No public source provides a defensible grade. Always NULL."),

    "coord_outside_city_bbox": ("boolean", DERIVED, "QA flag, not a filter"),
}

NULL_FIELDS = [k for k, v in INCIDENT_SCHEMA.items() if v[1] == NULL]

POST_RESOLUTION_FIELDS = [
    "closed_at", "status", "resolution_time_hours", "sla_met", "description",
]


def geo_feature_schema() -> dict[str, tuple[str, str, str]]:
    """
    Built from config/feature_config.yaml so the schema and the contract with the
    PostGIS layer can never drift apart.
    """
    from .paths import load_feature_config
    fc = load_feature_config()
    schema: dict[str, tuple[str, str, str]] = {
        "incident_id": ("string", DERIVED, "join key to incidents"),
    }
    for block in ("poi_proximity_features", "road_network_features",
                  "area_context_features", "historical_density_features",
                  "historical_recency_features"):
        for name, spec in fc.get(block, {}).items():
            dtype = spec.get("dtype", "float64")
            avail = spec.get("available_from_public_us_data", False)
            provider = spec.get("provider", "POSTGIS")
            if avail:
                prov = DERIVED
                note = spec.get("note", "") or f"computed from 311 history ({provider})"
            else:
                prov = SYSTEM
                note = ("REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. "
                        "Populated at runtime by the India PostGIS layer.")
            schema[name] = (dtype, prov, note)
    schema["geo_provider"] = ("string", DERIVED, "which provider produced this row, e.g. 'null_provider' or 'india_postgis'")
    schema["geo_features_available"] = ("Int64", DERIVED, "count of non-null geospatial features")
    return schema


def time_feature_schema() -> dict[str, tuple[str, str, str]]:
    """
    The report-time temporal block, built from config/feature_config.yaml.

    Documentation only — the values are produced (and typed) by
    scripts/utils/timefeatures.py. `conform()` is deliberately not applied to
    this table: reported_at_local is a tz-NAIVE wall-clock reading in the city's
    own timezone, and conform would localise it to UTC and destroy that meaning.
    """
    from .paths import load_feature_config
    fc = load_feature_config()
    pol = fc.get("temporal_policy", {}) or {}
    tzsrc = pol.get("timezone_source", "dataset_config.yaml")
    schema: dict[str, tuple[str, str, str]] = {
        "incident_id": ("string", DERIVED, "join key to incidents"),
        "reported_at_local": ("datetime64[ns]", DERIVED,
                              f"reported_at converted to the source city's timezone ({tzsrc}). "
                              "tz-naive wall clock, NOT UTC."),
        "local_timezone": ("string", DERIVED, "IANA timezone used for the conversion"),
    }
    for name, spec in fc.get("temporal_features", {}).items():
        schema[name] = (spec.get("dtype", "Int64"), DERIVED, spec.get("note", ""))
    schema["date_local"] = ("string", DERIVED, "local calendar date, for grouping")
    return schema


DUPLICATE_PAIR_SCHEMA: dict[str, tuple[str, str, str]] = {
    "incident_a":            ("string",  SOURCE,  "parent incident_id"),
    "incident_b":            ("string",  SOURCE,  "child incident_id"),
    "same_incident":         ("boolean", SOURCE,  "TRUE from Chicago PARENT_SR_NUMBER; FALSE from documented negative sampling"),
    "distance_meters":       ("float64", DERIVED, "haversine; NULL if either coordinate is missing"),
    "time_difference_hours": ("float64", DERIVED, ""),
    "category_match":        ("boolean", DERIVED, ""),
    "text_similarity":       ("float64", DERIVED, "NULL for Chicago — Chicago publishes no free-text description"),
    "source_dataset":        ("string",  DERIVED, ""),
    "image_similarity":      ("float64", NULL,    "NULL. The labelled source (Chicago) has no photographs."),
    "negative_strategy":     ("string",  DERIVED, "which negative-sampling rule produced a FALSE pair; NULL for positives"),
    "split":                 ("string",  DERIVED, "group-aware; a duplicate group never spans splits"),
}


def empty_frame(schema: dict[str, tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, (t, _, _) in schema.items()})


def conform(df: pd.DataFrame, schema: dict[str, tuple[str, str, str]]) -> pd.DataFrame:
    """Add missing columns as nulls, coerce dtypes, order columns to the schema."""
    df = df.copy()
    for col, (dtype, _, _) in schema.items():
        if col not in df.columns:
            df[col] = pd.Series([pd.NA] * len(df), dtype="object")
        try:
            if dtype.startswith("datetime64"):
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            else:
                df[col] = df[col].astype(dtype)
        except (TypeError, ValueError):
            pass
    return df[list(schema.keys())]


class ProvenanceViolation(RuntimeError):
    pass


def assert_null_fields_empty(df: pd.DataFrame, schema: dict[str, tuple[str, str, str]]) -> None:
    """Any field declared NULL-provenance must be entirely empty."""
    offenders = []
    for col, (_, prov, note) in schema.items():
        if prov != NULL or col not in df.columns:
            continue
        n = int(df[col].notna().sum())
        if n:
            offenders.append(f"{col} has {n:,} non-null values — {note}")
    if offenders:
        raise ProvenanceViolation(
            "Fields declared NULL-provenance were populated:\n  - " + "\n  - ".join(offenders)
        )


def finalise_incidents(df: pd.DataFrame) -> pd.DataFrame:
    df = conform(df, INCIDENT_SCHEMA)
    assert_null_fields_empty(df, INCIDENT_SCHEMA)
    return df


def provenance_table(schema: dict[str, tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"field": k, "dtype": v[0], "provenance": v[1], "note": v[2]} for k, v in schema.items()]
    )
