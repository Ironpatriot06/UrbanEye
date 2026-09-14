"""
Shared helpers for the ML-ready dataset builders.

Scope note: vision, severity and text->category builders were removed when the
ML scope narrowed (the citizen supplies the category, so there is nothing to
predict). Their code is preserved under optional_future_vision/ for reference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging_setup import get_logger
from ..utils.mapping import CANONICAL
from ..utils.paths import ensure_dir, load_config, p

log = get_logger("preprocess.ml")


def load_incidents(explicit: str | None = None) -> pd.DataFrame:
    """Prefer the prioritised table; fall back to the plain union."""
    if explicit:
        fp = Path(explicit)
    else:
        fp = p("processed", "incidents", "all_incidents_prioritised.parquet")
        if not fp.exists():
            fp = p("processed", "incidents", "all_incidents.parquet")
    if not fp.exists():
        raise SystemExit(f"No incidents table found at {fp}. Run build_incidents.py first.")
    df = pd.read_parquet(fp)
    log.info("loaded %d incidents from %s", len(df), fp.name)
    return df


def in_scope(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only real UrbanEye+ categories.

    UNMAPPED / REVIEW_REQUIRED / OUT_OF_SCOPE rows stay in `incidents` so the
    counts remain auditable, but they never reach a model-ready table.
    """
    keep = df[df["category"].isin(CANONICAL)].copy()
    log.info("in-scope: %d of %d rows (%d excluded as UNMAPPED/REVIEW_REQUIRED/OUT_OF_SCOPE)",
             len(keep), len(df), len(df) - len(keep))
    return keep


def _distinct_value_cuts(ts: pd.Series, q1: float, q2: float):
    """
    Cut on DISTINCT timestamps rather than on rows.

    Row quantiles assume a fine-grained time axis. On a coarse one — the weekly
    hotspot panel, where a whole city has ~10 distinct week_start values — the
    0.85 row quantile can land on the last week, leaving the test split empty.
    (Measured: NYC's hotspot panel got 5,061 train / 723 val / 0 test.) Cutting
    on the sorted distinct values and clamping the indices guarantees at least
    one period in each of train, val and test whenever three periods exist.
    """
    uniq = np.sort(pd.Series(ts.dropna().unique()))
    if len(uniq) < 3:
        return None
    i1 = int(round(q1 * len(uniq))) - 1
    i2 = int(round(q2 * len(uniq))) - 1
    i1 = min(max(i1, 0), len(uniq) - 3)
    i2 = min(max(i2, i1 + 1), len(uniq) - 2)
    return uniq[i1], uniq[i2]


def _cut_points(ts: pd.Series, sp: dict) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """
    Choose the two chronological cut timestamps for one timeline.

    Absolute cuts are preferred (they are stable across re-runs and across
    corpora), but only if they actually fall inside this timeline. A cut date
    beyond the last incident silently produces an empty val and test set — the
    exact defect this replaces — so when that happens we fall back to quantile
    cuts on the data's own timeline and SAY SO in the returned mode string.
    """
    ts = ts.dropna()
    mode = str(sp.get("cut_selection", "auto")).lower()
    fr = sp.get("fractions", {"train": 0.70, "val": 0.15, "test": 0.15})
    minrows = int(sp.get("min_rows_per_split", 1))
    q1 = float(fr.get("train", 0.70))
    q2 = q1 + float(fr.get("val", 0.15))

    def sizes(a, b) -> tuple[int, int, int]:
        return (int((ts <= a).sum()), int(((ts > a) & (ts <= b)).sum()), int((ts > b).sum()))

    def quantile_cuts(label: str):
        a, b = ts.quantile(q1), ts.quantile(q2)
        n_tr, n_v, n_te = sizes(a, b)
        if min(n_tr, n_v, n_te) >= minrows:
            return a, b, label
        # coarse/discrete timeline: fall back to cutting on distinct values
        alt = _distinct_value_cuts(ts, q1, q2)
        if alt is not None:
            a2, b2 = alt
            if min(sizes(a2, b2)) >= minrows:
                return a2, b2, label + "_distinct"
        return a, b, label

    if mode == "quantile" or ts.empty:
        return quantile_cuts("quantile")

    te = pd.Timestamp(sp["train_end"], tz="UTC")
    ve = pd.Timestamp(sp["val_end"], tz="UTC")
    if getattr(ts.dtype, "tz", None) is None:
        te, ve = te.tz_localize(None), ve.tz_localize(None)
    n_train, n_val, n_test = sizes(te, ve)
    if mode == "absolute" or min(n_train, n_val, n_test) >= minrows:
        return te, ve, "absolute"

    return quantile_cuts("quantile_fallback")


def chronological_split(df: pd.DataFrame, ts_col: str = "reported_at",
                        source_col: str = "source_dataset",
                        per_source: bool | None = None) -> tuple[pd.Series, dict]:
    """
    Leakage-resistant chronological split. Returns (split_series, metadata).

    TWO POLICIES, AND THE RIGHT ONE DEPENDS ON HOW YOU MODEL
    -------------------------------------------------------
    per_source=True  (column `split`, the default from splits.tabular.per_source)
        Each source_dataset is cut on its OWN timeline. Within a city, train
        strictly precedes val strictly precedes test. Every city appears in
        every fold, so `city` stays evaluable. This is the correct policy for a
        PER-CITY model.

        It is NOT globally chronological. The three cities occupy disjoint
        windows (NYC Jan-Mar 2010, SF Jan-May 2018, Chicago Jul 2018-Aug 2020),
        so on one wall clock 19.5% of resolution test rows — every SF and NYC
        test row — fall before the last training row. For a POOLED model that
        means being scored partly on the past.

    per_source=False (column `split_global`)
        One cut across the whole corpus. Every evaluation row is strictly after
        every training row on a single wall clock, which is the property a
        POOLED model needs. The price is real and must be reported: because SF
        and NYC end years before Chicago, validation and test contain Chicago
        and nothing else.

    Neither is universally right, so the pipeline writes both and each task
    manifest records which column to use. Rows with an unusable timestamp go to
    `unassigned`, never silently to train.
    """
    sp = load_config()["splits"]["tabular"]
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    out = pd.Series("unassigned", index=df.index, dtype="string")

    if per_source is None:
        per_source = bool(sp.get("per_source", False))
    per_source = bool(per_source) and source_col in df.columns
    if per_source:
        groups = {str(k): idx for k, idx in df.groupby(df[source_col].astype("string"),
                                                       dropna=False).groups.items()}
    else:
        groups = {"ALL": df.index}

    meta: dict = {"strategy": "chronological", "per_source": per_source,
                  "ts_column": ts_col, "groups": {}}
    for name, idx in groups.items():
        g = ts.loc[idx]
        a, b, mode = _cut_points(g, sp)
        s = pd.Series("test", index=idx, dtype="string")
        s[(g <= a).reindex(idx, fill_value=False)] = "train"
        s[((g > a) & (g <= b)).reindex(idx, fill_value=False)] = "val"
        s[g.isna().reindex(idx, fill_value=True)] = "unassigned"
        out.loc[idx] = s
        meta["groups"][name] = {
            "cut_mode": mode,
            "train_end": str(a), "val_end": str(b),
            "counts": {k: int(v) for k, v in s.value_counts().items()},
            "date_range": {"min": str(g.min()), "max": str(g.max())},
        }
    meta["counts"] = {k: int(v) for k, v in out.value_counts().items()}
    meta["globally_chronological"] = not per_source
    return out, meta


def dual_chronological_split(df: pd.DataFrame, ts_col: str = "reported_at",
                             source_col: str = "source_dataset") -> tuple[pd.Series, pd.Series, dict]:
    """
    Both split policies at once: (per_source_split, global_split, metadata).

    Writing both is deliberate. A per-city model needs every city present in
    every fold; a pooled model needs a single wall clock. Publishing one column
    and calling it "the" split forces every consumer into whichever question the
    pipeline happened to answer.
    """
    per_src, meta_src = chronological_split(df, ts_col, source_col, per_source=True)
    glob, meta_glob = chronological_split(df, ts_col, source_col, per_source=False)
    meta = {
        "policy": {
            "split": "per-source chronological — use for PER-CITY models",
            "split_global": "single global chronological cut — use for POOLED models",
        },
        "per_source": meta_src,
        "global": meta_glob,
    }
    return per_src, glob, meta


def add_time_features(df: pd.DataFrame, ts_col: str = "reported_at") -> pd.DataFrame:
    """
    Attach the local-time feature block (see scripts/utils/timefeatures.py).

    Prefers the precomputed table from build_time_features.py so that every
    consumer sees byte-identical values; recomputes in-process if it is absent.
    """
    from ..utils.timefeatures import TIME_FEATURE_COLUMNS, time_features

    df = df.copy()
    fp = p("processed", "features", "incident_time_features.parquet")
    if fp.exists() and "incident_id" in df.columns:
        tf = pd.read_parquet(fp)
        df = df.merge(tf, on="incident_id", how="left", suffixes=("", "_tf"))
        missing = [c for c in TIME_FEATURE_COLUMNS if c not in df.columns]
        if not missing:
            return df
        log.warning("time feature table missing %s — recomputing in process", missing)
    tf = time_features(df, ts_col=ts_col)
    for c in tf.columns:
        df[c] = tf[c]
    return df


def write(df: pd.DataFrame, subdir: str, name: str, meta: dict) -> Path:
    dest = ensure_dir(p("processed", subdir, f"{name}.parquet"))
    df.to_parquet(dest, index=False)
    meta = {**meta, "rows": int(len(df)), "path": str(dest), "columns": list(df.columns)}
    if "split" in df.columns and len(df):
        meta["split_counts"] = {str(k): int(v) for k, v in df["split"].value_counts().items()}
    ensure_dir(p("processed", subdir, f"{name}.meta.json")).write_text(json.dumps(meta, indent=2))
    log.info("%-24s %8d rows -> %s", name, len(df), dest)
    return dest
