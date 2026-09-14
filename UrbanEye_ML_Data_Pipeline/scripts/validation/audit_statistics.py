#!/usr/bin/env python3
"""
Statistical audit of every ML-ready table.

`audit_pipeline.py` answers "is the pipeline internally consistent?". This
answers the different and equally important question: "would a model trained on
this table be misleading?" A dataset can pass every integrity check and still
produce a model whose offline score means nothing — because a feature is
constant, because the classes are 400:1, because the validation fold sits inside
a regime change, or because a column's missingness happens to encode the city.

Every finding is CLASSIFIED rather than acted on:

  safe                 no action needed
  needs_preprocessing  usable, but only with a specific documented treatment
  dangerous            will produce a misleading model if used as-is
  policy_decision      a human has to choose; the pipeline must not choose for them

Nothing is dropped. This script only reads.

    python scripts/validation/audit_statistics.py
    python scripts/validation/audit_statistics.py --strict   # exit 1 on any `dangerous`
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import ensure_dir, p

log = get_logger("validation.statistics")

FINDINGS: list[dict] = []

SAFE = "safe"
PREP = "needs_preprocessing"
DANGER = "dangerous"
POLICY = "policy_decision"


def finding(table: str, column: str | None, issue: str, severity: str,
            detail: str, action: str) -> None:
    FINDINGS.append({"table": table, "column": column, "issue": issue,
                     "severity": severity, "detail": detail, "recommended_action": action})
    fn = {SAFE: log.info, PREP: log.info, POLICY: log.warning, DANGER: log.error}[severity]
    fn("%-18s %-34s %-20s %s", table, column or "-", severity, issue)


# ---------------------------------------------------------------------------
def _tv_distance(a: pd.Series, b: pd.Series, top: int = 40) -> float:
    """Total-variation distance between two categorical distributions, in [0,1]."""
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    keys = list(dict.fromkeys(list(pa.head(top).index) + list(pb.head(top).index)))
    return float(0.5 * sum(abs(float(pa.get(k, 0.0)) - float(pb.get(k, 0.0))) for k in keys))


def _psi(a: pd.Series, b: pd.Series, bins: int = 10) -> float:
    """Population-stability index for a numeric column. >0.25 is a large shift."""
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 100 or len(b) < 100:
        return float("nan")
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    ca = np.histogram(a, bins=edges)[0] / len(a)
    cb = np.histogram(b, bins=edges)[0] / len(b)
    eps = 1e-6
    return float(np.sum((ca - cb) * np.log((ca + eps) / (cb + eps))))


def column_health(table: str, df: pd.DataFrame, predictors: list[str]) -> dict:
    """Constant / near-constant / high-cardinality / missingness screening."""
    report = {}
    n = len(df)
    for c in predictors:
        if c not in df.columns:
            continue
        s = df[c]
        nun = int(s.nunique(dropna=True))
        null = float(s.isna().mean())
        top_share = float(s.value_counts(normalize=True, dropna=True).iloc[0]) if nun else 1.0
        entry = {"n_unique": nun, "null_share": round(null, 4),
                 "dominant_value_share": round(top_share, 4),
                 "unique_ratio": round(nun / max(1, n), 6)}
        if nun == 0:
            finding(table, c, "entirely null", DANGER,
                    "every value is missing", "declare unavailable; never impute")
        elif nun == 1:
            finding(table, c, "constant", DANGER,
                    f"one distinct value across {n:,} rows",
                    "remove from the predictor set — it cannot inform anything")
        elif top_share > 0.995:
            finding(table, c, "near-constant", PREP,
                    f"dominant value covers {top_share:.4%}",
                    "keep only if the rare level is the interesting one; otherwise drop")
        if null > 0.90 and nun > 1:
            finding(table, c, "almost entirely missing", PREP,
                    f"{null:.2%} missing",
                    "model on the sub-population that has it, or use as a presence flag")
        if nun > 1000:
            finding(table, c, "high cardinality", PREP, f"{nun:,} distinct values",
                    "hashing or per-group target encoding FITTED ON TRAIN ONLY; "
                    "never plain label-encode into a numeric feature")
        report[c] = entry
    return report


def missingness_by_group(table: str, df: pd.DataFrame, cols: list[str],
                         group: str) -> dict:
    """A column whose missingness is perfectly predicted by the group leaks the group."""
    out = {}
    if group not in df.columns:
        return out
    for c in cols:
        if c not in df.columns or df[c].notna().all() or df[c].isna().all():
            continue
        by = df.groupby(group, observed=True)[c].apply(lambda s: float(s.isna().mean()))
        out[c] = {str(k): round(float(v), 4) for k, v in by.items()}
        if len(by) > 1 and float(by.max()) > 0.99 and float(by.min()) < 0.01:
            finding(table, c, f"missingness is a perfect {group} indicator", PREP,
                    f"missing share by {group}: {out[c]}",
                    f"do not read importance for this column as anything but '{group}'; "
                    f"add an explicit is_missing flag and consider per-{group} models")
    return out


def target_health(table: str, y: pd.Series, kind: str) -> dict:
    y = y.dropna()
    if kind == "binary":
        pos = float(pd.Series(y).astype("boolean").fillna(False).mean())
        entry = {"kind": kind, "rows": int(len(y)), "positive_rate": round(pos, 5),
                 "imbalance_ratio": round((1 - pos) / pos, 2) if 0 < pos < 1 else None}
        if pos in (0.0, 1.0):
            finding(table, "TARGET", "degenerate target — one class only", DANGER,
                    f"positive rate {pos}",
                    "the task as defined is unlearnable; re-derive the target")
        elif pos < 0.05 or pos > 0.95:
            finding(table, "TARGET", "severe class imbalance", PREP,
                    f"positive rate {pos:.4%}",
                    "use PR-AUC not accuracy; pick the threshold from the cost matrix; "
                    "do not resample before choosing the metric")
        return entry
    v = pd.to_numeric(y, errors="coerce").dropna()
    entry = {"kind": kind, "rows": int(len(v)),
             "min": float(v.min()), "p50": float(v.median()), "p99": float(v.quantile(0.99)),
             "max": float(v.max()), "mean": float(v.mean()), "std": float(v.std()),
             "zero_share": round(float((v == 0).mean()), 4)}
    if len(v) and float(v.max()) > 50 * max(1e-9, float(v.quantile(0.99))):
        finding(table, "TARGET", "extreme right tail", PREP,
                f"max {v.max():,.0f} vs p99 {v.quantile(0.99):,.0f}",
                "fit on log1p or use a robust/quantile loss; report median absolute error")
    return entry


def drift(table: str, df: pd.DataFrame, cols: list[str], ts_col: str | None) -> dict:
    """train -> val -> test distribution movement for every predictor."""
    if "split" not in df.columns:
        return {}
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]
    if not len(tr) or not len(te):
        return {}
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            val = {"psi_train_val": round(_psi(tr[c], va[c]), 4) if len(va) else None,
                   "psi_train_test": round(_psi(tr[c], te[c]), 4)}
            worst = max([v for v in val.values() if v is not None and np.isfinite(v)], default=0)
            metric, thresh = "PSI", 0.25
        else:
            val = {"tv_train_val": round(_tv_distance(tr[c], va[c]), 4) if len(va) else None,
                   "tv_train_test": round(_tv_distance(tr[c], te[c]), 4)}
            worst = max([v for v in val.values() if v is not None and np.isfinite(v)], default=0)
            metric, thresh = "total-variation", 0.20
        out[c] = val
        if worst > thresh:
            finding(table, c, "distribution drift across splits", PREP,
                    f"{metric} {worst:.3f} (threshold {thresh})",
                    "expected under a chronological split — report per-split metrics and do "
                    "not treat a single pooled score as an estimate of future performance")
    return out


def duplicate_feature_rows(table: str, df: pd.DataFrame, predictors: list[str],
                           target: str | None) -> dict:
    cols = [c for c in predictors if c in df.columns]
    if not cols:
        return {}
    sample = df[cols + ([target] if target and target in df.columns else [])]
    if len(sample) > 400_000:
        sample = sample.sample(400_000, random_state=3)
    dup = int(sample[cols].duplicated().sum())
    entry = {"checked_rows": int(len(sample)), "duplicate_feature_rows": dup,
             "duplicate_share": round(dup / max(1, len(sample)), 4)}
    if target and target in sample.columns and dup:
        g = sample.groupby(cols, dropna=False, observed=True)[target].nunique()
        entry["contradictory_groups"] = int((g > 1).sum())
        if entry["duplicate_share"] > 0.5:
            finding(table, None, "most rows are feature-duplicates", PREP,
                    f"{entry['duplicate_share']:.1%} of sampled rows repeat an existing "
                    f"feature vector; {entry.get('contradictory_groups', 0):,} of those groups "
                    f"carry more than one target value",
                    "the feature set cannot separate these rows — irreducible error. "
                    "Do not read a ceiling on performance as a modelling failure")
    return entry


# ---------------------------------------------------------------------------
def audit_table(name: str, path_parts: tuple, ts_col: str | None,
                manifest_name: str | None = None) -> dict:
    fp = p(*path_parts)
    if not fp.exists():
        log.warning("%s not built — skipped", name)
        return {}
    df = pd.read_parquet(fp)

    predictors: list[str] = []
    target = None
    target_kind = "regression"
    if manifest_name:
        mf = p("processed", "ml", manifest_name)
        if mf.exists():
            m = json.loads(mf.read_text())
            if "column_roles" in m:
                predictors = [c for c, r in m["column_roles"].items() if r == "predictor"]
                target = "resolution_time_hours"
            else:
                predictors = [c for c, v in m.get("predictors", {}).items() if v.get("present")]
                predictors += [c for c, v in m.get("features", {}).items() if v.get("present")]
                target = m.get("target", {}).get("column")
                kind = (m.get("target", {}).get("kind") or "")
                target_kind = "binary" if "binary" in kind or "classification" in kind else "regression"

    out: dict = {"rows": int(len(df)), "columns": int(df.shape[1]),
                 "target": target, "predictors": len(predictors)}
    if target and target in df.columns:
        out["target_health"] = target_health(name, df[target], target_kind)
    out["column_health"] = column_health(name, df, predictors)
    for grp in ("source_dataset", "city"):
        mb = missingness_by_group(name, df, predictors, grp)
        if mb:
            out.setdefault("missingness_by_group", {})[grp] = mb
            break
    out["drift"] = drift(name, df, predictors, ts_col)
    out["duplicate_feature_rows"] = duplicate_feature_rows(name, df, predictors, target)
    if "split" in df.columns:
        out["split_counts"] = {str(k): int(v) for k, v in df["split"].value_counts().items()}
        for grp in ("source_dataset", "city"):
            if grp in df.columns:
                out["split_by_" + grp] = {
                    str(k): {str(s): int(v) for s, v in g["split"].value_counts().items()}
                    for k, g in df.groupby(grp, observed=True)}
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    tables = {
        "resolution_ml": (("processed", "ml", "resolution_ml.parquet"), "reported_at",
                          "resolution_ml.manifest.json"),
        "sla_ml": (("processed", "ml", "sla_ml.parquet"), "reported_at", "sla_ml.manifest.json"),
        "hotspot_ml": (("processed", "ml", "hotspot_ml.parquet"), "week_start",
                       "hotspot_ml.manifest.json"),
        "duplicate_ml": (("processed", "ml", "duplicate_ml.parquet"), None,
                         "duplicate_ml.manifest.json"),
        "priority_features": (("processed", "ml", "priority_features.parquet"), "reported_at",
                              "priority_features.manifest.json"),
        "urbaneye_ml": (("processed", "ml", "urbaneye_ml.parquet"), "reported_at",
                        "urbaneye_ml.manifest.json"),
    }
    result = {}
    for name, (parts, ts, mf) in tables.items():
        result[name] = audit_table(name, parts, ts, mf)

    # ---- corpus-level observations ---------------------------------------
    inc_fp = p("processed", "incidents", "all_incidents_prioritised.parquet")
    corpus = {}
    if inc_fp.exists():
        inc = pd.read_parquet(inc_fp, columns=["source_dataset", "city", "category",
                                               "reported_at", "latitude"])
        by_source = inc["source_dataset"].value_counts()
        corpus["source_balance"] = {str(k): int(v) for k, v in by_source.items()}
        corpus["source_share"] = {str(k): round(float(v), 4)
                                  for k, v in by_source.div(len(inc)).items()}
        if float(by_source.max() / by_source.sum()) > 0.6:
            finding("corpus", "source_dataset", "one source dominates the corpus", PREP,
                    f"{by_source.idxmax()} is {by_source.max() / by_source.sum():.1%} of all rows",
                    "report per-source metrics; a pooled score is a Chicago score")
        cat = inc["category"].value_counts(normalize=True)
        corpus["category_share"] = {str(k): round(float(v), 4) for k, v in cat.head(20).items()}
        unmapped = float(cat.get("UNMAPPED", 0) + cat.get("OUT_OF_SCOPE", 0)
                         + cat.get("REVIEW_REQUIRED", 0))
        if unmapped > 0.1:
            finding("corpus", "category", "large unmapped / out-of-scope population", POLICY,
                    f"{unmapped:.1%} of rows are UNMAPPED, OUT_OF_SCOPE or REVIEW_REQUIRED",
                    "extend config/category_mapping.csv — see "
                    "reports/category_mapping_candidates.csv. This is a taxonomy decision, "
                    "not one the pipeline may take on its own")
        corpus["temporal_coverage"] = {
            str(k): {"min": str(g.min()), "max": str(g.max()), "rows": int(len(g))}
            for k, g in inc.groupby("source_dataset", observed=True)["reported_at"]}
        spans = {k: (pd.Timestamp(v["max"]) - pd.Timestamp(v["min"])).days
                 for k, v in corpus["temporal_coverage"].items()}
        if max(spans.values()) > 5 * max(1, min(spans.values())):
            finding("corpus", "reported_at", "sources cover wildly different time spans", POLICY,
                    f"span in days: {spans}",
                    "the cities are not exchangeable; model per city or accept that `city` "
                    "carries most of the signal")
        # COVID / regime change probe on the dominant source
        res = p("processed", "ml", "resolution_ml.parquet")
        if res.exists():
            r = pd.read_parquet(res, columns=["source_dataset", "split", "resolution_time_hours"])
            r = r[r["source_dataset"] == by_source.idxmax()]
            med = r.groupby("split", observed=True)["resolution_time_hours"].median()
            corpus["dominant_source_median_resolution_by_split"] = {
                str(k): round(float(v), 2) for k, v in med.items()}
            if len(med) >= 2 and float(med.max() / max(1e-9, med.min())) > 1.5:
                finding("corpus", "resolution_time_hours", "regime change across splits", DANGER,
                        f"median resolution hours by split: "
                        f"{ {str(k): round(float(v), 1) for k, v in med.items()} } — the "
                        f"validation window covers the first COVID wave",
                        "do not present a single val/test score as an estimate of steady-state "
                        "performance; report per-period metrics and say which period they cover")

    sev_counts = {s: sum(1 for f in FINDINGS if f["severity"] == s)
                  for s in (SAFE, PREP, POLICY, DANGER)}
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification_scheme": {
            SAFE: "no action needed",
            PREP: "usable, with the documented treatment",
            DANGER: "will produce a misleading model if used as-is",
            POLICY: "a human must decide; the pipeline must not decide for them",
        },
        "summary": sev_counts,
        "findings": FINDINGS,
        "corpus": corpus,
        "tables": result,
        "note": ("Nothing was dropped or modified by this script. It only reads. A `dangerous` "
                 "classification is a statement about how a result should be reported, not an "
                 "instruction to delete a column."),
    }
    dest = ensure_dir(p("reports", "statistical_audit.json"))
    dest.write_text(json.dumps(report, indent=2, default=str))
    _markdown(report)
    log.info("wrote %s — %d findings (%d dangerous, %d needs_preprocessing, %d policy)",
             dest, len(FINDINGS), sev_counts[DANGER], sev_counts[PREP], sev_counts[POLICY])
    return 1 if (args.strict and sev_counts[DANGER]) else 0


def _markdown(r: dict) -> None:
    L = ["# UrbanEye+ statistical audit", "",
         f"Generated {r['generated_at_utc']}", "",
         "| Severity | Meaning | Count |", "|---|---|---|"]
    for k, v in r["classification_scheme"].items():
        L.append(f"| `{k}` | {v} | {r['summary'].get(k, 0)} |")
    L += ["", "Nothing is dropped automatically. A `dangerous` finding is a statement about "
          "how results must be reported, not an instruction to delete a column.", ""]
    for sev in (DANGER, POLICY, PREP, SAFE):
        rows = [f for f in r["findings"] if f["severity"] == sev]
        if not rows:
            continue
        L += [f"## {sev} ({len(rows)})", "",
              "| Table | Column | Issue | Detail | Action |", "|---|---|---|---|---|"]
        for f in rows:
            L.append(f"| {f['table']} | `{f['column'] or '-'}` | {f['issue']} | "
                     f"{str(f['detail'])[:160]} | {f['recommended_action'][:180]} |")
        L.append("")
    c = r.get("corpus", {})
    if c:
        L += ["## Corpus", "", f"- source balance: {c.get('source_balance')}",
              f"- temporal coverage: { {k: (v['min'][:10], v['max'][:10]) for k, v in c.get('temporal_coverage', {}).items()} }",
              f"- median resolution hours by split (dominant source): "
              f"{c.get('dominant_source_median_resolution_by_split')}", ""]
    ensure_dir(p("reports", "statistical_audit.md")).write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
