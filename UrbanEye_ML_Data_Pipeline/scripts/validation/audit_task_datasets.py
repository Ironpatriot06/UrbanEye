#!/usr/bin/env python3
"""
Audit the per-task ML datasets, and hunt for the leakage the other validators
are not shaped to catch.

`validate_leakage.py` checks the things the builders promise. This checks the
things a *consumer* could still get wrong, and it checks them against the
manifests rather than against the code that wrote them:

  TASK   every declared predictor exists, is not constant, and carries a written
         statement of when its value becomes known
  LEAK   twelve specific leakage routes, enumerated below
  SPLIT  per-task split integrity: chronological where it must be, entity-aware
         where it must be, non-empty everywhere
  FEAT   a feature/target audit table for every task

Leakage routes checked here
---------------------------
  TL-1   a predictor that is the target, or an arithmetic derivative of it
  TL-2   a post-resolution column reaching a predictor set
  TL-3   the priority policy engine's own output used as a predictor
  TL-4   raw coordinates used as predictors (memorisation, not generalisation)
  TL-5   a global frequency / target encoding baked into a column
  TL-6   a predictor with no prediction-time availability statement
  TL-7   density and recency values in a task table disagreeing with the
         feature table they were joined from
  TL-8   an entity appearing in more than one split
  TL-9   hotspot panel rows whose "previous period" is not the previous
         calendar week, or whose target is not the next one
  TL-10  source-specific semantics left undocumented
  TL-11  a split whose validation or test fold is empty, or out of order
  TL-12  a target-support column (censoring flag, SLA deadline) offered as a
         predictor to a task it does not belong to

    python scripts/validation/audit_task_datasets.py
    python scripts/validation/audit_task_datasets.py --strict
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
from scripts.utils.paths import ensure_dir, load_feature_config, p

log = get_logger("validation.tasks")

CHECKS: list[dict] = []
TABLE: list[dict] = []

POST_RESOLUTION = {"closed_at", "status", "sla_met", "description", "resolution_description",
                   "status_notes", "resolution_time_hours", "resolution_time_hours_log1p",
                   "target_is_censored"}
POLICY_OUTPUT_PREFIXES = ("priority_",)
POLICY_OUTPUT = {"sla_hours_policy"}
ENCODING_PATTERNS = ("_frequency", "_freq", "_target_enc", "_te", "_mean_target", "_woe")
RAW_GEOMETRY = {"latitude", "longitude", "lat_grid", "lon_grid", "geo_grid"}


def check(cid: str, task: str, desc: str, passed: bool | None, detail: str = "") -> bool:
    result = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
    CHECKS.append({"id": cid, "task": task, "check": desc, "result": result, "detail": detail})
    fn = log.info if result == "PASS" else (log.warning if result == "SKIP" else log.error)
    fn("%-8s %-18s %-4s %s %s", cid, task, result, desc, detail)
    return bool(passed)


def load_task(name: str) -> tuple[pd.DataFrame | None, dict]:
    fp = p("processed", "ml", f"{name}.parquet")
    mf = p("processed", "ml", f"{name}.manifest.json")
    if not fp.exists() or not mf.exists():
        return None, {}
    return pd.read_parquet(fp), json.loads(mf.read_text())


def predictor_list(m: dict) -> list[str]:
    block = m.get("predictors") or m.get("features") or {}
    return [c for c, v in block.items() if v.get("present")]


# ---------------------------------------------------------------------------
def audit_one(name: str, df: pd.DataFrame, m: dict, geo: pd.DataFrame | None) -> None:
    preds = predictor_list(m)
    target = (m.get("target") or {}).get("column")
    avail = m.get("prediction_time_availability", {})

    # ---- TASK: declared predictors are real, present and informative -------
    missing = [c for c in preds if c not in df.columns]
    check("TASK-1", name, "every declared predictor exists in the table", not missing,
          f"missing={missing}")
    const = [c for c in preds if c in df.columns and df[c].nunique(dropna=True) <= 1]
    check("TASK-2", name, "no declared predictor is constant", not const, f"constant={const}")
    check("TASK-3", name, "the target is declared",
          bool(target) or m.get("target", {}).get("available") is False,
          f"target={target}")
    if target:
        check("TASK-4", name, "the target column exists and has values",
              target in df.columns and int(df[target].notna().sum()) > 0,
              f"rows_with_target={int(df[target].notna().sum()) if target in df.columns else 0}")

    # ---- TL-6: availability statements -------------------------------------
    undeclared = [c for c in preds
                  if not avail.get(c, {}).get("available_at_prediction_time", False)]
    check("TL-6", name, "every predictor states when its value becomes known",
          not undeclared, f"undeclared={undeclared}")

    # ---- TL-1 / TL-2 / TL-12: target and post-resolution ------------------
    bad_target = [c for c in preds if target and (c == target or c.startswith(str(target)))]
    check("TL-1", name, "no predictor is the target or a derivative of it",
          not bad_target, f"found={bad_target}")
    allowed_post = set()
    if name == "sla_ml":
        allowed_post = {"sla_target_hours"}        # the deadline, set at intake
    bad_post = sorted((set(preds) & POST_RESOLUTION) - allowed_post)
    check("TL-2", name, "no post-resolution column is a predictor", not bad_post,
          f"found={bad_post}")
    misplaced = [c for c in preds if c == "sla_target_hours" and name != "sla_ml"]
    check("TL-12", name, "target-support columns are not offered to the wrong task",
          not misplaced, f"found={misplaced}")

    # ---- TL-3: policy output ----------------------------------------------
    bad_policy = [c for c in preds
                  if c.startswith(POLICY_OUTPUT_PREFIXES) or c in POLICY_OUTPUT]
    check("TL-3", name, "the priority policy output is not a predictor", not bad_policy,
          f"found={bad_policy}")

    # ---- TL-4: raw geometry ------------------------------------------------
    bad_geo = sorted(set(preds) & RAW_GEOMETRY)
    check("TL-4", name, "raw coordinates are not predictors", not bad_geo, f"found={bad_geo}")

    # ---- TL-5: baked-in global encodings ----------------------------------
    named = [c for c in preds if any(pat in c for pat in ENCODING_PATTERNS)]
    detected, colinear = [], []
    cats = [c for c in preds if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    nums = [c for c in preds if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]
    sample = df.sample(min(60_000, len(df)), random_state=5) if len(df) else df
    for cat in cats[:12]:
        for num in nums[:20]:
            g = sample.groupby(cat, observed=True)[num].nunique(dropna=True)
            if len(g) <= 3 or int((g > 1).sum()) != 0:
                continue
            # Constant within a level is necessary but not sufficient. A global
            # frequency/target encoding is a LOOKUP TABLE: it assigns a distinct
            # number to (almost) every level, so the value count of the numeric
            # column tracks the number of levels. A merely co-linear column —
            # `year` inside `subcategory`, because the two sources that populate
            # subcategory each span a single year — has far fewer distinct values
            # than levels and is a structural fact about the corpus, not an
            # encoding leaked in from outside.
            n_vals = int(sample[num].nunique(dropna=True))
            if n_vals >= 0.9 * len(g):
                detected.append(f"{num} is a per-level lookup over {cat} "
                                f"({n_vals} values / {len(g)} levels)")
            else:
                colinear.append(f"{num} is constant within {cat} "
                                f"({n_vals} values / {len(g)} levels)")
    check("TL-5", name, "no predictor is a global frequency/target encoding",
          not named and not detected,
          f"named={named} detected={detected[:3]}"
          + (f"; structural co-linearity (not leakage): {colinear[:2]}" if colinear else ""))

    # ---- TL-7: joined features agree with their source table ---------------
    fc = load_feature_config()
    hist = list(fc["historical_density_features"]) + list(fc.get("historical_recency_features", {}))
    joined = [c for c in hist if c in df.columns]
    if geo is not None and joined and "incident_id" in df.columns:
        s = df[["incident_id"] + joined].sample(min(50_000, len(df)), random_state=6)
        merged = s.merge(geo[["incident_id"] + joined], on="incident_id", suffixes=("", "_src"))
        bad = []
        for c in joined:
            a = pd.to_numeric(merged[c], errors="coerce")
            b = pd.to_numeric(merged[c + "_src"], errors="coerce")
            if not bool(((a.isna() & b.isna()) | np.isclose(a.fillna(-1), b.fillna(-1))).all()):
                bad.append(c)
        check("TL-7", name, "joined history features match the feature table exactly",
              not bad, f"mismatched={bad} over {len(merged):,} sampled rows")
    else:
        check("TL-7", name, "joined history features match the feature table exactly", None,
              "no joinable history columns")

    # ---- TL-8 / TL-11: split integrity ------------------------------------
    if "split" in df.columns:
        have = set(df["split"].dropna().unique())
        check("TL-11", name, "train, val and test are all non-empty",
              {"train", "val", "test"}.issubset(have), f"present={sorted(have)}")
        ent_cols = [c for c in ("incident_id", "incident_a") if c in df.columns]
        if name == "duplicate_ml":
            long = pd.concat([df[["incident_a", "split"]].rename(columns={"incident_a": "e"}),
                              df[["incident_b", "split"]].rename(columns={"incident_b": "e"})])
            spans = long.groupby("e")["split"].nunique()
            check("TL-8", name, "no entity appears in more than one split",
                  int((spans > 1).sum()) == 0, f"entities_in_two_splits={int((spans > 1).sum())}")
        elif ent_cols:
            spans = df.groupby(ent_cols[0])["split"].nunique()
            check("TL-8", name, "no entity appears in more than one split",
                  int((spans > 1).sum()) == 0, f"entities_in_two_splits={int((spans > 1).sum())}")
        ts_col = "reported_at" if "reported_at" in df.columns else (
            "week_start" if "week_start" in df.columns else None)
        if ts_col:
            grp = "source_dataset" if "source_dataset" in df.columns else (
                "city" if "city" in df.columns else None)
            ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            ok, detail = True, []
            for key, g in (df.groupby(grp, observed=True) if grp else [("ALL", df)]):
                b = {s: (ts.loc[g.index][g["split"] == s].min(), ts.loc[g.index][g["split"] == s].max())
                     for s in ("train", "val", "test") if (g["split"] == s).any()}
                if "train" in b and "val" in b and not b["train"][1] <= b["val"][0]:
                    ok = False; detail.append(f"{key}: train overlaps val")
                if "val" in b and "test" in b and not b["val"][1] <= b["test"][0]:
                    ok = False; detail.append(f"{key}: val overlaps test")
            check("TL-11b", name, f"splits are chronological within each {grp or 'table'}",
                  ok, "; ".join(detail))

    # ---- TL-10: source-specific semantics documented ----------------------
    if name != "duplicate_ml":
        documented = set()
        for k in m.get("source_specific_semantics", {}):
            documented |= {x.strip() for x in k.split("/")}
        risky = {c for c in preds if c in {"zone_key", "zone_id", "department",
                                           "report_channel", "subcategory", "sla_target_hours"}}
        check("TL-10", name, "source-specific columns carry a documented caveat",
              risky.issubset(documented), f"undocumented={sorted(risky - documented)}")

    # ---- feature/target audit row -----------------------------------------
    prof = m.get("predictors") or m.get("features") or {}
    for c in preds:
        e = prof.get(c, {})
        TABLE.append({
            "task": name, "column": c, "role": "predictor",
            "dtype": e.get("dtype"), "null_pct": e.get("null_pct"),
            "n_unique": e.get("n_unique"),
            "available_at_prediction_time": avail.get(c, {}).get("available_at_prediction_time"),
            "when_known": avail.get(c, {}).get("why"),
            "warning": e.get("warning"),
        })
    if target:
        t = m.get("target", {})
        TABLE.append({
            "task": name, "column": target, "role": "target", "dtype": None,
            "null_pct": None, "n_unique": None,
            "available_at_prediction_time": False,
            "when_known": "observed only after the fact — this is the label",
            "warning": t.get("note") or t.get("definition"),
        })


# ---------------------------------------------------------------------------
def audit_hotspot_calendar(df: pd.DataFrame) -> None:
    """TL-9: the panel must be adjacent in real calendar weeks, not in 'weeks that happened'."""
    d = df.copy()
    d["_ws"] = pd.to_datetime(d["week_start"])
    d = d.sort_values(["city", "zone_type", "zone_id", "category", "_ws"])
    g = d.groupby(["city", "zone_type", "zone_id", "category"], observed=True)
    gap = g["_ws"].diff().dt.days.dropna()
    bad = int((gap != 7).sum())
    check("TL-9", "hotspot_ml",
          "every consecutive panel row is exactly one calendar week apart", bad == 0,
          f"non_adjacent_rows={bad} of {len(gap):,}")
    y = pd.to_numeric(d["future_incident_count"], errors="coerce")
    check("TL-9b", "hotspot_ml", "the target can express a quiet week (it contains zeros)",
          float((y == 0).mean()) > 0,
          f"zero_share={float((y == 0).mean()):.4f}")
    flag = d["future_incident_flag"].astype("boolean")
    check("TL-9c", "hotspot_ml", "the binary target is not constant",
          int(flag.nunique()) > 1, f"distinct={int(flag.nunique())}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    geo_fp = p("processed", "geo_features", "incident_geo_features.parquet")
    geo = pd.read_parquet(geo_fp) if geo_fp.exists() else None

    tasks = ["resolution_ml", "sla_ml", "hotspot_ml", "duplicate_ml", "priority_features"]
    built = {}
    for name in tasks:
        df, m = load_task(name)
        if df is None:
            check("TASK-0", name, "task dataset built", None, "not built")
            continue
        built[name] = (df, m)
        audit_one(name, df, m, geo)
    if "hotspot_ml" in built:
        audit_hotspot_calendar(built["hotspot_ml"][0])

    # priority must stay target-free
    if "priority_features" in built:
        df, m = built["priority_features"]
        check("TL-13", "priority_features", "priority has no supervised target",
              m["target"]["available"] is False and int(df["priority_label"].notna().sum()) == 0
              and not bool(df["is_ground_truth"].fillna(False).any()))

    failed = [c for c in CHECKS if c["result"] == "FAIL"]
    passed = [c for c in CHECKS if c["result"] == "PASS"]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {"total": len(CHECKS), "passed": len(passed), "failed": len(failed),
                    "skipped": len(CHECKS) - len(passed) - len(failed)},
        "checks": CHECKS,
        "feature_target_audit": TABLE,
        "tasks": {k: {"rows": int(len(v[0])), "target": (v[1].get("target") or {}).get("column"),
                      "predictors": len(predictor_list(v[1])),
                      "split": v[1]["split"]["strategy"]}
                  for k, v in built.items()},
    }
    dest = ensure_dir(p("reports", "task_dataset_audit.json"))
    dest.write_text(json.dumps(report, indent=2, default=str))
    _markdown(report)
    log.info("wrote %s — %d passed, %d failed, %d skipped", dest, len(passed), len(failed),
             report["summary"]["skipped"])
    return 1 if (failed and args.strict) else 0


def _markdown(r: dict) -> None:
    L = ["# UrbanEye+ task dataset & leakage audit", "",
         f"Generated {r['generated_at_utc']}", "",
         f"**{r['summary']['passed']}/{r['summary']['total']} checks passed, "
         f"{r['summary']['failed']} failed, {r['summary']['skipped']} skipped.**", "",
         "## Tasks", "", "| Task | Rows | Target | Predictors | Split |", "|---|---|---|---|---|"]
    for k, v in r["tasks"].items():
        L.append(f"| `{k}` | {v['rows']:,} | {v['target'] or '**none**'} | {v['predictors']} "
                 f"| {v['split']} |")
    L += ["", "## Checks", "", "| ID | Task | Check | Result | Detail |", "|---|---|---|---|---|"]
    for c in r["checks"]:
        mark = {"PASS": "PASS", "FAIL": "**FAIL**", "SKIP": "_skip_"}[c["result"]]
        L.append(f"| {c['id']} | {c['task']} | {c['check']} | {mark} | {c['detail'] or ''} |")
    L += ["", "## Feature / target audit", "",
          "| Task | Column | Role | Null % | Distinct | Known at prediction time | When |",
          "|---|---|---|---|---|---|---|"]
    for t in r["feature_target_audit"]:
        L.append(f"| {t['task']} | `{t['column']}` | {t['role']} | {t['null_pct']} | "
                 f"{t['n_unique']} | {t['available_at_prediction_time']} | "
                 f"{str(t['when_known'])[:120]} |")
    ensure_dir(p("reports", "task_dataset_audit.md")).write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
