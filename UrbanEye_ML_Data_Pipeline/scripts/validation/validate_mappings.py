#!/usr/bin/env python3
"""
Validate config/category_mapping.csv against the canonical taxonomy and against
the source vocabularies actually present in the processed data.

Checks:
  MAP-1  every urbaneye_category is canonical or a known sentinel
  MAP-2  no duplicate (source_dataset, source_category) rows
  MAP-3  every confidence value is one of high/medium/low/none
  MAP-4  low/none-confidence mappings are flagged REVIEW or sent to a sentinel
  MAP-5  RDD2022 D50 is NOT mapped to OPEN_MANHOLE. D50 marks an INTACT cover;
         mapping it would invert the label's meaning
  MAP-6  every canonical category is reachable from at least one source

Also reports which UrbanEye+ categories have NO data at all. That list is the
direct input to your own data-collection plan.

    python scripts/validation/validate_mappings.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from scripts.utils.logging_setup import get_logger
from scripts.utils.mapping import CANONICAL, SENTINELS, load_mapping
from scripts.utils.paths import ensure_dir, p

log = get_logger("validation.mappings")
R: list[dict] = []


def rec(cid: str, desc: str, ok: bool, detail: str = "") -> bool:
    R.append({"id": cid, "check": desc, "result": "PASS" if ok else "FAIL", "detail": detail})
    (log.info if ok else log.error)("%-6s %-4s %s %s", cid, "PASS" if ok else "FAIL", desc, detail)
    return ok


def main() -> int:
    argparse.ArgumentParser().parse_args()
    m = load_mapping()
    log.info("loaded %d mapping rows across %d source datasets",
             len(m), m["source_dataset"].nunique())

    allowed = set(CANONICAL) | set(SENTINELS)
    bad = sorted(set(m["urbaneye_category"]) - allowed)
    rec("MAP-1", "all target categories are canonical or known sentinels", not bad, str(bad))

    dup = m.duplicated(subset=["source_dataset", "source_category"], keep=False)
    rec("MAP-2", "no duplicate (source_dataset, source_category) rows", not dup.any(),
        f"{int(dup.sum())} duplicated rows")

    ok_conf = {"high", "medium", "low", "none", ""}
    badc = sorted(set(m["confidence"]) - ok_conf)
    rec("MAP-3", "confidence values are valid", not badc, str(badc))

    weak = m[m["confidence"].isin(["low", "none"])]
    unflagged = weak[~weak["urbaneye_category"].isin(SENTINELS)
                     & ~weak["notes"].str.contains("REVIEW", case=False, na=False)]
    rec("MAP-4", "low/none-confidence mappings are flagged REVIEW or sent to a sentinel",
        len(unflagged) == 0,
        f"{len(unflagged)} unflagged: {unflagged['source_category'].tolist()[:5]}")

    d50 = m[(m["source_dataset"] == "rdd2022") & (m["source_category"] == "D50")]
    d50_target = d50["urbaneye_category"].iloc[0] if len(d50) else "absent"
    rec("MAP-5", "RDD2022 D50 is NOT mapped to OPEN_MANHOLE",
        d50_target != "OPEN_MANHOLE",
        f"D50 -> {d50_target} (D50 marks an INTACT cover; mapping it to OPEN_MANHOLE "
        "would invert the label)")

    reachable = set(m.loc[m["urbaneye_category"].isin(CANONICAL), "urbaneye_category"])
    gaps = sorted(set(CANONICAL) - reachable)
    # MAP-6 is reported as a COVERAGE GAP, not a FAIL. OPEN_MANHOLE genuinely has
    # no counterpart in any of the four US 311 vocabularies, and OTHER is a
    # deliberate catch-all with no source of its own. Failing the build forever
    # over a true fact about the world would just train people to ignore the
    # validator. The gap is recorded and surfaced instead.
    R.append({"id": "MAP-6", "check": "every canonical category is reachable from some source",
              "result": "PASS" if not gaps else "COVERAGE_GAP",
              "detail": (f"no source vocabulary maps to: {gaps}. This is a real gap in the "
                         f"public data, not a mapping error. It is the primary justification "
                         f"for collecting UrbanEye+'s own Indian dataset.") if gaps else ""})
    log.warning("MAP-6 COVERAGE_GAP  no source maps to: %s", gaps) if gaps else \
        log.info("MAP-6  PASS every canonical category is reachable")

    # Observed coverage in processed data, as opposed to theoretical reachability
    coverage: dict[str, int] = {}
    inc = p("processed", "incidents", "all_incidents.parquet")
    if inc.exists():
        df = pd.read_parquet(inc)
        vc = df["category"].value_counts()
        coverage = {c: int(vc.get(c, 0)) for c in CANONICAL}

    vision_cov: dict[str, int] = {}
    for ds in ("rdd2022", "smartathon"):
        fp = p("processed", "vision", f"{ds}.parquet")
        if fp.exists():
            v = pd.read_parquet(fp)
            for k, n in v["class"].value_counts().items():
                if k in CANONICAL:
                    vision_cov[k] = vision_cov.get(k, 0) + int(n)

    zero = [c for c, n in coverage.items() if n == 0] if coverage else gaps
    weak_cov = {c: n for c, n in coverage.items() if 0 < n < 100} if coverage else {}

    unmapped_fp = p("reports", "unmapped_categories.csv")
    unmapped = pd.read_csv(unmapped_fp).to_dict("records") if unmapped_fp.exists() else []

    failures = [r for r in R if r["result"] == "FAIL"]  # COVERAGE_GAP is not a failure
    report = {
        "summary": {"total": len(R), "passed": sum(1 for r in R if r["result"] == "PASS"),
                    "coverage_gaps": sum(1 for r in R if r["result"] == "COVERAGE_GAP"),
                    "failed": len(failures)},
        "checks": R,
        "mapping_rows": int(len(m)),
        "by_source": {k: int(v) for k, v in m["source_dataset"].value_counts().items()},
        "by_confidence": {k: int(v) for k, v in m["confidence"].value_counts().items()},
        "targets_review_required": int((m["urbaneye_category"] == "REVIEW_REQUIRED").sum()),
        "targets_out_of_scope": int((m["urbaneye_category"] == "OUT_OF_SCOPE").sum()),
        "observed_incident_counts": coverage,
        "observed_vision_counts": vision_cov,
        "categories_with_no_incident_data": zero,
        "categories_with_weak_incident_data_lt_100": weak_cov,
        "unmapped_source_values_awaiting_review": unmapped,
        "interpretation": (
            "categories_with_no_incident_data is the direct input to the UrbanEye+ "
            "data-collection plan: civic issues the product must handle but for which no "
            "public source supplies a single training row."),
    }
    dest = ensure_dir(p("reports", "mapping_validation.json"))
    dest.write_text(json.dumps(report, indent=2))
    log.info("categories with NO incident data: %s", zero)
    log.info("wrote %s", dest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
