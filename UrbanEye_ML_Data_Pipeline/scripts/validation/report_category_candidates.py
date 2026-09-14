#!/usr/bin/env python3
"""
Rank the unmapped source categories and propose candidate mappings — as a
REPORT, for a human to accept or reject.

This script writes nothing to `config/category_mapping.csv`. It cannot: deciding
that "Alley Light Out Complaint" is a STREETLIGHT_FAULT is a taxonomy decision
with consequences for what every downstream model sees, and 42,315 rows would
move on the strength of it. The pipeline's job is to make that decision cheap and
well-evidenced, not to take it.

Method — deliberately conservative
----------------------------------
Every proposal is justified by an EXISTING mapping already in
config/category_mapping.csv. For each unmapped source string we score the
canonical categories by token overlap against the source strings that are
already mapped to them, preferring evidence from the same publisher, and report:

  proposed_category   the best-scoring canonical category, or blank
  confidence          high / medium / low / none
  evidence            the already-mapped source strings that produced the score
  shared_tokens       exactly which words drove it

`high` requires a decisive margin over the runner-up AND a shared token that is
not a generic civic word ("complaint", "request", "service", ...). Anything
weaker is reported with its evidence and left for review. Nothing here is
applied automatically, and a proposal with no evidence gets no proposal at all
rather than a guess.

    python scripts/validation/report_category_candidates.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from scripts.utils.logging_setup import get_logger
from scripts.utils.mapping import CANONICAL, load_mapping
from scripts.utils.paths import ensure_dir, p

log = get_logger("validation.category_candidates")

# Words that appear in every 311 vocabulary and carry no category signal.
STOPWORDS = {
    "complaint", "complaints", "request", "requests", "service", "report", "reported",
    "inspection", "inspect", "issue", "issues", "problem", "condition", "conditions",
    "new", "other", "misc", "miscellaneous", "the", "a", "an", "of", "for", "to", "and",
    "or", "in", "on", "at", "by", "with", "not", "no", "non", "city", "public",
}
TOKEN = re.compile(r"[a-z0-9]+")


def tokens(s: str) -> set[str]:
    return {t for t in TOKEN.findall(str(s).lower()) if t not in STOPWORDS and len(t) > 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-rows", type=int, default=1,
                    help="only report source categories with at least this many rows")
    args = ap.parse_args()

    unmapped_fp = p("reports", "unmapped_categories.csv")
    if not unmapped_fp.exists():
        log.error("no %s — run the preprocessors first", unmapped_fp)
        return 2
    un = pd.read_csv(unmapped_fp, dtype={"source_category": str})
    un = un[un["row_count"] >= args.min_rows].copy()

    mapping = load_mapping()
    mapped = mapping[mapping["urbaneye_category"].isin(CANONICAL)]

    # canonical -> list of (source_dataset, source_string, token set)
    evidence: dict[str, list[tuple[str, str, set[str]]]] = defaultdict(list)
    for _, r in mapped.iterrows():
        src = str(r["source_category"]).split("|")[-1]
        evidence[r["urbaneye_category"]].append(
            (r["source_dataset"], str(r["source_category"]), tokens(src)))

    total_rows = int(un["row_count"].sum())
    rows = []
    for _, r in un.iterrows():
        ds, raw, n = r["source_dataset"], str(r["source_category"]), int(r["row_count"])
        t = tokens(raw)
        scored = []
        for canon, examples in evidence.items():
            best, best_ex, best_shared = 0.0, None, set()
            for ex_ds, ex_src, ex_tokens in examples:
                shared = t & ex_tokens
                if not shared:
                    continue
                # Jaccard, with a bonus when the evidence is from the same publisher
                j = len(shared) / len(t | ex_tokens)
                score = j * (1.25 if ex_ds == ds else 1.0)
                if score > best:
                    best, best_ex, best_shared = score, f"{ex_ds}:{ex_src}", shared
            if best > 0:
                scored.append((best, canon, best_ex, best_shared))
        scored.sort(reverse=True)

        proposal, conf, why, shared_tokens = "", "none", "no token overlap with any existing mapping", ""
        if scored:
            top = scored[0]
            runner = scored[1][0] if len(scored) > 1 else 0.0
            margin = top[0] - runner
            shared_tokens = " ".join(sorted(top[3]))
            proposal = top[1]
            why = f"resembles already-mapped {top[2]}"
            if top[0] >= 0.5 and margin >= 0.2:
                conf = "high"
            elif top[0] >= 0.3 and margin >= 0.1:
                conf = "medium"
            else:
                conf = "low"
                why += (f"; ambiguous — runner-up {scored[1][1]} scores {runner:.2f} "
                        f"vs {top[0]:.2f}" if len(scored) > 1 else "; weak overlap")
        rows.append({
            "source_dataset": ds,
            "source_category": raw,
            "row_count": n,
            "share_of_unmapped": round(n / max(1, total_rows), 4),
            "proposed_category": proposal,
            "confidence": conf,
            "shared_tokens": shared_tokens,
            "evidence": why,
            "alternatives": "; ".join(f"{c}({s:.2f})" for s, c, _, _ in scored[1:4]),
            "decision": "REVIEW — not applied by any script",
        })

    out = pd.DataFrame(rows).sort_values(["row_count"], ascending=False)
    dest = ensure_dir(p("reports", "category_mapping_candidates.csv"))
    out.to_csv(dest, index=False)

    by_conf = out.groupby("confidence")["row_count"].agg(["count", "sum"]).to_dict("index")
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unmapped_source_categories": int(len(out)),
        "unmapped_rows_total": total_rows,
        "by_confidence": {k: {"source_categories": int(v["count"]), "rows": int(v["sum"])}
                          for k, v in by_conf.items()},
        "rows_recoverable_at_high_confidence": int(
            out.loc[out["confidence"] == "high", "row_count"].sum()),
        "policy": ("NOTHING IS APPLIED. Every row above is a proposal backed by an existing "
                   "mapping, for a human to accept or reject in config/category_mapping.csv. "
                   "Mapping a category changes which rows reach every model-ready table, so "
                   "it is a taxonomy decision, not a data-cleaning one."),
        "modelling_impact": (
            "UNMAPPED and OUT_OF_SCOPE rows stay in all_incidents for auditability but never "
            "reach a model-ready table. Accepting the high-confidence proposals would move "
            f"{int(out.loc[out['confidence'] == 'high', 'row_count'].sum()):,} rows into scope, "
            "changing the category prior every model sees and the density features of "
            "neighbouring incidents."),
    }
    ensure_dir(p("reports", "category_mapping_candidates.json")).write_text(
        json.dumps(summary, indent=2))

    L = ["# Candidate category mappings", "",
         f"Generated {summary['generated_at_utc']}", "",
         f"**{summary['unmapped_source_categories']} unmapped source categories covering "
         f"{total_rows:,} rows.** Nothing below has been applied.", "",
         "| Confidence | Source categories | Rows |", "|---|---|---|"]
    for k in ("high", "medium", "low", "none"):
        v = summary["by_confidence"].get(k)
        if v:
            L.append(f"| {k} | {v['source_categories']} | {v['rows']:,} |")
    L += ["", "## Top 40 by row count", "",
          "| Source | Category | Rows | Proposal | Confidence | Shared tokens | Evidence |",
          "|---|---|---|---|---|---|---|"]
    for _, r in out.head(40).iterrows():
        L.append(f"| {r['source_dataset']} | {r['source_category']} | {r['row_count']:,} | "
                 f"{r['proposed_category'] or '—'} | {r['confidence']} | {r['shared_tokens']} | "
                 f"{r['evidence'][:90]} |")
    L += ["", "## How to act on this", "",
          "1. Read the `high` rows first — they are the cheapest wins and the least ambiguous.",
          "2. Append accepted rows to `config/category_mapping.csv` with a `notes` field "
          "saying who accepted them and why.",
          "3. Re-run the pipeline. The category prior and the density features both change, "
          "so re-read `reports/statistical_audit.md` afterwards.",
          "4. Anything genuinely ambiguous belongs in `REVIEW_REQUIRED`, not in a canonical "
          "category — that sentinel exists precisely so a guess is never necessary.", ""]
    ensure_dir(p("reports", "category_mapping_candidates.md")).write_text("\n".join(L))

    log.info("wrote %s — %d candidates, %d rows; high-confidence recoverable: %d rows",
             dest, len(out), total_rows, summary["rows_recoverable_at_high_confidence"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
