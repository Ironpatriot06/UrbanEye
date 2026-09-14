#!/usr/bin/env python3
"""
Generate reports/dataset_status.md — the at-a-glance answer to "where do we
actually stand?".

Per dataset it reports one acquisition status and one licence status:

  acquisition:  AVAILABLE LOCALLY | NOT DOWNLOADED | PROCESSED | FIXTURE ONLY
  licence:      VERIFIED | LICENSE UNVERIFIED | EXCLUDED

It checks the filesystem rather than trusting config, so it cannot claim a
dataset is present when it is not.

    python scripts/validation/dataset_status.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import ensure_dir, load_config, p, repo_root

log = get_logger("validation.status")

RAW_GLOBS = {
    "sf311": ["*.csv", "*.jsonl"],
    "boston311": ["*.csv"],
    "chicago311": ["*.csv", "*.jsonl"],
    "nyc311": ["*.csv", "*.jsonl"],
    "rdd2022": ["**/*.xml", "*.zip", "*.tar*"],
    "smartathon": ["*.csv", "*.zip"],
}


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or u == "TB":
            return f"{n:,.1f} {u}"
        n /= 1024
    return str(n)


def main() -> int:
    argparse.ArgumentParser().parse_args()
    cfg = load_config()
    rows, detail = [], {}

    for name, d in cfg["datasets"].items():
        raw_dir = p("raw", name)
        files, size = [], 0
        for g in RAW_GLOBS.get(name, ["*"]):
            for f in raw_dir.glob(g):
                if f.is_file() and not f.name.startswith("."):
                    files.append(f)
                    size += f.stat().st_size

        kind = d.get("kind", "tabular")
        proc = (p("processed", "incidents", f"{name}.parquet") if kind == "tabular"
                else p("processed", "vision", f"{name}.parquet"))
        processed_rows = None
        if proc.exists():
            # Read row count from Parquet footer metadata — no need to load the data.
            try:
                import pyarrow.parquet as _pq
                processed_rows = _pq.ParquetFile(proc).metadata.num_rows
            except Exception:
                processed_rows = len(pd.read_parquet(proc))

        if files and proc.exists():
            acq = "PROCESSED"
        elif files:
            acq = "AVAILABLE LOCALLY"
        elif proc.exists():
            acq = "FIXTURE ONLY"     # output exists but no raw file backs it
        else:
            acq = "NOT DOWNLOADED"

        lic = d.get("licence", {})
        decision = lic.get("decision")
        lic_status = {"INCLUDE": "VERIFIED", "REVIEW": "LICENSE UNVERIFIED",
                      "EXCLUDE": "EXCLUDED"}.get(decision, str(decision))

        rows.append({
            "dataset": name, "kind": kind, "acquisition": acq,
            "raw_files": len(files), "raw_size": human(size) if size else "—",
            "processed_rows": f"{processed_rows:,}" if processed_rows is not None else "—",
            "licence": lic.get("name"), "licence_status": lic_status,
            "decision": decision,
        })
        detail[name] = {
            "landing_page": d.get("landing_page"),
            "acquisition": acq,
            "raw_dir": str(raw_dir.relative_to(repo_root())),
            "raw_files_found": [f.name for f in files[:10]],
            "licence": lic, "licence_status": lic_status,
        }

    excluded = []
    for name, d in cfg.get("excluded_datasets", {}).items():
        excluded.append({"dataset": name, "url": d.get("url"), "licence": d.get("licence"),
                         "decision": d.get("decision"), "reason": d.get("reason")})

    fixture_mode = any(r["acquisition"] == "FIXTURE ONLY" for r in rows)

    L = ["# UrbanEye+ dataset status", "",
         f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}", ""]
    if fixture_mode:
        L += ["> **WARNING — FIXTURE-DERIVED OUTPUTS PRESENT.**",
              "> One or more processed tables exist with no raw file behind them, which means",
              "> they came from `tests/fixtures/` (synthetic). They are NOT real data and must",
              "> not be used for training or reported as results. Delete `data/processed/` and",
              "> re-run once you have downloaded the real datasets.", ""]

    L += ["## Core datasets", "",
          "| Dataset | Type | Acquisition | Raw files | Raw size | Processed rows | Licence status |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['dataset']} | {r['kind']} | **{r['acquisition']}** | {r['raw_files']} | "
                 f"{r['raw_size']} | {r['processed_rows']} | **{r['licence_status']}** |")

    L += ["", "## Licences", "", "| Dataset | Licence | Commercial | Redistribute | Decision |",
          "|---|---|---|---|---|"]
    for name, d in cfg["datasets"].items():
        lic = d.get("licence", {})
        L.append(f"| {name} | {lic.get('name')} | {lic.get('commercial_use')} | "
                 f"{lic.get('redistribution')} | **{lic.get('decision')}** |")

    L += ["", "## Deliberately excluded (documented, not silently dropped)", "",
          "| Dataset | Licence | Decision | Reason |", "|---|---|---|---|"]
    for e in excluded:
        reason = " ".join(str(e["reason"]).split())
        L.append(f"| {e['dataset']} | {e['licence']} | **{e['decision']}** | {reason} |")

    not_downloaded = [r["dataset"] for r in rows if r["acquisition"] == "NOT DOWNLOADED"]
    unverified = [r["dataset"] for r in rows if r["licence_status"] == "LICENSE UNVERIFIED"]

    L += ["", "## What this means for you", ""]
    if not_downloaded:
        L += [f"**Still to download ({len(not_downloaded)}):** {', '.join(not_downloaded)}",
              "", "See `DOWNLOAD_INSTRUCTIONS.md` for the exact URL, target directory and",
              "command for each.", ""]
    else:
        L += ["All core datasets have raw files present.", ""]
    if unverified:
        L += [f"**Licence unverified ({len(unverified)}):** {', '.join(unverified)}", "",
              "These are preprocessed for inspection but are EXCLUDED from",
              "`vision_category_dataset` until you confirm the original terms and change",
              "`decision` to `INCLUDE` in `config/dataset_config.yaml`. The pipeline will not",
              "quietly train on data whose licence nobody has read.", ""]

    dest = ensure_dir(p("reports", "dataset_status.md"))
    dest.write_text("\n".join(L))
    ensure_dir(p("reports", "dataset_status.json")).write_text(
        json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "fixture_outputs_present": fixture_mode,
                    "datasets": detail, "excluded": excluded}, indent=2))
    log.info("wrote %s", dest)
    for r in rows:
        log.info("  %-12s %-18s %s", r["dataset"], r["acquisition"], r["licence_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
