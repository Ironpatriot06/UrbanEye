#!/usr/bin/env python3
"""
Download NYC 311 2010-2019 (Socrata 76ig-c548) as raw newline-delimited JSON.

The full asset is 22.6M rows, of which roughly 70% is housing, noise and parking.
By default we restrict to the complaint types listed in dataset_config.yaml under
include_complaint_types. Pass --all to take everything.

  python scripts/download/download_nyc311.py
  python scripts/download/download_nyc311.py --all --estimate-only
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.download._common import (DownloadBlocked, build_where, preflight,
                                      socrata_count, socrata_pages, write_jsonl_pages, fail)
from scripts.utils.manifest import human_bytes, record
from scripts.utils.paths import load_config, p, ensure_dir
from scripts.utils.logging_setup import get_logger

log = get_logger("download.nyc311")
DATASET = "nyc311"
BYTES_PER_ROW = 650  # resolution_description is long


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="do not filter complaint types")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--app-token", default=os.environ.get("SOCRATA_APP_TOKEN"))
    args = ap.parse_args()

    cfg = load_config()["datasets"][DATASET]
    sc = cfg["socrata"]
    types = sc.get("include_complaint_types") or []
    where = None if args.all or not types else build_where(include_values={"complaint_type": types})
    log.info("server-side filter: %s", (where[:120] + "...") if where and len(where) > 120 else (where or "(none)"))

    pre = preflight(sc["resource_url"], DATASET, estimated_bytes=cfg["expected"]["approx_rows_total"] * BYTES_PER_ROW)
    if not pre.get("reachable"):
        fail(DATASET, sc["resource_url"], "Socrata GET", pre.get("error", "unreachable"),
             blocked="host_not_allowed" in str(pre.get("error", "")))
        return 2

    n = socrata_count(sc["resource_url"], where)
    log.info("rows=%s  raw estimate=%s", f"{n:,}" if n else "unknown",
             human_bytes((n or cfg["expected"]["approx_rows_total"]) * BYTES_PER_ROW))
    if args.estimate_only:
        return 0

    dest = ensure_dir(p("raw", DATASET, "nyc311_2010_2019.jsonl"))
    try:
        rows = write_jsonl_pages(
            socrata_pages(sc["resource_url"], select=sc["select"], where=where,
                          order_by=sc["order_by"], page_size=sc["page_size"],
                          app_token=args.app_token, max_pages=args.max_pages),
            dest)
    except DownloadBlocked as e:
        fail(DATASET, sc["resource_url"], "Socrata GET", e, blocked=True); return 2
    except Exception as e:
        fail(DATASET, sc["resource_url"], "Socrata GET", e); return 1

    record(DATASET, dest, sc["resource_url"], row_count=rows,
           extra={"where": where, "filtered_complaint_types": [] if args.all else types})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
