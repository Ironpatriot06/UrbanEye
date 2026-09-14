#!/usr/bin/env python3

"""
Download SF 311 Cases from Socrata as newline-delimited JSON.

Examples:

    # Download from 2018 onward
    python scripts/download/download_sf311.py --since 2018-01-01

    # Download only first 2 pages for testing
    python scripts/download/download_sf311.py --since 2018-01-01 --max-pages 2

    # Estimate only
    python scripts/download/download_sf311.py --since 2018-01-01 --estimate-only
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    ),
)

from scripts.download._common import (
    DownloadBlocked,
    build_where,
    preflight,
    socrata_count,
    socrata_pages,
    write_jsonl_pages,
    fail,
)

from scripts.utils.manifest import human_bytes, record
from scripts.utils.paths import load_config, p, ensure_dir
from scripts.utils.logging_setup import get_logger


log = get_logger("download.sf311")

DATASET = "sf311"

# Approximate size per row for the projected columns.
BYTES_PER_ROW = 420


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Download SF 311 Cases from Socrata."
    )

    parser.add_argument(
        "--since",
        default=None,
        help="YYYY-MM-DD floor on requested_datetime",
    )

    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate the download size; do not download",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Download only this many pages (useful for testing)",
    )

    parser.add_argument(
        "--app-token",
        default=os.environ.get("SOCRATA_APP_TOKEN"),
        help="Optional Socrata app token",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------

    cfg = load_config()["datasets"][DATASET]
    sc = cfg["socrata"]

    # Default to the configured date.
    since = args.since or sc.get("min_requested_datetime")

    where = build_where(
        min_datetime={
            "requested_datetime": since
        }
        if since
        else None
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    # Your YAML does NOT have:
    #
    #     expected.approx_rows
    #
    # It has:
    #
    #     expected.approx_rows_full
    #     expected.approx_rows_since_2018
    #
    # So choose the appropriate estimate.
    # ------------------------------------------------------------

    if since:
        configured_rows = cfg["expected"].get(
            "approx_rows_since_2018",
            cfg["expected"].get("approx_rows_full", 0),
        )
    else:
        configured_rows = cfg["expected"].get(
            "approx_rows_full",
            0,
        )

    configured_estimate = configured_rows * BYTES_PER_ROW

    # ------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------

    pre = preflight(
        sc["resource_url"],
        DATASET,
        estimated_bytes=configured_estimate,
    )

    if not pre.get("reachable"):
        fail(
            DATASET,
            sc["resource_url"],
            "Socrata GET",
            pre.get("error", "unreachable"),
            blocked="host_not_allowed"
            in str(pre.get("error", "")),
        )
        return 2

    # ------------------------------------------------------------
    # Ask Socrata for the actual row count
    # ------------------------------------------------------------

    n = socrata_count(
        sc["resource_url"],
        where,
    )

    actual_rows = n if n else configured_rows

    estimated_bytes = actual_rows * BYTES_PER_ROW

    log.info(
        "rows=%s  raw JSONL estimate=%s  parquet estimate=%s",
        f"{n:,}" if n else "unknown",
        human_bytes(estimated_bytes),
        human_bytes(estimated_bytes * 0.18),
    )

    # ------------------------------------------------------------
    # Estimate-only mode
    # ------------------------------------------------------------

    if args.estimate_only:
        return 0

    # ------------------------------------------------------------
    # Destination
    # ------------------------------------------------------------

    dest = ensure_dir(
        p(
            "raw",
            DATASET,
            "sf311_cases.jsonl",
        )
    )

    # ------------------------------------------------------------
    # Download
    # ------------------------------------------------------------

    try:

        rows = write_jsonl_pages(
            socrata_pages(
                sc["resource_url"],
                select=sc["select"],
                where=where,
                order_by=sc["order_by"],
                page_size=sc["page_size"],
                app_token=args.app_token,
                max_pages=args.max_pages,
            ),
            dest,
        )

    except DownloadBlocked as e:

        fail(
            DATASET,
            sc["resource_url"],
            "Socrata GET",
            e,
            blocked=True,
        )

        return 2

    except KeyboardInterrupt:

        log.warning("Download interrupted by user.")

        return 130

    except Exception as e:

        fail(
            DATASET,
            sc["resource_url"],
            "Socrata GET",
            e,
        )

        return 1

    # ------------------------------------------------------------
    # Record manifest
    # ------------------------------------------------------------

    record(
        DATASET,
        dest,
        sc["resource_url"],
        row_count=rows,
        extra={
            "where": where,
            "select": sc["select"],
            "socrata_reported_count": n,
        },
    )

    log.info(
        "SF311 download complete: %s rows -> %s",
        f"{rows:,}",
        dest,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())