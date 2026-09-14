#!/usr/bin/env python3
"""
Download Boston 311 yearly CSVs from the Analyze Boston CKAN portal.

Default:
    python scripts/download/download_boston311.py

Specific years:
    python scripts/download/download_boston311.py --years 2019 2020 2021

Estimate only:
    python scripts/download/download_boston311.py --estimate-only

Downloads are written to .part files first and renamed to .csv only after
successful completion. This prevents interrupted downloads from being treated
as valid datasets.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

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
    preflight,
    session,
    _request,
    stream_to_file,
    fail,
)

from scripts.utils.manifest import human_bytes, record
from scripts.utils.paths import load_config, p, ensure_dir
from scripts.utils.logging_setup import get_logger


log = get_logger("download.boston311")

DATASET = "boston311"


def resolve_urls(years: list[int], cfg: dict) -> dict[int, str]:
    """
    Resolve current Boston CKAN resource URLs.

    CKAN is queried first because Boston may change the actual download URL
    when a resource is re-uploaded.
    """

    ck = cfg["ckan"]
    urls: dict[int, str] = {}

    try:
        r = _request(session(), "GET", ck["package_api"])
        resources = r.json()["result"]["resources"]

        for res in resources:
            name = (res.get("name") or "").upper()

            for year in years:
                if (
                    str(year) in name
                    and "NEW SYSTEM" not in name
                    and res.get("url")
                ):
                    urls[year] = res["url"]

        log.info(
            "CKAN resolved %d/%d yearly resources",
            len(urls),
            len(years),
        )

    except DownloadBlocked:
        raise

    except Exception as e:
        log.warning(
            "CKAN package_show failed (%s); using configured resource IDs",
            e,
        )

    return urls


def count_rows(path: Path) -> int:
    """Count CSV data rows, excluding the header."""

    with open(
        path,
        newline="",
        encoding="utf-8",
        errors="replace",
    ) as fh:
        return max(
            0,
            sum(1 for _ in csv.reader(fh)) - 1,
        )


def download_year(year: int, url: str) -> bool:
    """
    Download one Boston 311 yearly CSV.

    The file is first written to .part.
    It becomes a real .csv only after successful completion.
    """

    dest = ensure_dir(
        p(
            "raw",
            DATASET,
            f"boston311_{year}.csv",
        )
    )

    part = Path(str(dest) + ".part")

    # Completed file already exists.
    if dest.exists():
        log.info(
            "%s already exists, skipping",
            dest.name,
        )
        return True

    # Never trust an old partial file.
    if part.exists():
        log.warning(
            "Removing incomplete partial file: %s",
            part,
        )
        part.unlink()

    log.info(
        "Downloading Boston 311 %d",
        year,
    )

    log.info(
        "URL: %s",
        url,
    )

    try:
        stream_to_file(
            url,
            part,
            DATASET,
        )

        rows = count_rows(part)

        if rows <= 0:
            raise RuntimeError(
                f"{year}: downloaded file contains no data rows"
            )

        # Atomic-ish completion step:
        # only expose the .csv after successful validation.
        part.replace(dest)

        record(
            DATASET,
            dest,
            url,
            row_count=rows,
            extra={
                "year": year,
                "download_status": "complete",
            },
        )

        log.info(
            "Boston 311 %d complete: %s rows -> %s",
            year,
            f"{rows:,}",
            dest,
        )

        return True

    except KeyboardInterrupt:
        log.info(
            "Download interrupted for Boston 311 %d",
            year,
        )

        if part.exists():
            part.unlink()

        log.info(
            "Incomplete %d download removed",
            year,
        )

        return False

    except DownloadBlocked as e:
        if part.exists():
            part.unlink()

        fail(
            DATASET,
            url,
            "GET",
            e,
            blocked=True,
        )

        return False

    except Exception as e:
        if part.exists():
            part.unlink()

        fail(
            DATASET,
            url,
            "GET",
            e,
        )

        return False


def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=None,
    )

    ap.add_argument(
        "--estimate-only",
        action="store_true",
    )

    args = ap.parse_args()

    cfg = load_config()["datasets"][DATASET]

    # Use configured years unless explicitly supplied.
    years = args.years or cfg["years"]

    log.info(
        "Boston 311 years: %s",
        years,
    )

    try:
        urls = resolve_urls(
            years,
            cfg,
        )

    except DownloadBlocked as e:
        fail(
            DATASET,
            cfg["ckan"]["package_api"],
            "CKAN package_show",
            e,
            blocked=True,
        )
        return 2

    if not urls:
        fail(
            DATASET,
            cfg["ckan"]["package_api"],
            "CKAN package_show",
            "no resource URLs resolved",
        )
        return 1

    log.info(
        "Resolved years: %s",
        sorted(urls),
    )

    # Estimate using first available resource.
    first_year = sorted(urls)[0]
    first_url = urls[first_year]

    pre = preflight(
        first_url,
        DATASET,
        estimated_bytes=180 * 1024**2,
    )

    if not pre.get("reachable"):
        fail(
            DATASET,
            first_url,
            "GET",
            pre.get(
                "error",
                "unreachable",
            ),
            blocked="host_not_allowed"
            in str(pre.get("error", "")),
        )

        return 2

    per_year = (
        pre.get("content_length")
        or 180 * 1024**2
    )

    log.info(
        "estimated total raw: %s across %d years",
        human_bytes(
            per_year * len(urls)
        ),
        len(urls),
    )

    if args.estimate_only:
        return 0

    ok = 0
    failed = 0

    for year in sorted(urls):

        log.info(
            "========== Boston 311 %d ==========",
            year,
        )

        if download_year(
            year,
            urls[year],
        ):
            ok += 1
        else:
            failed += 1

            # Stop on failure so we don't hide a systemic
            # Boston/S3 problem behind many failures.
            break

    log.info(
        "Boston 311: %d ok, %d failed",
        ok,
        failed,
    )

    return (
        0
        if failed == 0
        else (2 if ok == 0 else 1)
    )


if __name__ == "__main__":
    raise SystemExit(main())