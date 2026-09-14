#!/usr/bin/env python3
"""
Incremental / resumable Chicago 311 downloader.

Downloads Chicago 311 from Socrata as newline-delimited JSON.

Features:
  - Server-side exclusion of non-incident SR types.
  - Appends to an existing JSONL file.
  - Automatically resumes from the number of rows already downloaded.
  - Safe to stop with Ctrl+C.
  - --max-pages means additional pages from the current checkpoint.

Examples:

  # Start / resume download
  python scripts/download/download_chicago311.py

  # Download 20 pages (1,000,000 rows) and stop
  python scripts/download/download_chicago311.py --max-pages 20

  # Check current server-side row count only
  python scripts/download/download_chicago311.py --estimate-only

  # Download without exclusions
  python scripts/download/download_chicago311.py --keep-excluded
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from urllib.parse import urlencode

import requests

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
    fail,
)

from scripts.utils.manifest import human_bytes, record
from scripts.utils.paths import load_config, p, ensure_dir
from scripts.utils.logging_setup import get_logger


log = get_logger("download.chicago311")

DATASET = "chicago311"

BYTES_PER_ROW = 380

# Socrata page size.
# Your existing configuration should also contain this value.
DEFAULT_PAGE_SIZE = 50000

REQUEST_TIMEOUT = 180

MAX_RETRIES = 5


def count_existing_rows(path: str) -> int:
    """
    Count rows already present in the JSONL file.

    JSONL has exactly one JSON object per line.
    """
    if not os.path.exists(path):
        return 0

    if os.path.getsize(path) == 0:
        return 0

    count = 0

    with open(path, "rb") as f:
        for _ in f:
            count += 1

    return count


def repair_partial_last_line(path: str) -> int:
    """
    If the process was interrupted while writing the last JSON object,
    remove the incomplete final line.

    Returns the resulting row count.
    """

    if not os.path.exists(path):
        return 0

    size = os.path.getsize(path)

    if size == 0:
        return 0

    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        last_byte = f.read(1)

        # Normal JSONL file ends with newline.
        if last_byte == b"\n":
            return count_existing_rows(path)

        # Find the last complete newline in the tail.
        tail_size = min(size, 1024 * 1024)

        f.seek(-tail_size, os.SEEK_END)
        tail = f.read(tail_size)

        last_newline = tail.rfind(b"\n")

        if last_newline == -1:
            # No complete line exists.
            f.truncate(0)
            log.warning(
                "Removed incomplete JSONL contents from %s",
                path,
            )
            return 0

        truncate_position = size - tail_size + last_newline + 1

        f.truncate(truncate_position)

        log.warning(
            "Removed incomplete final JSONL line from %s",
            path,
        )

    return count_existing_rows(path)


def request_page(
    session: requests.Session,
    resource_url: str,
    select: list[str],
    where: str | None,
    order_by: str,
    page_size: int,
    offset: int,
    app_token: str | None,
) -> list[dict]:
    """
    Download one Socrata page with retries.
    """

    params = {
        "$select": ",".join(select),
        "$limit": page_size,
        "$offset": offset,
        "$order": order_by,
    }

    if where:
        params["$where"] = where

    headers = {
        "Accept": "application/json",
        "User-Agent": "UrbanEye-ML-Data-Pipeline/2.0",
    }

    if app_token:
        headers["X-App-Token"] = app_token

    url = f"{resource_url}?{urlencode(params)}"

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = session.get(
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            data = response.json()

            if not isinstance(data, list):
                raise RuntimeError(
                    f"Unexpected Socrata response type: "
                    f"{type(data).__name__}"
                )

            return data

        except KeyboardInterrupt:
            raise

        except Exception as exc:

            if attempt >= MAX_RETRIES:
                raise

            delay = min(2 ** (attempt - 1), 30)

            log.warning(
                "Chicago page offset=%s failed: %s "
                "— retry %s/%s in %ss",
                offset,
                exc,
                attempt,
                MAX_RETRIES,
                delay,
            )

            time.sleep(delay)

    return []


def append_page(path: str, rows: list[dict]) -> None:
    """
    Append one complete page to the JSONL file.

    The page is fully serialized before opening the output file so that
    network failures cannot leave a half-written page.
    """

    payload = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )

    with open(path, "a", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--estimate-only",
        action="store_true",
    )

    ap.add_argument(
        "--keep-excluded",
        action="store_true",
    )

    ap.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help=(
            "Download this many additional pages from the "
            "current checkpoint."
        ),
    )

    ap.add_argument(
        "--app-token",
        default=os.environ.get("SOCRATA_APP_TOKEN"),
    )

    args = ap.parse_args()

    cfg = load_config()["datasets"][DATASET]

    sc = cfg["socrata"]

    page_size = sc.get(
        "page_size",
        DEFAULT_PAGE_SIZE,
    )

    # ---------------------------------------------------------
    # Server-side filtering
    # ---------------------------------------------------------

    where = (
        None
        if args.keep_excluded
        else build_where(
            exclude_values={
                "sr_type": sc.get(
                    "exclude_sr_types",
                    [],
                )
            }
        )
    )

    log.info(
        "server-side filter: %s",
        where or "(none)",
    )

    # ---------------------------------------------------------
    # Destination
    # ---------------------------------------------------------

    dest = ensure_dir(
        p(
            "raw",
            DATASET,
            "chicago311.jsonl",
        )
    )

    # ---------------------------------------------------------
    # Repair / resume
    # ---------------------------------------------------------

    existing_rows = repair_partial_last_line(dest)

    if existing_rows:
        log.info(
            "existing checkpoint: %s rows",
            f"{existing_rows:,}",
        )

    # ---------------------------------------------------------
    # Preflight
    # ---------------------------------------------------------

    expected_rows = cfg["expected"].get(
        "approx_rows_after_exclusions",
        cfg["expected"].get(
            "approx_rows",
            0,
        ),
    )

    pre = preflight(
        sc["resource_url"],
        DATASET,
        estimated_bytes=expected_rows * BYTES_PER_ROW,
    )

    if not pre.get("reachable"):

        fail(
            DATASET,
            sc["resource_url"],
            "Socrata GET",
            pre.get(
                "error",
                "unreachable",
            ),
            blocked="host_not_allowed"
            in str(
                pre.get(
                    "error",
                    "",
                )
            ),
        )

        return 2

    # ---------------------------------------------------------
    # Server-side row count
    # ---------------------------------------------------------

    n = socrata_count(
        sc["resource_url"],
        where,
    )

    est = (
        (n or expected_rows)
        * BYTES_PER_ROW
    )

    log.info(
        "rows after exclusions=%s  raw estimate=%s",
        f"{n:,}" if n else "unknown",
        human_bytes(est),
    )

    if args.estimate_only:
        return 0

    # ---------------------------------------------------------
    # Already complete?
    # ---------------------------------------------------------

    if n is not None and existing_rows >= n:

        log.info(
            "Chicago 311 already complete: %s/%s rows",
            f"{existing_rows:,}",
            f"{n:,}",
        )

        return 0

    # ---------------------------------------------------------
    # Calculate resume offset
    # ---------------------------------------------------------

    offset = existing_rows

    log.info(
        "resuming Chicago 311 from offset=%s "
        "(already downloaded=%s rows)",
        offset,
        f"{existing_rows:,}",
    )

    # ---------------------------------------------------------
    # Download
    # ---------------------------------------------------------

    session = requests.Session()

    pages_downloaded = 0
    total_rows_written = existing_rows

    try:

        while True:

            # Stop after requested number of ADDITIONAL pages.
            if (
                args.max_pages is not None
                and pages_downloaded >= args.max_pages
            ):
                log.info(
                    "stopping at max_pages=%s",
                    args.max_pages,
                )
                break

            # If Socrata gave us a count, don't request beyond it.
            if n is not None and offset >= n:
                log.info(
                    "reached server-reported row count: %s",
                    f"{n:,}",
                )
                break

            log.info(
                "requesting Chicago 311 page: "
                "offset=%s limit=%s",
                offset,
                page_size,
            )

            rows = request_page(
                session=session,
                resource_url=sc["resource_url"],
                select=sc["select"],
                where=where,
                order_by=sc["order_by"],
                page_size=page_size,
                offset=offset,
                app_token=args.app_token,
            )

            if not rows:
                log.info(
                    "Socrata returned no more rows at offset=%s",
                    offset,
                )
                break

            # IMPORTANT:
            # Only update checkpoint AFTER the entire page has been
            # written successfully.
            append_page(
                dest,
                rows,
            )

            page_rows = len(rows)

            offset += page_rows

            total_rows_written += page_rows

            pages_downloaded += 1

            log.info(
                "Chicago 311 progress: %s rows "
                "(%.2f%% of %s)",
                f"{total_rows_written:,}",
                (
                    total_rows_written / n * 100
                    if n
                    else 0
                ),
                f"{n:,}" if n else "unknown",
            )

            # If this was a short page, we're finished.
            if page_rows < page_size:
                log.info(
                    "received final partial page "
                    "(%s rows)",
                    page_rows,
                )
                break

    except KeyboardInterrupt:

        log.info(
            "download interrupted safely with Ctrl+C"
        )

        log.info(
            "checkpoint preserved: %s rows",
            f"{count_existing_rows(dest):,}",
        )

        return 130

    except DownloadBlocked as e:

        fail(
            DATASET,
            sc["resource_url"],
            "Socrata GET",
            e,
            blocked=True,
        )

        return 2

    except Exception as e:

        fail(
            DATASET,
            sc["resource_url"],
            "Socrata GET",
            e,
        )

        return 1

    finally:

        session.close()

    # ---------------------------------------------------------
    # Final verification
    # ---------------------------------------------------------

    final_rows = count_existing_rows(dest)

    log.info(
        "Chicago 311 checkpoint: %s rows -> %s",
        f"{final_rows:,}",
        dest,
    )

    # Only record a normal manifest when we intentionally reach
    # the end / complete the requested operation.
    record(
        DATASET,
        dest,
        sc["resource_url"],
        row_count=final_rows,
        extra={
            "where": where,
            "excluded_sr_types": (
                []
                if args.keep_excluded
                else sc.get(
                    "exclude_sr_types"
                )
            ),
            "socrata_reported_count": n,
            "incremental": True,
            "page_size": page_size,
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())