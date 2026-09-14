"""
Shared download machinery.

Three things this handles that a plain `requests.get` does not:

  * Socrata pagination with a stable sort key. Socrata's default ordering is not
    guaranteed stable across pages, so a naive $offset walk silently drops and
    duplicates rows on a dataset that is being updated nightly. Every paged read
    here pins $order.
  * Resumable writes. Multi-GB downloads over a flaky link should not restart
    from zero.
  * A preflight that refuses to start a download it cannot finish, and reports
    the estimate instead.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import urlencode

import requests

from ..utils.logging_setup import get_logger
from ..utils.manifest import human_bytes, free_space_bytes, record_failure
from ..utils.paths import ensure_dir

log = get_logger("download")

DEFAULT_TIMEOUT = (15, 120)   # (connect, read)
MAX_RETRIES = 5
BACKOFF = 2.0
UA = "UrbanEyePlus-DataPipeline/1.0 (academic research)"


class DownloadBlocked(RuntimeError):
    """Raised when the network refuses the host. Distinct from a transient failure."""


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    return s


def _request(sess: requests.Session, method: str, url: str, **kw):
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = sess.request(method, url, timeout=DEFAULT_TIMEOUT, **kw)
            if r.status_code == 403 and "host_not_allowed" in str(r.headers.get("x-deny-reason", "")):
                raise DownloadBlocked(
                    f"Egress proxy refused host for {url} "
                    f"(x-deny-reason: {r.headers.get('x-deny-reason')}). "
                    "This host must be added to the environment's allowed-domains list."
                )
            if r.status_code in (429, 500, 502, 503, 504):
                wait = BACKOFF ** attempt
                log.warning("HTTP %s from %s — retry %d/%d in %.0fs", r.status_code, url, attempt, MAX_RETRIES, wait)
                time.sleep(wait)
                last = f"HTTP {r.status_code}"
                continue
            r.raise_for_status()
            return r
        except DownloadBlocked:
            raise
        except requests.RequestException as e:
            last = str(e)
            if attempt == MAX_RETRIES:
                break
            wait = BACKOFF ** attempt
            log.warning("%s — retry %d/%d in %.0fs", e, attempt, MAX_RETRIES, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {url} ({last})")


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def preflight(url: str, dataset: str, *, estimated_bytes: int | None = None, safety_factor: float = 3.0) -> dict:
    """
    Check reachability and disk BEFORE committing to a long download.

    safety_factor covers raw + decompressed + Parquet + temp working space.
    """
    info = {"url": url, "reachable": None, "content_length": None, "free_bytes": free_space_bytes()}
    try:
        r = _request(session(), "HEAD", url, allow_redirects=True)
        info["reachable"] = True
        cl = r.headers.get("Content-Length")
        info["content_length"] = int(cl) if cl and cl.isdigit() else None
    except DownloadBlocked as e:
        info["reachable"] = False
        info["error"] = str(e)
        record_failure(dataset, url, "HEAD", str(e), status="BLOCKED")
        return info
    except Exception as e:  # noqa: BLE001
        info["reachable"] = False
        info["error"] = str(e)
        return info

    need = (info["content_length"] or estimated_bytes or 0) * safety_factor
    info["estimated_peak_bytes"] = int(need)
    info["sufficient_disk"] = need == 0 or need < info["free_bytes"]
    log.info(
        "preflight %s: reachable=%s size=%s free=%s peak_estimate=%s ok=%s",
        dataset, info["reachable"],
        human_bytes(info["content_length"] or 0), human_bytes(info["free_bytes"]),
        human_bytes(need), info.get("sufficient_disk"),
    )
    return info


# ---------------------------------------------------------------------------
# Streaming file download (resumable)
# ---------------------------------------------------------------------------
def stream_to_file(url: str, dest: Path, dataset: str, *, resume: bool = True) -> Path:
    dest = ensure_dir(Path(dest))
    part = dest.with_suffix(dest.suffix + ".part")
    sess = session()
    headers = {}
    mode = "wb"
    if resume and part.exists():
        headers["Range"] = f"bytes={part.stat().st_size}-"
        mode = "ab"
        log.info("resuming %s at %s", dest.name, human_bytes(part.stat().st_size))
    r = _request(sess, "GET", url, stream=True, headers=headers, allow_redirects=True)
    total = 0
    with open(part, mode) as fh:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)
                total += len(chunk)
    part.rename(dest)
    log.info("wrote %s (%s)", dest, human_bytes(dest.stat().st_size))
    return dest


# ---------------------------------------------------------------------------
# Socrata
# ---------------------------------------------------------------------------
def socrata_count(resource_url: str, where: str | None = None) -> int | None:
    q = {"$select": "count(1) AS n"}
    if where:
        q["$where"] = where
    try:
        r = _request(session(), "GET", f"{resource_url}?{urlencode(q)}")
        return int(r.json()[0]["n"])
    except DownloadBlocked:
        raise
    except Exception as e:  # noqa: BLE001
        log.warning("count query failed (%s); proceeding without a row estimate", e)
        return None


def socrata_pages(
    resource_url: str,
    *,
    select: list[str] | None = None,
    where: str | None = None,
    order_by: str,
    page_size: int = 50000,
    app_token: str | None = None,
    max_pages: int | None = None,
) -> Iterator[list[dict]]:
    """
    Yield pages of records.

    $order is mandatory, not optional — see module docstring.
    """
    sess = session()
    if app_token:
        sess.headers["X-App-Token"] = app_token
    offset, page_no = 0, 0
    while True:
        q = {"$limit": page_size, "$offset": offset, "$order": order_by}
        if select:
            q["$select"] = ",".join(select)
        if where:
            q["$where"] = where
        r = _request(sess, "GET", f"{resource_url}?{urlencode(q)}")
        rows = r.json()
        if not rows:
            return
        page_no += 1
        yield rows
        if len(rows) < page_size:
            return
        offset += page_size
        if max_pages and page_no >= max_pages:
            log.info("stopping at max_pages=%d", max_pages)
            return


def build_where(
    *,
    exclude_values: dict[str, list[str]] | None = None,
    include_values: dict[str, list[str]] | None = None,
    min_datetime: dict[str, str] | None = None,
) -> str | None:
    """Assemble a SoQL $where clause. Filtering server-side is the whole point."""
    clauses: list[str] = []
    for col, vals in (exclude_values or {}).items():
        for v in vals:
            clauses.append(f"{col} != '{v.replace(chr(39), chr(39) * 2)}'")
    for col, vals in (include_values or {}).items():
        if vals:
            joined = ",".join("'" + v.replace("'", "''") + "'" for v in vals)
            clauses.append(f"{col} in({joined})")
    for col, ts in (min_datetime or {}).items():
        clauses.append(f"{col} >= '{ts}T00:00:00.000'")
    return " AND ".join(clauses) if clauses else None


def write_jsonl_pages(pages: Iterator[list[dict]], dest: Path, log_every: int = 10) -> int:
    """Write paged records to newline-delimited JSON. Raw, unmodified."""
    dest = ensure_dir(Path(dest))
    n = 0
    with open(dest, "w", encoding="utf-8") as fh:
        for i, page in enumerate(pages, 1):
            for rec in page:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            if i % log_every == 0:
                log.info("  %s: %,d rows".replace(",d", "d"), dest.name, n)
    log.info("wrote %s (%d rows)", dest, n)
    return n


def fail(dataset: str, url: str, method: str, err: Exception | str, blocked: bool = False) -> None:
    """Record and report a non-acquisition. Never silently substitute a source."""
    status = "BLOCKED" if blocked else "FAILED"
    record_failure(dataset, url, method, str(err), status=status)
    log.error("%s %s: %s", status, dataset, err)
    print(
        f"\n  {status}: {dataset}\n"
        f"    url    : {url}\n"
        f"    method : {method}\n"
        f"    error  : {err}\n"
        f"    -> recorded in data/manifests/{dataset}.FAILED.json\n"
        f"    -> NOT substituted with any other source.\n",
        file=sys.stderr,
    )
