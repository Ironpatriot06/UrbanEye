#!/usr/bin/env python3
"""
Probe whether 311 photo URLs actually resolve.

The point of this script: a populated `Media URL` column proves a URL was once
recorded. It does not prove the image still exists, is publicly readable, or is
an image at all. SF's photos are hosted on external SF311 infrastructure, some
records are 17 years old, and the column carries no cached-contents statistics in
the portal metadata, so its fill rate cannot even be read without downloading.
Everything downstream that assumes "SF gives us images" rests on this check.

It samples rather than downloads everything (per the image policy), issues HEAD
requests, and reports the full status distribution rather than a pass/fail.

  python scripts/validation/probe_image_urls.py --dataset sf311 --sample 200
  python scripts/validation/probe_image_urls.py --dataset boston311 --column image_path_after
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import requests

from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import ensure_dir, load_config, p

log = get_logger("validation.image_urls")


def probe_one(url: str, timeout: int, ua: str, method: str) -> dict:
    r = {"url": url, "status": None, "content_type": None, "content_length": None,
         "error": None, "redirected": False, "auth_required": False}
    try:
        resp = requests.request(method, url, timeout=timeout, allow_redirects=True,
                                headers={"User-Agent": ua})
        r["status"] = resp.status_code
        r["content_type"] = resp.headers.get("Content-Type")
        cl = resp.headers.get("Content-Length")
        r["content_length"] = int(cl) if cl and cl.isdigit() else None
        r["redirected"] = len(resp.history) > 0
        r["auth_required"] = resp.status_code in (401, 403)
        if resp.status_code == 405 and method == "HEAD":
            g = requests.get(url, timeout=timeout, stream=True,
                             headers={"User-Agent": ua, "Range": "bytes=0-1023"})
            r["status"] = g.status_code
            r["content_type"] = g.headers.get("Content-Type")
            r["method_fallback"] = "GET"
            g.close()
        if "host_not_allowed" in str(resp.headers.get("x-deny-reason", "")):
            r["error"] = "BLOCKED_BY_EGRESS_PROXY"
    except requests.RequestException as e:
        r["error"] = type(e).__name__ + ": " + str(e)[:200]
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sf311", choices=["sf311", "boston311"])
    ap.add_argument("--column", default="image_path", choices=["image_path", "image_path_after"])
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--input", default=None)
    args = ap.parse_args()

    cfg = load_config()["images"]["url_probe"]
    n_sample = args.sample or (cfg["sf_media_url_sample"] if args.dataset == "sf311"
                               else cfg["boston_photo_sample"])

    src = __import__("pathlib").Path(args.input) if args.input \
        else p("processed", "incidents", f"{args.dataset}.parquet")
    if not src.exists():
        log.error("not found: %s — run the corresponding process_*.py first", src)
        return 2

    df = pd.read_parquet(src, columns=[args.column])
    total_rows = len(df)
    urls = df[args.column].dropna().astype(str)
    urls = urls[urls.str.startswith(("http://", "https://"))]
    n_non_empty = len(urls)
    fill_rate = n_non_empty / total_rows if total_rows else 0.0

    log.info("%s.%s: %d/%d non-empty (%.2f%%)", args.dataset, args.column,
             n_non_empty, total_rows, fill_rate * 100)

    if n_non_empty == 0:
        report = {"dataset": args.dataset, "column": args.column, "total_rows": total_rows,
                  "non_empty_urls": 0, "fill_rate": 0.0, "tested": 0,
                  "conclusion": ("No photo URLs present at all. Any plan that depends on images "
                                 "from this source is unsupported by the data.")}
        _write(report, args)
        return 0

    sample = urls.sample(min(n_sample, n_non_empty), random_state=42).tolist()
    log.info("probing %d URLs with %s, %d workers", len(sample), cfg["method"], cfg["max_workers"])

    results = []
    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as ex:
        futs = [ex.submit(probe_one, u, cfg["timeout_seconds"], cfg["user_agent"], cfg["method"])
                for u in sample]
        for f in as_completed(futs):
            results.append(f.result())

    status_dist = Counter(str(r["status"]) for r in results)
    ctype_dist = Counter((r["content_type"] or "unknown").split(";")[0] for r in results)
    ok = [r for r in results if r["status"] and 200 <= r["status"] < 300]
    is_image = [r for r in ok if (r["content_type"] or "").startswith("image/")]
    blocked = [r for r in results if r.get("error") == "BLOCKED_BY_EGRESS_PROXY"]
    errors = Counter(r["error"].split(":")[0] for r in results if r["error"])
    sizes = [r["content_length"] for r in ok if r["content_length"]]

    # Control probe. If every image URL fails we cannot tell whether the images
    # are gone or our own network is restricted — and those call for opposite
    # decisions. A probe against a host known to be reachable disambiguates it.
    control = probe_one("https://pypi.org/simple/", cfg["timeout_seconds"], cfg["user_agent"], "HEAD")
    control_ok = bool(control["status"] and control["status"] < 500)

    if blocked:
        conclusion = (f"INCONCLUSIVE — {len(blocked)}/{len(results)} probes were refused by the "
                      f"local egress proxy, not by the host. Re-run from a network that permits "
                      f"the image host before drawing any conclusion.")
    elif not ok and not control_ok:
        conclusion = ("INCONCLUSIVE — every image probe failed AND the control probe failed, so "
                      "this is a local network restriction, not evidence about the images. "
                      "Re-run from an unrestricted network. Do NOT record this as 'images are dead'.")
    elif not ok:
        conclusion = ("All probes failed while the control host was reachable, so the failure is "
                      "attributable to the image host, not the network. Do not plan on images "
                      "from this source until this is explained.")
    else:
        rate = len(ok) / len(results)
        img_rate = len(is_image) / len(ok) if ok else 0
        verdict = "worth downloading" if rate > 0.8 and img_rate > 0.8 else \
                  "NOT worth bulk downloading — too high a failure rate"
        conclusion = (f"{len(ok)}/{len(results)} resolved ({rate:.1%}); {len(is_image)} returned an "
                      f"image content-type ({img_rate:.1%} of successes). Verdict: {verdict}. "
                      f"Projected retrievable images across the full column: "
                      f"~{int(n_non_empty * rate * img_rate):,}.")

    report = {
        "dataset": args.dataset, "column": args.column,
        "total_rows": int(total_rows),
        "non_empty_urls": int(n_non_empty),
        "fill_rate_pct": round(fill_rate * 100, 3),
        "urls_tested": len(results),
        "successful": len(ok),
        "failed": len(results) - len(ok),
        "returned_image_content_type": len(is_image),
        "auth_required": sum(1 for r in results if r["auth_required"]),
        "redirected": sum(1 for r in results if r["redirected"]),
        "http_status_distribution": dict(status_dist),
        "content_type_distribution": dict(ctype_dist),
        "error_types": dict(errors),
        "median_image_bytes": int(pd.Series(sizes).median()) if sizes else None,
        "estimated_full_download_gb": (round(n_non_empty * pd.Series(sizes).median() / 1024**3, 2)
                                       if sizes else None),
        "blocked_by_local_proxy": len(blocked),
        "conclusion": conclusion,
        "licence_note": ("Photographs are submitted by members of the public. Even under the "
                         "dataset's PDDL dedication, treat faces, number plates and house numbers "
                         "as a privacy obligation before using these images for training."),
    }
    _write(report, args)
    log.info("conclusion: %s", conclusion)
    return 0


def _write(report: dict, args) -> None:
    dest = ensure_dir(p("reports", f"image_url_probe_{args.dataset}_{args.column}.json"))
    dest.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", dest)


if __name__ == "__main__":
    raise SystemExit(main())
