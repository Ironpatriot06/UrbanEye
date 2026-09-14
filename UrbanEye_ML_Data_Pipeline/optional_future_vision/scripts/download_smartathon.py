#!/usr/bin/env python3
"""
Acquire the SDAIA Smartathon Theme 1 (Visual Pollution) dataset.

This script is deliberately the most restrictive of the six.

The original competition distribution is a Google Drive link behind HackerEarth
competition terms that we could not read. Roboflow mirrors of the same images
declare CC BY 4.0 — but a mirror cannot grant rights the original publisher did
not. Until someone confirms the original terms, this dataset is REVIEW, and the
script will not pull it into the training pipeline without an explicit override
that is recorded in the manifest.

  python scripts/download/download_smartathon.py --check-licence
  python scripts/download/download_smartathon.py --from-roboflow --workspace W --project P --version 1 \
      --acknowledge-licence-risk
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.download._common import DownloadBlocked, fail, preflight
from scripts.utils.logging_setup import get_logger
from scripts.utils.manifest import record, record_failure
from scripts.utils.paths import ensure_dir, load_config, p

log = get_logger("download.smartathon")
DATASET = "smartathon"

LICENCE_NOTICE = """
  ------------------------------------------------------------------
  LICENCE NOT VERIFIED - Smartathon Theme 1 / Visual Pollution
  ------------------------------------------------------------------
  Original source : https://smartathon.hackerearth.com/
                    (distribution is a Google Drive link under
                     competition terms we were unable to read)
  Mirror          : https://universe.roboflow.com/university-of-jeddah-leayd/-visual-pollution
  Mirror declares : CC BY 4.0

  A mirror cannot grant rights the original publisher did not grant.
  Before this dataset enters the UrbanEye+ training pipeline someone
  must confirm the ORIGINAL terms, in writing, and record the outcome
  in config/dataset_config.yaml (datasets.smartathon.licence).

  Until then its decision stays REVIEW and process_smartathon.py will
  refuse to emit ml_ready outputs from it.
  ------------------------------------------------------------------
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-licence", action="store_true",
                    help="print the licence position and exit (default behaviour)")
    ap.add_argument("--from-roboflow", action="store_true")
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--version", default="1")
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    ap.add_argument("--format", default="voc", choices=["voc", "yolov8", "coco"])
    ap.add_argument("--acknowledge-licence-risk", action="store_true",
                    help="record an explicit, attributed override in the manifest")
    args = ap.parse_args()

    cfg = load_config()["datasets"][DATASET]
    print(LICENCE_NOTICE)

    if args.check_licence or not args.from_roboflow:
        (ensure_dir(p("raw", DATASET, "LICENCE_STATUS.json"))).write_text(json.dumps({
            "decision": cfg["licence"]["decision"],
            "original_source": cfg["landing_page"],
            "mirror": cfg.get("mirror"),
            "mirror_declared_licence": "CC BY 4.0",
            "original_licence": "UNVERIFIED",
            "blocking_reason": "Original competition terms could not be read.",
            "resolved": False,
        }, indent=2))
        log.info("licence status written; no data acquired")
        return 0

    if cfg["licence"]["decision"] != "INCLUDE" and not args.acknowledge_licence_risk:
        record_failure(DATASET, cfg.get("mirror", ""), "roboflow export",
                       "Refused: licence decision is REVIEW and --acknowledge-licence-risk was not passed",
                       status="BLOCKED")
        log.error("refusing to download: licence decision is %s", cfg["licence"]["decision"])
        return 3

    if not (args.workspace and args.project and args.api_key):
        log.error("--workspace, --project and a Roboflow API key are required")
        return 2

    url = (f"https://api.roboflow.com/{args.workspace}/{args.project}/{args.version}"
           f"/{args.format}?api_key=***")
    real = url.replace("***", args.api_key)

    pre = preflight("https://api.roboflow.com/", DATASET)
    if not pre.get("reachable"):
        fail(DATASET, "https://api.roboflow.com/", "GET", pre.get("error", "unreachable"),
             blocked="host_not_allowed" in str(pre.get("error", "")))
        return 2

    from scripts.download._common import stream_to_file
    dest = ensure_dir(p("raw", DATASET, f"smartathon_{args.format}.zip"))
    try:
        stream_to_file(real, dest, DATASET)
    except DownloadBlocked as e:
        fail(DATASET, url, "GET", e, blocked=True)
        return 2
    except Exception as e:  # noqa: BLE001
        fail(DATASET, url, "GET", e)
        return 1

    record(DATASET, dest, url, extra={
        "acquired_from": "roboflow_mirror",
        "mirror_declared_licence": "CC BY 4.0",
        "original_licence": "UNVERIFIED",
        "licence_risk_acknowledged_by": os.environ.get("USER", "unknown"),
        "warning": "Mirror provenance. Original SDAIA terms unconfirmed.",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
