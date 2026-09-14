#!/usr/bin/env python3
"""
Download RDD2022 from Figshare (article 21431547).

Two things this does that matter:

  1. It reads the licence straight off the Figshare API and writes it into the
     manifest. RDD2022 is marked REVIEW in dataset_config.yaml precisely because
     we could not confirm its licence terms; this is how that gets resolved.
  2. It respects images.download_images. RDD2022 is a multi-GB archive of JPEGs.
     Under the default annotations-only policy we fetch the file LIST and licence
     and stop, because this stage does not need pixels.

  python scripts/download/download_rdd2022.py                    # metadata + licence
  python scripts/download/download_rdd2022.py --countries India  # fetch India archive
  python scripts/download/download_rdd2022.py --estimate-only
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.download._common import DownloadBlocked, preflight, session, _request, stream_to_file, fail
from scripts.utils.manifest import human_bytes, record
from scripts.utils.paths import load_config, p, ensure_dir
from scripts.utils.logging_setup import get_logger

log = get_logger("download.rdd2022")
DATASET = "rdd2022"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="*", default=None,
                    help="country subsets to actually download (default: none, metadata only)")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--force-images", action="store_true",
                    help="override images.download_images=false in config")
    args = ap.parse_args()

    cfg_all = load_config()
    cfg = cfg_all["datasets"][DATASET]
    api = cfg["figshare"]["api_url"]

    pre = preflight(api, DATASET)
    if not pre.get("reachable"):
        fail(DATASET, api, "Figshare API GET", pre.get("error", "unreachable"),
             blocked="host_not_allowed" in str(pre.get("error", "")))
        return 2

    try:
        meta = _request(session(), "GET", api).json()
    except DownloadBlocked as e:
        fail(DATASET, api, "Figshare API GET", e, blocked=True); return 2
    except Exception as e:  # noqa: BLE001
        fail(DATASET, api, "Figshare API GET", e); return 1

    # --- licence capture: the whole reason this dataset is marked REVIEW -------
    lic = meta.get("license") or {}
    licence_info = {
        "name": lic.get("name"), "url": lic.get("url"),
        "figshare_item_id": meta.get("id"), "doi": meta.get("doi"),
        "title": meta.get("title"),
    }
    meta_dir = ensure_dir(p("raw", DATASET, "figshare_article.json"))
    meta_dir.write_text(json.dumps(meta, indent=2))
    (p("raw", DATASET, "LICENCE_AS_PUBLISHED.json")).write_text(json.dumps(licence_info, indent=2))
    log.info("Figshare-declared licence: %s (%s)", licence_info["name"], licence_info["url"])
    print("\n  ACTION REQUIRED: review the licence above, then set")
    print("  datasets.rdd2022.licence.decision in config/dataset_config.yaml")
    print("  from REVIEW to INCLUDE or EXCLUDE.\n")

    files = meta.get("files", [])
    total = sum(f.get("size", 0) for f in files)
    log.info("article has %d files, %s total", len(files), human_bytes(total))
    for f in files:
        log.info("   %-45s %s", f.get("name"), human_bytes(f.get("size", 0)))

    record(DATASET, meta_dir, api, extra={"licence_as_published": licence_info,
                                          "file_listing": [{"name": f.get("name"), "size": f.get("size")} for f in files]})
    if args.estimate_only:
        return 0

    allow = args.force_images or cfg_all["images"]["download_images"]
    wanted = args.countries or []
    if not wanted:
        log.info("no --countries requested: metadata + licence only, no image archives fetched")
        return 0
    if not allow:
        log.warning("images.download_images=false and --force-images not set; refusing to fetch %s", wanted)
        print("  Set images.download_images: true in config/dataset_config.yaml, or pass --force-images.")
        return 0

    ok = 0
    for f in files:
        name = f.get("name", "")
        if not any(c.lower() in name.lower() for c in wanted):
            continue
        url = f.get("download_url")
        dest = ensure_dir(p("raw", DATASET, name))
        try:
            stream_to_file(url, dest, DATASET)
            record(DATASET, dest, url, extra={"figshare_file": f, "licence_as_published": licence_info})
            ok += 1
        except DownloadBlocked as e:
            fail(DATASET, url, "GET", e, blocked=True); return 2
        except Exception as e:  # noqa: BLE001
            fail(DATASET, url, "GET", e)
    log.info("downloaded %d archives", ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
