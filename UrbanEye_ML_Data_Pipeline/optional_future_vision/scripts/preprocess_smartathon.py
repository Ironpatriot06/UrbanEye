#!/usr/bin/env python3
"""
Normalise Smartathon Theme 1 bounding boxes into `vision_samples`.

Licence gate: this dataset's decision is REVIEW in dataset_config.yaml because
the original competition terms could not be confirmed. The script will process
the annotations (so class distribution and structure can be inspected) but marks
the output licence_status=UNVERIFIED, and build_ml_datasets.py will not pull it
into a training set while the decision remains REVIEW.

Input is the competition train.csv: class,image_path,name,xmax,xmin,ymax,ymin
(note the unusual max-before-min column order).
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from scripts.utils.logging_setup import get_logger
from scripts.utils.mapping import map_categories, report_unmapped
from scripts.utils.paths import ensure_dir, load_config, p
from scripts.utils.schema import VISION_SCHEMA, conform, assert_null_fields_empty

log = get_logger("preprocess.smartathon")
DATASET = "smartathon"
MIN_INSTANCES_TO_TRAIN = 50


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default=None)
    ap.add_argument("--images-dir", default=None)
    args = ap.parse_args()

    cfg = load_config()["datasets"][DATASET]
    src = Path(args.train_csv) if args.train_csv else p("raw", DATASET, "train.csv")
    if not src.exists():
        log.error("train.csv not found at %s — see download_smartathon.py (licence gate applies)", src)
        return 2

    raw = pd.read_csv(src)
    log.info("read %d annotation rows, %d columns", len(raw), raw.shape[1])

    cols = {c.lower().strip(): c for c in raw.columns}
    need = ["name", "image_path", "xmin", "ymin", "xmax", "ymax"]
    missing = [c for c in need if c not in cols]
    if missing:
        log.error("expected columns missing: %s (found: %s)", missing, list(raw.columns))
        return 1

    images_dir = Path(args.images_dir) if args.images_dir else p("raw", DATASET, "images")
    img_path = raw[cols["image_path"]].astype(str)

    df = pd.DataFrame({
        "image_id": DATASET + ":" + img_path,
        "source_dataset": DATASET,
        "image_path": img_path,
        "image_present": [ (images_dir / n).exists() for n in img_path ],
        "annotation_format": "csv_bbox",
        "source_class": raw[cols["name"]].astype(str).str.strip().str.upper(),
        "bbox_xmin": pd.to_numeric(raw[cols["xmin"]], errors="coerce"),
        "bbox_ymin": pd.to_numeric(raw[cols["ymin"]], errors="coerce"),
        "bbox_xmax": pd.to_numeric(raw[cols["xmax"]], errors="coerce"),
        "bbox_ymax": pd.to_numeric(raw[cols["ymax"]], errors="coerce"),
        "image_width": pd.NA,     # not supplied by the competition CSV
        "image_height": pd.NA,
        "country": "Saudi Arabia",
        "split": "pool",
        "severity": pd.NA,
    })

    degenerate = (df.bbox_xmax <= df.bbox_xmin) | (df.bbox_ymax <= df.bbox_ymin)
    if degenerate.any():
        log.warning("%d degenerate boxes (xmax<=xmin or ymax<=ymin) — nulled, rows retained",
                    int(degenerate.sum()))
        df.loc[degenerate, ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]] = pd.NA

    mapped = map_categories(df, DATASET, "source_class")
    df["class"] = mapped
    report_unmapped(df, DATASET, "source_class", mapped)

    df = conform(df, VISION_SCHEMA)
    assert_null_fields_empty(df, VISION_SCHEMA)

    out = ensure_dir(p("processed", "vision", f"{DATASET}.parquet"))
    df.to_parquet(out, index=False)

    counts = df["source_class"].value_counts()
    log.info("source class distribution:\n%s", counts.to_string())
    starved = counts[counts < MIN_INSTANCES_TO_TRAIN]
    if len(starved):
        log.warning("classes below %d instances — NOT trainable as separate classes: %s",
                    MIN_INSTANCES_TO_TRAIN, dict(starved))

    (p("processed", "vision", f"{DATASET}.notes.json")).write_text(pd.Series({
        "licence_decision": cfg["licence"]["decision"],
        "licence_status": "UNVERIFIED at original source; mirror declares CC BY 4.0",
        "excluded_from_ml_ready_while_review": cfg["licence"]["decision"] != "INCLUDE",
        "starved_classes": {k: int(v) for k, v in starved.items()},
        "annotations": int(len(df)),
        "images": int(df["image_id"].nunique()),
    }).to_json(indent=2))
    log.info("wrote %s — %d annotations across %d images", out, len(df), df["image_id"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
