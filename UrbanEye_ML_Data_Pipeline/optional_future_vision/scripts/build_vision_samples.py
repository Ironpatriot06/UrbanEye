#!/usr/bin/env python3
"""
MODEL 1 — build vision_category_dataset from the preprocessed image datasets.

Combines RDD2022 and Smartathon into one unified class space, but ONLY for
sources whose licence decision is INCLUDE in config/dataset_config.yaml. A
source still marked REVIEW is skipped and the reason is written into the .meta
file rather than being silently dropped.

These rows are NEVER joined to 311 incidents. The only thing image data and
incident data share is the UrbanEye+ category taxonomy.

    python scripts/preprocess/build_vision_samples.py
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.preprocess._ml_common import build_vision_samples, build_severity_schema


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-severity-schema", action="store_true")
    args = ap.parse_args()
    build_vision_samples()
    if not args.skip_severity_schema:
        # Model 2's empty schema lives here because it is the image-side output.
        build_severity_schema()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
