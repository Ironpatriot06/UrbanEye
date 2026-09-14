#!/usr/bin/env python3
"""
MODEL 3 — build text_category_dataset (text + metadata -> category).

Chronological split. Chicago contributes nothing here: it has no free-text
column at all, which is a property of the source, not a bug.

    python scripts/preprocess/build_text_dataset.py
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.preprocess._ml_common import load_incidents, build_text_dataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incidents", default=None)
    args = ap.parse_args()
    build_text_dataset(load_incidents(args.incidents))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
