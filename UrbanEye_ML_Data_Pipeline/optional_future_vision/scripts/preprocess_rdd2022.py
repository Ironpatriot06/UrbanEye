#!/usr/bin/env python3
"""
Normalise RDD2022 PASCAL VOC annotations into `vision_samples`, and optionally
emit YOLO label files.

Two correctness points:
  * The official test images ship WITHOUT annotations. They are tagged
    split='official_test_unlabelled' and excluded from every supervised split.
  * D50 ('manhole cover') is NOT mapped to OPEN_MANHOLE. The class marks the
    presence of an intact cover; mapping it to OPEN_MANHOLE would invert the
    label's meaning. It resolves to REVIEW_REQUIRED via category_mapping.csv.

Works on extracted VOC directory trees. If no images are on disk (the default
annotations-only policy), image_present is False and bbox geometry still parses
from the XML, which carries width/height.
"""
from __future__ import annotations
import argparse, os, sys, xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from scripts.utils.logging_setup import get_logger
from scripts.utils.mapping import map_categories, report_unmapped
from scripts.utils.paths import ensure_dir, p
from scripts.utils.schema import VISION_SCHEMA, conform, assert_null_fields_empty

log = get_logger("preprocess.rdd2022")
DATASET = "rdd2022"


def parse_voc(xml_path: Path) -> tuple[list[dict], int, int]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width", "0"))) if size is not None else 0
    h = int(float(size.findtext("height", "0"))) if size is not None else 0
    objs = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None or not name:
            continue
        try:
            objs.append({
                "source_class": name,
                "bbox_xmin": float(bb.findtext("xmin", "nan")),
                "bbox_ymin": float(bb.findtext("ymin", "nan")),
                "bbox_xmax": float(bb.findtext("xmax", "nan")),
                "bbox_ymax": float(bb.findtext("ymax", "nan")),
            })
        except ValueError:
            continue
    return objs, w, h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="extracted RDD2022 root (default data/raw/rdd2022)")
    ap.add_argument("--emit-yolo", action="store_true", help="also write YOLO .txt label files")
    args = ap.parse_args()

    root = Path(args.root) if args.root else p("raw", DATASET)
    xmls = sorted(root.rglob("*.xml"))
    if not xmls:
        log.error("no VOC XML found under %s — run download_rdd2022.py --countries India "
                  "and extract the archive first", root)
        return 2
    log.info("found %d annotation files", len(xmls))

    rows, malformed, no_objects = [], 0, 0
    for xp in xmls:
        try:
            objs, w, h = parse_voc(xp)
        except ET.ParseError:
            malformed += 1
            continue
        if not objs:
            no_objects += 1
        # country = first path segment under root that names a country subset
        parts = [s for s in xp.relative_to(root).parts]
        country = parts[0] if parts else "unknown"
        is_test = any("test" in s.lower() for s in parts)
        img_rel = str(xp.relative_to(root)).replace("/annotations/xmls/", "/images/").replace(".xml", ".jpg")
        img_abs = root / img_rel
        for o in objs:
            rows.append({
                "image_id": f"{DATASET}:{img_rel}",
                "source_dataset": DATASET,
                "image_path": img_rel,
                "image_present": img_abs.exists(),
                "annotation_format": "pascal_voc_xml",
                "image_width": w or pd.NA,
                "image_height": h or pd.NA,
                "country": country,
                "split": "official_test_unlabelled" if is_test else "pool",
                **o,
            })

    if not rows:
        log.error("parsed 0 objects from %d files", len(xmls))
        return 1

    df = pd.DataFrame(rows)
    mapped = map_categories(df, DATASET, "source_class")
    df["class"] = mapped
    report_unmapped(df, DATASET, "source_class", mapped)
    df["severity"] = pd.NA        # no image dataset provides severity

    df = conform(df, VISION_SCHEMA)
    assert_null_fields_empty(df, VISION_SCHEMA)

    out = ensure_dir(p("processed", "vision", f"{DATASET}.parquet"))
    df.to_parquet(out, index=False)
    log.info("wrote %s — %d annotations across %d images",
             out, len(df), df["image_id"].nunique())
    log.info("class distribution:\n%s", df["class"].value_counts().to_string())
    if malformed or no_objects:
        log.warning("malformed XML: %d | annotation files with zero objects: %d", malformed, no_objects)

    if args.emit_yolo:
        n = emit_yolo(df, root)
        log.info("wrote %d YOLO label files", n)
    return 0


def emit_yolo(df: pd.DataFrame, root: Path) -> int:
    """VOC -> YOLO. Only for rows with a real mapped class and known image size."""
    from scripts.utils.mapping import CANONICAL
    usable = df[df["class"].isin(CANONICAL)
                & df["image_width"].notna() & df["image_height"].notna()
                & df["split"].ne("official_test_unlabelled")]
    classes = sorted(usable["class"].unique())
    idx = {c: i for i, c in enumerate(classes)}
    outdir = ensure_dir(p("processed", "vision", "rdd2022_yolo", "labels"))
    outdir.mkdir(parents=True, exist_ok=True)
    (p("processed", "vision", "rdd2022_yolo") / "classes.txt").write_text("\n".join(classes))
    n = 0
    for img, g in usable.groupby("image_path"):
        w, h = float(g["image_width"].iloc[0]), float(g["image_height"].iloc[0])
        if w <= 0 or h <= 0:
            continue
        lines = []
        for _, r in g.iterrows():
            cx = ((r.bbox_xmin + r.bbox_xmax) / 2) / w
            cy = ((r.bbox_ymin + r.bbox_ymax) / 2) / h
            bw = (r.bbox_xmax - r.bbox_xmin) / w
            bh = (r.bbox_ymax - r.bbox_ymin) / h
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                continue
            lines.append(f"{idx[r['class']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if lines:
            fp = outdir / (Path(img).stem + ".txt")
            fp.write_text("\n".join(lines))
            n += 1
    return n


if __name__ == "__main__":
    raise SystemExit(main())
