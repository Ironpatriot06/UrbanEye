"""
Category mapping.

Two hard rules, both enforced here rather than left to convention:

  1. An unmapped source value is NEVER guessed. It becomes UNMAPPED and is
     reported with a row count so a human can extend the CSV.
  2. UNMAPPED, REVIEW_REQUIRED and OUT_OF_SCOPE rows survive into `incidents`
     (so nothing is lost and the counts stay auditable) but are excluded from
     every ml_ready output.
"""
from __future__ import annotations

import functools
import pandas as pd

from .paths import repo_root, p, ensure_dir

CANONICAL = [
    "ROAD_DAMAGE", "POTHOLE", "GARBAGE_DUMPING", "STREETLIGHT_FAULT",
    "TRAFFIC_SIGNAL_FAULT", "FOOTPATH_DAMAGE", "OPEN_MANHOLE", "DRAINAGE_SEWER",
    "WATER_LEAKAGE_WATERLOGGING", "FALLEN_TREE", "PUBLIC_INFRASTRUCTURE_DAMAGE",
    "SIGNAGE_DAMAGE", "GRAFFITI_VISUAL_POLLUTION", "CONSTRUCTION_OBSTRUCTION",
    "OTHER",
]
SENTINELS = ["REVIEW_REQUIRED", "OUT_OF_SCOPE", "UNMAPPED"]
ALLOWED = set(CANONICAL) | set(SENTINELS)


@functools.lru_cache(maxsize=1)
def load_mapping() -> pd.DataFrame:
    path = repo_root() / "config" / "category_mapping.csv"
    df = pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)
    df = df[df["source_dataset"].str.strip() != ""]
    for c in ("source_dataset", "source_category", "urbaneye_category", "confidence"):
        df[c] = df[c].str.strip()
    bad = sorted(set(df["urbaneye_category"]) - ALLOWED)
    if bad:
        raise ValueError(f"category_mapping.csv contains unknown target categories: {bad}")
    return df


def _lookup(dataset: str) -> tuple[dict, dict]:
    """Return (top_level_map, override_map). Overrides are keyed 'Category|Subcategory'."""
    m = load_mapping()
    m = m[m["source_dataset"] == dataset]
    top, override = {}, {}
    for _, r in m.iterrows():
        key = r["source_category"]
        (override if "|" in key else top)[key] = r["urbaneye_category"]
    return top, override


def map_categories(
    df: pd.DataFrame,
    dataset: str,
    category_col: str,
    subcategory_col: str | None = None,
) -> pd.Series:
    """
    Resolve each row to an UrbanEye+ category.

    Precedence: 'Category|Subcategory' override  >  'Category'  >  UNMAPPED.
    The override tier is what lets SF's 3.49M-row 'Street and Sidewalk Cleaning'
    bucket split correctly instead of being force-mapped to one category.
    """
    top, override = _lookup(dataset)
    cat = df[category_col].astype("string").fillna("")
    if subcategory_col and subcategory_col in df.columns:
        sub = df[subcategory_col].astype("string").fillna("")
        composite = cat.str.cat(sub, sep="|")
        mapped = composite.map(override)
    else:
        mapped = pd.Series(pd.NA, index=df.index, dtype="string")
    mapped = mapped.fillna(cat.map(top))
    return mapped.fillna("UNMAPPED").astype("string")


def report_unmapped(df: pd.DataFrame, dataset: str, category_col: str, mapped: pd.Series) -> pd.DataFrame:
    """
    Append every unmapped source value, with its row count, to
    reports/unmapped_categories.csv. This file is the work queue for extending
    the mapping after a real download.
    """
    mask = mapped.eq("UNMAPPED")
    if not mask.any():
        counts = pd.DataFrame(columns=["source_dataset", "source_category", "row_count"])
    else:
        counts = (
            df.loc[mask, category_col].astype("string").fillna("<NULL>")
            .value_counts().rename_axis("source_category").reset_index(name="row_count")
        )
        counts.insert(0, "source_dataset", dataset)

    out = ensure_dir(p("reports", "unmapped_categories.csv"))
    if out.exists():
        prev = pd.read_csv(out, dtype={"source_category": str})
        prev = prev[prev["source_dataset"] != dataset]
        counts = pd.concat([prev, counts], ignore_index=True)
    counts.sort_values(["source_dataset", "row_count"], ascending=[True, False]).to_csv(out, index=False)
    return counts


def is_trainable(series: pd.Series) -> pd.Series:
    """True only for rows whose category is a real UrbanEye+ incident category."""
    return series.isin(CANONICAL) & ~series.isin(["OTHER"]) | series.eq("OTHER")


def in_scope(series: pd.Series) -> pd.Series:
    """Excludes the three sentinels. This is the ml_ready gate."""
    return series.isin(CANONICAL)
