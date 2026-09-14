#!/usr/bin/env python3
"""
Build the supervised duplicate-detection pair table.

POSITIVES come exclusively from Chicago's PARENT_SR_NUMBER pointer (~706,730
non-null). That is a real municipal determination that two service requests
describe the same physical incident — the only such labelling in open civic data.

SF's 'Case is a Duplicate' note is a WEAK signal: it says a case was a duplicate
but not of what, so it yields no pair and is excluded from positives. It is
counted in the report so the loss is visible.

NEGATIVES are constructed, never assumed. Four documented strategies, each
tagged in negative_strategy so the mix can be audited and re-weighted:

  N1 same_category_far_apart   same category, >2 km apart, any time
                               -> teaches that category alone is not identity
  N2 same_category_time_shifted same category, within 200 m, >90 days apart
                               -> the HARDEST negatives: same defect location,
                                  different reporting episode. Without these a
                                  model degenerates into a proximity detector.
  N3 different_category_nearby  different category, within 200 m, within 7 days
                               -> teaches that co-location is not identity
  N4 random_unrelated           uniform random pair
                               -> easy negatives for calibration

image_similarity is NULL throughout and that is not an oversight: the duplicate
labels come from Chicago, and Chicago publishes no photographs. There is no
public data with which to supervise image-based duplicate matching.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.preprocess._ml_common import _cut_points
from scripts.utils.cleaning import haversine_m
from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import ensure_dir, load_config, p
from scripts.utils.schema import DUPLICATE_PAIR_SCHEMA, conform, assert_null_fields_empty

log = get_logger("preprocess.duplicates")

NEAR_M = 200.0
FAR_M = 2000.0
SHIFT_DAYS = 90
CO_LOCATED_DAYS = 7


def _pair_features(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    dist = haversine_m(a["latitude"].to_numpy(float), a["longitude"].to_numpy(float),
                       b["latitude"].to_numpy(float), b["longitude"].to_numpy(float))
    both_geo = a["latitude"].notna().to_numpy() & b["latitude"].notna().to_numpy()
    dist = np.where(both_geo, dist, np.nan)
    dt = (b["reported_at"].to_numpy() - a["reported_at"].to_numpy())
    hours = pd.to_timedelta(dt).total_seconds() / 3600.0
    return pd.DataFrame({
        "incident_a": a["incident_id"].to_numpy(),
        "incident_b": b["incident_id"].to_numpy(),
        "distance_meters": dist,
        "time_difference_hours": np.abs(hours),
        "category_match": (a["category"].to_numpy() == b["category"].to_numpy()),
        # Chicago has no free-text description, so no lexical similarity is
        # computable for these pairs. NULL rather than a fabricated 0.0.
        "text_similarity": np.nan,
        "source_dataset": "chicago311",
    })


def _components(pairs: pd.DataFrame) -> dict[str, int]:
    """
    Union-find over the POSITIVE pairs: every incident that is transitively
    linked by a parent pointer lands in one component.

    Grouping by `incident_a` alone is not enough. Chicago's pointers form chains
    (a child can itself be a parent), so two pairs can share an incident without
    sharing an `incident_a`, and splitting on `incident_a` would put the same
    physical incident in two different splits.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:          # path compression
            parent[x], x = root, parent[x]
        return root

    pos = pairs[pairs["same_incident"].fillna(False)]
    for a, b in zip(pos["incident_a"].to_numpy(), pos["incident_b"].to_numpy()):
        ra, rb = find(str(a)), find(str(b))
        if ra != rb:
            parent[rb] = ra
    return {k: find(k) for k in list(parent)}


def incident_splits(pos: pd.DataFrame, chi: pd.DataFrame) -> tuple[pd.Series, dict]:
    """
    Assign every Chicago incident to train/val/test, group-aware and chronological.

    The previous implementation assigned splits by the order in which
    `incident_a` first appeared in the concatenated frame. Positives were
    concatenated first, so they filled train and **validation and test ended up
    100% negative** — 231,955 positives in train, 0 in val, 0 in test. A model
    scored on that would look perfect by always answering "not a duplicate", and
    PR-AUC would be undefined. The negative-strategy mix was skewed the same way:
    the test split was entirely N4 random-unrelated, the easiest negatives here.

    What replaces it:

      * every incident maps to a connected component of the positive-pair graph,
        so a duplicate cluster can never straddle a split;
      * a component's timestamp is the earliest report among its members — a
        single canonical value, so the component lands in exactly one split
        whichever side of a pair it appears on;
      * cut points are chronological, pair-weighted, and use the same config
        fractions as every other table;
      * the assignment covers the WHOLE incident pool, so negatives can be drawn
        inside a split rather than drawn first and discarded afterwards.
    """
    comp = _components(pos)
    ids = chi["incident_id"].astype(str)
    group = ids.map(comp).fillna(ids)
    t = pd.to_datetime(chi["reported_at"], utc=True, errors="coerce")
    comp_time = pd.Series(t.to_numpy(), index=group.to_numpy()).groupby(level=0).min()

    sp = load_config()["splits"]["tabular"]
    pair_time = pos["incident_a"].astype(str).map(comp).fillna(
        pos["incident_a"].astype(str)).map(comp_time)
    cut_a, cut_b, mode = _cut_points(pair_time, sp)

    sog = pd.Series("test", index=comp_time.index, dtype="string")
    sog[comp_time <= cut_a] = "train"
    sog[(comp_time > cut_a) & (comp_time <= cut_b)] = "val"
    sog[comp_time.isna()] = "unassigned"

    split = pd.Series(group.map(sog).to_numpy(), index=ids.to_numpy(), dtype="string")
    meta = {
        "strategy": "connected_component_chronological",
        "group_key": "connected component of the positive-pair graph",
        "cut_mode": mode, "train_end": str(cut_a), "val_end": str(cut_b),
        "components": int(comp_time.size),
        "note": ("Negatives are sampled INSIDE each split, so no pair crosses a split "
                 "boundary and none has to be discarded after the fact."),
    }
    return split, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--negative-ratio", type=float, default=1.0,
                    help="negatives per positive (default 1.0)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = p("processed", "incidents", "all_incidents.parquet") if not args.input \
        else __import__("pathlib").Path(args.input)
    if not src.exists():
        log.error("not found: %s — run build_incidents.py first", src)
        return 2

    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(src)
    chi = df[df["source_dataset"] == "chicago311"].copy()
    if chi.empty:
        log.error("no Chicago rows present; Chicago is the only source of duplicate pointers")
        return 2
    chi = chi.set_index("incident_id", drop=False)

    # ---- positives -------------------------------------------------------
    children = chi[chi["parent_incident_id"].notna()]
    valid = children[children["parent_incident_id"].isin(chi.index)]
    orphaned = len(children) - len(valid)
    if orphaned:
        log.warning("%d child rows point at a parent not present in this extract "
                    "(expected when the download was filtered); excluded from positives", orphaned)

    if valid.empty:
        log.error("no resolvable parent pointers found")
        return 1

    parents = chi.loc[valid["parent_incident_id"].to_numpy()]
    pos = _pair_features(parents.reset_index(drop=True), valid.reset_index(drop=True))
    pos["same_incident"] = True
    pos["negative_strategy"] = pd.NA
    log.info("positives from PARENT_SR_NUMBER: %d", len(pos))

    # ---- split assignment (before negatives, so negatives stay inside a split)
    inc_split, split_meta = incident_splits(pos, chi.reset_index(drop=True))
    pos["split"] = pos["incident_a"].astype(str).map(inc_split)
    pos = pos[pos["split"].notna()].copy()

    # ---- negatives -------------------------------------------------------
    known = set(zip(pos["incident_a"], pos["incident_b"])) | \
        set(zip(pos["incident_b"], pos["incident_a"]))
    chi_pool = chi.reset_index(drop=True)
    chi_pool["_split"] = chi_pool["incident_id"].astype(str).map(inc_split)
    negs = []

    def sample_random(pool: pd.DataFrame, n: int):
        ia = rng.integers(0, len(pool), size=n)
        ib = rng.integers(0, len(pool), size=n)
        keep = ia != ib
        return pool.iloc[ia[keep]].reset_index(drop=True), pool.iloc[ib[keep]].reset_index(drop=True)

    def sample_colocated(pool: pd.DataFrame, n: int):
        """
        Draw pairs that are already spatially close, via grid blocking.

        Uniform random sampling almost never lands two incidents within 200 m of
        each other — in a city-sized dataset the probability is order 1e-5 — so
        rejection sampling cannot produce the near-miss negatives (N2, N3) that
        actually teach the model anything. Blocking on a ~200 m grid cell first
        makes them cheap to find, and is the same technique the duplicate
        detector itself will use at inference time.

        Sampling is vectorised: a cell is chosen uniformly, then two members of
        it uniformly. Building the same sample row by row with .iloc took longer
        than the entire rest of the pipeline put together.
        """
        geo = pool[pool["latitude"].notna() & pool["longitude"].notna()]
        if geo.empty:
            return geo, geo
        cell = 0.0018  # ~200 m at mid-latitudes
        cx = (geo["latitude"].to_numpy(float) / cell).round().astype(np.int64)
        cy = (geo["longitude"].to_numpy(float) / cell).round().astype(np.int64)
        codes = pd.factorize(pd.Series(cx * 4_000_003 + cy))[0]

        order = np.argsort(codes, kind="stable")
        sizes = np.bincount(codes)
        starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        usable = np.flatnonzero(sizes > 1)
        if usable.size == 0:
            return geo.head(0), geo.head(0)

        g = usable[rng.integers(0, usable.size, size=n)]
        sz, st = sizes[g], starts[g]
        i = rng.integers(0, sz)
        j = rng.integers(0, sz)
        keep = i != j
        a_pos = order[st[keep] + i[keep]]
        b_pos = order[st[keep] + j[keep]]
        if a_pos.size == 0:
            return geo.head(0), geo.head(0)
        return (geo.iloc[a_pos].reset_index(drop=True),
                geo.iloc[b_pos].reset_index(drop=True))

    STRATEGIES = [
        ("N1_same_category_far_apart", 12, sample_random,
         lambda f: f["category_match"] & (f["distance_meters"] > FAR_M)),
        ("N2_same_category_time_shifted", 40, sample_colocated,
         lambda f: f["category_match"] & (f["distance_meters"] < NEAR_M)
                   & (f["time_difference_hours"] > SHIFT_DAYS * 24)),
        ("N3_different_category_nearby", 40, sample_colocated,
         lambda f: (~f["category_match"]) & (f["distance_meters"] < NEAR_M)
                   & (f["time_difference_hours"] < CO_LOCATED_DAYS * 24)),
        ("N4_random_unrelated", 2, sample_random,
         lambda f: pd.Series(True, index=f.index)),
    ]

    # Negatives are drawn per split, in proportion to that split's positives, so
    # every split ends up with both classes and a comparable strategy mix.
    for sname, pool in chi_pool.groupby("_split", observed=True):
        n_pos = int((pos["split"] == sname).sum())
        if n_pos == 0 or len(pool) < 2:
            continue
        per_strategy = max(1, int(n_pos * args.negative_ratio) // len(STRATEGIES))
        for strategy, oversample, sampler, predicate in STRATEGIES:
            a, b = sampler(pool, per_strategy * oversample)
            if len(a) == 0:
                log.warning("  %-8s %-32s      0 pairs (no candidates)", sname, strategy)
                continue
            feats = _pair_features(a, b)
            sel = feats[predicate(feats).fillna(False)].head(per_strategy).copy()
            sel["same_incident"] = False
            sel["negative_strategy"] = strategy
            # Never emit a "negative" that is actually a known positive pair.
            sel = sel[[(x, y) not in known for x, y in zip(sel["incident_a"], sel["incident_b"])]]
            sel["split"] = sname
            log.info("  %-6s %-32s %6d pairs", sname, strategy, len(sel))
            negs.append(sel)

    pairs = pd.concat([pos] + negs, ignore_index=True)
    pairs["image_similarity"] = pd.NA      # no images exist for these incidents
    log.info("split: %s", {k: int(v) for k, v in pairs["split"].value_counts().items()})

    pairs = conform(pairs, DUPLICATE_PAIR_SCHEMA)
    assert_null_fields_empty(pairs, DUPLICATE_PAIR_SCHEMA)

    dest = ensure_dir(p("processed", "duplicates", "duplicate_pairs.parquet"))
    pairs.to_parquet(dest, index=False)

    weak_sf = int(df[(df["source_dataset"] == "sf311") & (df["is_duplicate"] == True)].shape[0])  # noqa: E712
    summary = {
        "positives": int(pairs["same_incident"].sum()),
        "negatives": int((~pairs["same_incident"].fillna(False)).sum()),
        "negative_strategy_counts": {k: int(v) for k, v in pairs["negative_strategy"].value_counts().items()},
        "positive_source": "Chicago 311 PARENT_SR_NUMBER (municipal duplicate determination)",
        "orphaned_parent_pointers_excluded": int(orphaned),
        "sf_weak_duplicates_not_usable_as_pairs": weak_sf,
        "sf_weak_note": ("SF flags that a case WAS a duplicate but not of WHAT. No pair can be "
                         "formed, so these are excluded from positives rather than guessed at."),
        "text_similarity_null_reason": "Chicago 311 publishes no free-text description field.",
        "image_similarity_null_reason": "Chicago 311 publishes no photographs. No public data can supervise this.",
        "thresholds": {"near_m": NEAR_M, "far_m": FAR_M,
                       "time_shift_days": SHIFT_DAYS, "co_located_days": CO_LOCATED_DAYS},
        "split": split_meta,
        "split_counts": {str(k): int(v) for k, v in pairs["split"].value_counts().items()},
        "class_balance_per_split": {
            str(s): {"positive": int(g["same_incident"].fillna(False).sum()),
                     "negative": int((~g["same_incident"].fillna(False)).sum())}
            for s, g in pairs.groupby("split")},
        "evaluation_warning": (
            "The negatives are CONSTRUCTED, so the 1:1 class balance here is a design "
            "choice, not the deployment prevalence — in production the overwhelming "
            "majority of candidate pairs are not duplicates. Worse, the sampling rules "
            "themselves correlate with the label: N1 negatives are >2 km apart and N2 "
            "negatives are >90 days apart by construction, so distance and time "
            "difference separate the classes partly by fiat. Report metrics per "
            "negative_strategy and treat any pooled PR-AUC as an upper bound."),
    }
    ensure_dir(p("reports", "duplicate_pairs_summary.json")).write_text(json.dumps(summary, indent=2))
    log.info("wrote %s (%d pairs)", dest, len(pairs))
    log.info("summary: %s", {k: summary[k] for k in ("positives", "negatives")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
