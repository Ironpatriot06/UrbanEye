#!/usr/bin/env python3
"""
Build `incident_geo_features` — the pluggable geospatial feature layer.

This script exists to solve one problem honestly: UrbanEye+ needs POI proximity
features (school, hospital, metro, major road, intersection...) that the public
US 311 datasets simply do not contain. There are two dishonest ways to handle
that and one honest one.

  Dishonest A: omit the columns. The schema then changes shape the day PostGIS
               arrives, and every downstream consumer breaks.
  Dishonest B: fill them with plausible values. This is fabrication.
  Honest:      emit every column with its correct name and dtype, fill the ones
               that require an external geospatial layer with NULL, and record
               which provider produced the row.

This script does the third. It is a PROVIDER pattern:

  null_provider   (default) — emits the full schema, POI/road/area features NULL.
                  Used for all US 311 data. Historical-density features ARE
                  computed, because those need only past incidents.
  india_postgis   (stub, not implemented) — the future provider. The interface
                  it must satisfy is defined by GeoProvider below and documented
                  in GEOSPATIAL_FEATURES.md.

LEAKAGE
-------
The historical-density features are the one family that can be computed now, and
they are also the easiest place in the whole pipeline to leak future information.
A naive "count incidents within 250 m" counts incidents that had not yet been
reported. Every window here is STRICTLY BACKWARD: for a row reported at time t,
only incidents with reported_at < t are counted. validate_leakage.py
re-verifies this independently by resampling rows.

  python scripts/preprocess/build_geo_features.py
  python scripts/preprocess/build_geo_features.py --provider null_provider
  python scripts/preprocess/build_geo_features.py --provider reference_layer

The provider classes live in scripts/preprocess/geo_providers.py, which defines
the contract an external geospatial layer must satisfy and what happens when it
cannot supply a value (answer: NULL, always, and the count is reported).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.preprocess.geo_providers import PROVIDERS, resolve_provider
from scripts.utils.cleaning import haversine_m
from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import ensure_dir, load_feature_config, p
from scripts.utils.schema import conform, geo_feature_schema

log = get_logger("preprocess.geo_features")


def _na_series(index, dtype: str) -> pd.Series:
    """
    An all-missing Series of the requested dtype.

    numpy-backed float64 cannot hold pd.NA, and pandas' nullable dtypes cannot
    hold np.nan meaningfully, so the sentinel has to match the dtype.
    """
    if dtype in ("float64", "float32"):
        return pd.Series(np.nan, index=index, dtype=dtype)
    return pd.Series(pd.array([pd.NA] * len(index), dtype=dtype), index=index)


# ---------------------------------------------------------------------------
# Historical density — computed, strictly backward-looking
# ---------------------------------------------------------------------------
def _cell(series: pd.Series, size: float) -> pd.Series:
    return np.floor(pd.to_numeric(series, errors="coerce") / size)


# Conservative metres-per-degree. Latitude uses the SMALLEST real value
# (110,574 m at the equator) and longitude is scaled by the cosine of the
# highest-magnitude latitude a cell touches, so both convert a cell to the
# SMALLEST metric size it could have. Undersizing the cell can only ever widen
# the candidate ring, never narrow it, which is the safe direction.
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQUATOR = 111_320.0


def required_rings(cell_degrees: float, radius_m: float, lat_deg: float) -> tuple[int, int]:
    """
    How many grid cells out the candidate block must reach to be an exact
    superset of a `radius_m` disc, in (latitude, longitude) cell units.

    Two points k cells apart are at least (k-1) cell widths apart, so every k
    with (k-1) * cell_width < radius must be searched: k_max = ceil(radius/width).

    This is computed rather than configured because the original fixed
    `neighbour_ring: 1` is simply wrong for longitude: at Chicago's latitude a
    0.0025 degree longitude cell is ~207 m across, so an incident 240 m away sits
    TWO cells out and was silently never counted. The ring must follow from the
    radius and the latitude, not from a constant.
    """
    lat_cell_m = cell_degrees * M_PER_DEG_LAT
    cos_lat = max(math.cos(math.radians(min(abs(lat_deg), 89.9))), 1e-6)
    lon_cell_m = cell_degrees * M_PER_DEG_LON_EQUATOR * cos_lat
    return (max(1, math.ceil(radius_m / lat_cell_m)),
            max(1, math.ceil(radius_m / lon_cell_m)))


# Grid-cell packing. The longitude index is biased positive before packing so
# that floor-division decoding stays exact for the negative longitudes of every
# US city in this corpus.
_CELL_STRIDE = 2_000_003
_CELL_BIAS = 1_000_000


def _cell_key(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """Pack a 2-D integer grid cell into one int64 so it can be grouped cheaply."""
    return cx.astype(np.int64) * _CELL_STRIDE + (cy.astype(np.int64) + _CELL_BIAS)


def _cell_unkey(key: int) -> tuple[int, int]:
    cx = int(key) // _CELL_STRIDE
    return cx, int(key) - cx * _CELL_STRIDE - _CELL_BIAS


def _ragged_take(lo: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Expand per-row slices [lo_i, lo_i+counts_i) into one flat index array.

    This is the vectorised replacement for `for i in rows: neighbours[lo:hi]`.
    """
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    starts = np.repeat(lo - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
    return (np.arange(total, dtype=np.int64) + starts).astype(np.int64)


def historical_density(df: pd.DataFrame, max_pairs_per_block: int = 4_000_000) -> pd.DataFrame:
    """
    For each incident, count EARLIER incidents within radius_m and each window.

    Definitions (all from config/feature_config.yaml, none hard-coded here):

      nearby_similar_incidents_24h   count of incidents with the SAME category,
                                     within 250 m, reported in [t-24h,  t)
      nearby_similar_incidents_7d    same, [t-168h, t)
      nearby_similar_incidents_30d   same, [t-720h, t)
      local_incident_density         ALL categories, within 250 m, [t-720h, t)
      category_incident_density      nearby_similar_incidents_30d / local_incident_density
                                     (0.0 when the denominator is 0)

    The interval is half-open on the right: `reported_at < t`, never `<=`, so a
    row never counts itself and never counts a simultaneous report. Nothing at or
    after t is visible to the row. This is what makes the feature computable at
    inference time and what validate_leakage.py LEAK-4 re-derives independently.

    ALGORITHM
    ---------
    Rows are sorted by time once. Each row is assigned a ~250 m grid cell, and
    the candidate neighbours of a cell are that cell plus its 8 touching
    neighbours — an exact superset of the true radius neighbourhood, so no true
    neighbour is missed and the distance test still decides membership.

    Within a cell the candidates are already time-ordered, so the backward window
    is two `searchsorted` calls, and the resulting variable-length slices are
    flattened into one pair array and evaluated with vectorised numpy. The
    previous implementation looped in Python over every row and compared it
    against every incident that had EVER occurred in its 3x3 cell block, ignoring
    the time window when selecting candidates; on 1.9M rows that does not finish
    in a sensible time. This version does the same arithmetic in numpy and prunes
    by time before computing any distance.
    """
    fc = load_feature_config()
    specs = fc["historical_density_features"]
    grid = fc["density_grid"]
    size = float(grid["cell_degrees"])
    min_ring = int(grid.get("min_neighbour_ring", 1) or 1)
    max_radius = max(float(sp.get("radius_m", 250)) for sp in specs.values()
                     if sp.get("radius_m"))

    recency = fc.get("historical_recency_features", {}) or {}

    out = pd.DataFrame(index=df.index)
    for name in specs:
        out[name] = _na_series(df.index, specs[name].get("dtype", "Int64"))
    for name in recency:
        out[name] = _na_series(df.index, recency[name].get("dtype", "float64"))

    work = df[["latitude", "longitude", "category", "reported_at"]].copy()
    work["_lat"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["_lon"] = pd.to_numeric(work["longitude"], errors="coerce")
    work["_ts"] = pd.to_datetime(work["reported_at"], utc=True, errors="coerce")
    usable = work["_lat"].notna() & work["_lon"].notna() & work["_ts"].notna()
    if not usable.any():
        log.warning("no rows have both coordinates and a timestamp; all density features NULL")
        return out

    w = work[usable].sort_values("_ts", kind="stable")
    n = len(w)
    lat = w["_lat"].to_numpy(np.float64)
    lon = w["_lon"].to_numpy(np.float64)
    ts = w["_ts"].to_numpy("datetime64[ns]").astype(np.int64) / 1e9 / 3600.0   # hours
    cat = pd.factorize(w["category"].astype("string").fillna(""))[0].astype(np.int64)

    cx = np.floor(lat / size).astype(np.int64)
    cy = np.floor(lon / size).astype(np.int64)
    keys = _cell_key(cx, cy)

    # position lists per cell; positions are ascending == time ascending
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    bounds = np.flatnonzero(np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1])))
    cell_starts = dict(zip(sorted_keys[bounds], bounds))
    cell_ends = dict(zip(sorted_keys[bounds],
                         np.concatenate((bounds[1:], [len(sorted_keys)]))))

    windows = {name: float(sp.get("window_hours", 720)) for name, sp in specs.items()
               if sp.get("window_hours")}
    max_window = max(windows.values())

    counted = [(name, float(sp.get("radius_m", 250)), float(sp.get("window_hours", 720)),
                bool(sp.get("same_category_only", False)))
               for name, sp in specs.items() if name != "category_incident_density"]
    results = {name: np.zeros(n, dtype=np.int64) for name, _, _, _ in counted}

    # Recency: the timestamp of the most recent qualifying EARLIER neighbour.
    # -inf means "none seen", which becomes NULL rather than 0 or a sentinel.
    rec_specs = [(name, float(sp.get("radius_m", 250)), float(sp.get("window_hours", 720)),
                  bool(sp.get("same_category_only", True)))
                 for name, sp in recency.items()]
    rec_last = {name: np.full(n, -np.inf, dtype=np.float64) for name, _, _, _ in rec_specs}
    if rec_specs:
        max_window = max(max_window, max(w for _, _, w, _ in rec_specs))

    n_pairs_total = 0
    ring_cache: dict[int, list[tuple[int, int]]] = {}
    rings_used: set[tuple[int, int]] = set()

    for key in cell_starts:
        # argsort(kind="stable") preserved the original order inside each cell,
        # and positions ascend with time, so `rows` is already time-ordered.
        rows = order[cell_starts[key]:cell_ends[key]]
        base_cx, base_cy = _cell_unkey(key)

        offsets = ring_cache.get(base_cx)
        if offsets is None:
            # widest |latitude| the cell touches -> smallest longitude cell ->
            # widest ring, so the block stays an exact superset everywhere in it
            edge_lat = max(abs(base_cx * size), abs((base_cx + 1) * size))
            r_lat, r_lon = required_rings(size, max_radius, edge_lat)
            r_lat, r_lon = max(r_lat, min_ring), max(r_lon, min_ring)
            rings_used.add((r_lat, r_lon))
            offsets = [(dx, dy) for dx in range(-r_lat, r_lat + 1)
                       for dy in range(-r_lon, r_lon + 1)]
            ring_cache[base_cx] = offsets

        blocks = []
        for dx, dy in offsets:
            k2 = _cell_key(np.array([base_cx + dx]), np.array([base_cy + dy]))[0]
            st = cell_starts.get(k2)
            if st is not None:
                blocks.append(order[st:cell_ends[k2]])
        neigh = np.concatenate(blocks)
        neigh.sort()                                  # positions ascending == time ascending
        t_n = ts[neigh]

        for s0 in range(0, len(rows), 4096):
            chunk = rows[s0:s0 + 4096]
            t_r = ts[chunk]
            lo = np.searchsorted(t_n, t_r - max_window, side="left")
            hi = np.searchsorted(t_n, t_r, side="left")      # STRICTLY earlier
            cnt = (hi - lo).astype(np.int64)
            if cnt.sum() == 0:
                continue
            # keep each evaluated pair block bounded regardless of local density
            step = max(1, int(len(chunk) * max_pairs_per_block / max(1, cnt.sum())))
            for b0 in range(0, len(chunk), step):
                sub = slice(b0, b0 + step)
                c_sub, lo_sub = cnt[sub], lo[sub]
                total = int(c_sub.sum())
                if total == 0:
                    continue
                n_pairs_total += total
                right = neigh[_ragged_take(lo_sub, c_sub)]
                grp = np.repeat(np.arange(len(c_sub), dtype=np.int64), c_sub)
                left_rows = chunk[sub]
                l_lat = np.repeat(lat[left_rows], c_sub)
                l_lon = np.repeat(lon[left_rows], c_sub)
                l_ts = np.repeat(ts[left_rows], c_sub)
                l_cat = np.repeat(cat[left_rows], c_sub)
                d = haversine_m(l_lat, l_lon, lat[right], lon[right])
                dt = l_ts - ts[right]
                same = cat[right] == l_cat
                for name, radius, window, same_only in counted:
                    m = (d <= radius) & (dt <= window)
                    if same_only:
                        m &= same
                    if m.any():
                        results[name][left_rows] += np.bincount(grp[m], minlength=len(c_sub))

                for name, radius, window, same_only in rec_specs:
                    m = (d <= radius) & (dt <= window)
                    if same_only:
                        m &= same
                    if m.any():
                        block = np.full(len(c_sub), -np.inf, dtype=np.float64)
                        np.maximum.at(block, grp[m], ts[right][m])
                        # fancy indexing returns a copy, so combine and write back
                        rec_last[name][left_rows] = np.maximum(
                            rec_last[name][left_rows], block)

    log.info("density: %d rows, %d candidate pairs evaluated, cell rings (lat,lon)=%s",
             n, n_pairs_total, sorted(rings_used))

    frame = pd.DataFrame(index=w.index)
    for name, _, _, _ in counted:
        frame[name] = results[name]
    if "category_incident_density" in specs:
        denom = frame["local_incident_density"].to_numpy(np.float64) \
            if "local_incident_density" in frame.columns else np.zeros(n)
        numer = frame["nearby_similar_incidents_30d"].to_numpy(np.float64) \
            if "nearby_similar_incidents_30d" in frame.columns else np.zeros(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(denom > 0, numer / np.where(denom > 0, denom, 1), 0.0)
        frame["category_incident_density"] = np.round(share, 4)

    for name, _, _, _ in rec_specs:
        last = rec_last[name]
        hours = np.where(np.isfinite(last), ts - last, np.nan)
        frame[name] = np.round(hours, 4)

    for name, spec in list(specs.items()) + list(recency.items()):
        if name not in frame.columns:
            continue
        dtype = spec.get("dtype", "Int64")
        out.loc[w.index, name] = frame[name].astype("float64") if dtype == "float64" \
            else frame[name].astype("Int64")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--provider", default=None, choices=sorted(PROVIDERS),
                    help="default: feature_config.yaml geo_provider.name, else null_provider")
    ap.add_argument("--skip-density", action="store_true",
                    help="skip historical density (it is O(n * neighbours); slow on full data)")
    args = ap.parse_args()

    src = Path(args.input) if args.input else p("processed", "incidents", "all_incidents.parquet")
    if not src.exists():
        log.error("not found: %s — run build_incidents.py first", src)
        return 2

    df = pd.read_parquet(src)
    log.info("loaded %d incidents", len(df))

    provider = resolve_provider(args.provider)
    poi = provider.poi_features(df)
    log.info("provider '%s' supplied %d POI/road/area columns (all NULL by design for US data)",
             provider.name, poi.shape[1])

    if args.skip_density:
        fc = load_feature_config()
        dens = pd.DataFrame(
            {n: _na_series(df.index, sp.get("dtype", "Int64"))
             for n, sp in fc["historical_density_features"].items()}, index=df.index)
        log.warning("historical density SKIPPED — those columns are NULL")
    else:
        log.info("computing strictly-backward historical density features...")
        dens = historical_density(df)

    geo = pd.concat([df[["incident_id"]].reset_index(drop=True),
                     poi.reset_index(drop=True), dens.reset_index(drop=True)], axis=1)
    geo["geo_provider"] = provider.name
    feature_cols = [c for c in geo.columns if c not in ("incident_id", "geo_provider")]
    geo["geo_features_available"] = geo[feature_cols].notna().sum(axis=1).astype("Int64")

    schema = geo_feature_schema()
    geo = conform(geo, schema)

    dest = ensure_dir(p("processed", "geo_features", "incident_geo_features.parquet"))
    geo.to_parquet(dest, index=False)

    postgis_cols = [c for c, (_, prov, _) in schema.items() if prov == "SYSTEM"]
    derived_cols = [c for c in feature_cols if c not in postgis_cols]
    summary = {
        "provider": provider.name,
        "provider_coverage": provider.coverage(),
        "rows": int(len(geo)),
        "columns_total": int(geo.shape[1]),
        "columns_requiring_external_geospatial": len(postgis_cols),
        "columns_computed_from_311_history": len(derived_cols),
        "non_null_counts": {c: int(geo[c].notna().sum()) for c in feature_cols},
        "availability_note": (
            "Every column listed under columns_requiring_external_geospatial is NULL. "
            "That is correct and deliberate: the public US 311 datasets contain no POI, "
            "road-network or urban-classification layer, and US POI data would be "
            "meaningless for an Indian deployment. These are populated at runtime by the "
            "India PostGIS layer. See GEOSPATIAL_FEATURES.md."),
        "leakage_note": (
            "Historical-density features use strictly backward windows: for a row reported "
            "at t, only incidents with reported_at < t are counted."),
        "requires_external_geospatial": sorted(postgis_cols),
        "computed_now": sorted(derived_cols),
    }
    ensure_dir(p("reports", "geo_features_summary.json")).write_text(json.dumps(summary, indent=2))
    log.info("wrote %s (%d rows, %d cols)", dest, len(geo), geo.shape[1])
    log.info("  computed now          : %s", derived_cols)
    log.info("  awaiting PostGIS (NULL): %d columns", len(postgis_cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
