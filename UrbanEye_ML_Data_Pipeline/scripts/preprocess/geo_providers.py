"""
Geospatial providers: the pluggable layer that turns a coordinate into POI,
road-network and area-context features.

THE CONTRACT
------------
A provider receives a frame with `incident_id`, `latitude`, `longitude` and
returns a frame, indexed the same way, holding every column declared in
`config/feature_config.yaml` under poi_proximity_features, road_network_features
and area_context_features.

Three rules, and the third is the one that matters:

  1. Return every declared column, with the declared dtype, always.
  2. Compute what you can from real reference data.
  3. **Return NULL for anything you cannot compute.** Never a sentinel, never an
     imputed mean, never a plausible-looking number. A row outside the
     provider's coverage area must come back NULL, and the summary must say how
     many rows that was. A provider that returns a number it cannot justify is
     worse than no provider at all, because nothing downstream can tell.

Registered providers
--------------------
  null_provider     the default. Emits the full schema, everything NULL. Correct
                    for US 311 data, where no reference layer exists and where a
                    US POI layer would in any case be meaningless for an Indian
                    deployment.
  reference_layer   computes nearest-POI distances from local reference files
                    declared in data/external/reference_layers.yaml. Emits NULL
                    for any layer that is not present — it never invents one.
  india_postgis     stub. Documents the query contract a PostGIS deployment must
                    satisfy and refuses to run until one is configured, rather
                    than silently degrading to NULLs that look like the default.

Selection order: --provider on the command line, else feature_config.yaml
`geo_provider.name`, else null_provider.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..utils.cleaning import haversine_m
from ..utils.logging_setup import get_logger
from ..utils.paths import load_feature_config, p, repo_root

log = get_logger("preprocess.geo_providers")

M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQUATOR = 111_320.0

# Which declared feature belongs to which reference layer, and which distance
# column carries it. The provider fills a pair only when the layer file exists.
LAYER_FEATURES = {
    "school": ("near_school", "distance_to_nearest_school_m"),
    "hospital": ("near_hospital", "distance_to_nearest_hospital_m"),
    "metro": ("near_metro", "distance_to_nearest_metro_m"),
    "rail_station": ("near_rail_station", "distance_to_nearest_rail_station_m"),
    "public_transport": ("near_public_transport", "distance_to_nearest_public_transport_m"),
    "fire_station": ("near_fire_station", "distance_to_nearest_fire_station_m"),
    "police_station": ("near_police_station", "distance_to_nearest_police_station_m"),
    "government_building": ("near_government_building",
                            "distance_to_nearest_government_building_m"),
    "major_road": ("near_major_road", "distance_to_major_road_m"),
    "intersection": ("near_intersection", "distance_to_nearest_intersection_m"),
}


def _na_series(index, dtype: str) -> pd.Series:
    if dtype in ("float64", "float32"):
        return pd.Series(np.nan, index=index, dtype=dtype)
    return pd.Series(pd.array([pd.NA] * len(index), dtype=dtype), index=index)


def declared_columns() -> dict[str, str]:
    """Every POI/road/area column the config declares, mapped to its dtype."""
    fc = load_feature_config()
    cols: dict[str, str] = {}
    for block in ("poi_proximity_features", "road_network_features", "area_context_features"):
        for name, spec in fc.get(block, {}).items():
            cols[name] = spec.get("dtype", "float64")
    return cols


class GeoProvider:
    """Interface every provider must satisfy. See the module docstring."""

    name = "abstract"

    def poi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def coverage(self) -> dict:
        """What this provider could and could not supply, for the run report."""
        return {"provider": self.name}

    def empty_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: _na_series(df.index, t) for c, t in declared_columns().items()},
                            index=df.index)


class NullProvider(GeoProvider):
    """
    The only provider that can run today.

    Returns the full POI/road/area column set, entirely NULL. This is the
    correct and honest output for US 311 coordinates: UrbanEye+ has no licensed
    Indian POI layer plugged in, and US POI data would be worse than useless
    because the product deploys in India.
    """

    name = "null_provider"

    def poi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.empty_frame(df)

    def coverage(self) -> dict:
        return {
            "provider": self.name,
            "columns_supplied": 0,
            "columns_null": len(declared_columns()),
            "reason": ("no reference geodata is configured. data/external/reference_layers.yaml "
                       "is absent, and the 311 publishers supply a coordinate and nothing else."),
        }


class ReferenceLayerProvider(GeoProvider):
    """
    Nearest-POI distances from local reference files.

    Declare the layers in `data/external/reference_layers.yaml`:

        crs: EPSG:4326
        max_search_radius_m: 5000
        layers:
          school:
            path: schools_chennai.parquet     # relative to data/external/
            lat_column: lat
            lon_column: lon
            city: Chennai                     # optional: restrict to one city
            licence: "ODbL — OpenStreetMap contributors"
          hospital:
            path: hospitals_chennai.csv
            ...

    Only the layers actually present are computed. Every other declared column
    comes back NULL, and `coverage()` names which layers were missing, so a
    half-populated schema can never be mistaken for a complete one.

    Beyond `max_search_radius_m` the exact distance is NOT computed, and the
    distance is returned as NULL while `near_*` is returned as False. That
    asymmetry is deliberate: "there is no school within 5 km" is knowledge and is
    recorded, but the precise distance to a school 12 km away is not, and
    inventing one would be fabrication. The priority engine reads the distance
    column, so such rows simply contribute no signal for that feature.

    The search is grid-blocked and vectorised: reference points are bucketed on
    a metric-sized lattice and each incident scans outward ring by ring until the
    ring's lower distance bound exceeds the best match found. It never forms the
    full incidents x POIs product.
    """

    name = "reference_layer"

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or (repo_root() / "data" / "external"
                                           / "reference_layers.yaml")
        self.cfg: dict = {}
        self.missing: list[str] = []
        self.loaded: dict[str, pd.DataFrame] = {}
        self.stats: dict = {}
        if self.config_path.exists():
            self.cfg = yaml.safe_load(self.config_path.read_text()) or {}
        else:
            log.warning("no reference layer config at %s — every POI column will be NULL",
                        self.config_path)

    # -- loading ----------------------------------------------------------
    def _load_layer(self, key: str, spec: dict) -> pd.DataFrame | None:
        base = self.config_path.parent
        fp = base / str(spec.get("path", ""))
        if not fp.exists():
            self.missing.append(f"{key}: file not found ({fp})")
            return None
        df = pd.read_parquet(fp) if fp.suffix == ".parquet" else pd.read_csv(fp)
        lat_c = spec.get("lat_column", "latitude")
        lon_c = spec.get("lon_column", "longitude")
        if lat_c not in df.columns or lon_c not in df.columns:
            self.missing.append(f"{key}: {fp.name} lacks {lat_c}/{lon_c}")
            return None
        out = pd.DataFrame({
            "lat": pd.to_numeric(df[lat_c], errors="coerce"),
            "lon": pd.to_numeric(df[lon_c], errors="coerce"),
        }).dropna()
        if out.empty:
            self.missing.append(f"{key}: {fp.name} has no usable coordinates")
            return None
        return out

    # -- nearest search ---------------------------------------------------
    @staticmethod
    def _nearest(lat: np.ndarray, lon: np.ndarray, ref: pd.DataFrame,
                 max_radius_m: float) -> np.ndarray:
        """
        Distance in metres from each (lat, lon) to the closest reference point,
        NaN when none lies within `max_radius_m`.

        Grid-blocked ring search: cells are sized at the search radius, so ring
        k has a lower bound of (k-1) * cell_size metres and the scan can stop as
        soon as that bound exceeds the best distance found so far. Reference
        layers are small (thousands of points), incidents are millions, so the
        cost is dominated by one pass over the incidents.
        """
        if ref.empty:
            return np.full(len(lat), np.nan)
        rlat = ref["lat"].to_numpy(np.float64)
        rlon = ref["lon"].to_numpy(np.float64)

        mid_lat = float(np.nanmedian(np.concatenate([lat[~np.isnan(lat)][:1000], rlat[:1000]])))
        cos_lat = max(math.cos(math.radians(min(abs(mid_lat), 89.9))), 1e-6)
        cell_lat = max_radius_m / M_PER_DEG_LAT
        cell_lon = max_radius_m / (M_PER_DEG_LON_EQUATOR * cos_lat)

        rcx = np.floor(rlat / cell_lat).astype(np.int64)
        rcy = np.floor(rlon / cell_lon).astype(np.int64)
        buckets: dict[tuple[int, int], np.ndarray] = {}
        keys = rcx * 4_000_003 + rcy
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        bounds = np.flatnonzero(np.concatenate(([True], sk[1:] != sk[:-1])))
        ends = np.concatenate((bounds[1:], [len(sk)]))
        for b, e in zip(bounds, ends):
            buckets[(int(rcx[order[b]]), int(rcy[order[b]]))] = order[b:e]

        out = np.full(len(lat), np.inf)
        valid = ~(np.isnan(lat) | np.isnan(lon))
        icx = np.where(valid, np.floor(np.nan_to_num(lat) / cell_lat), 0).astype(np.int64)
        icy = np.where(valid, np.floor(np.nan_to_num(lon) / cell_lon), 0).astype(np.int64)

        # group incidents by cell so each cell's reference candidates are gathered once
        ikeys = icx * 4_000_003 + icy
        iorder = np.argsort(ikeys, kind="stable")
        isk = ikeys[iorder]
        ib = np.flatnonzero(np.concatenate(([True], isk[1:] != isk[:-1])))
        ie = np.concatenate((ib[1:], [len(isk)]))
        for b, e in zip(ib, ie):
            idx = iorder[b:e]
            idx = idx[valid[idx]]
            if idx.size == 0:
                continue
            cx, cy = int(icx[idx[0]]), int(icy[idx[0]])
            cand: list[np.ndarray] = []
            for ring in (0, 1):
                for dx in range(-ring, ring + 1):
                    for dy in range(-ring, ring + 1):
                        if ring and max(abs(dx), abs(dy)) != ring:
                            continue
                        got = buckets.get((cx + dx, cy + dy))
                        if got is not None:
                            cand.append(got)
                if cand and ring >= 1:
                    break
            if not cand:
                continue
            c = np.concatenate(cand)
            for s0 in range(0, idx.size, 4096):
                chunk = idx[s0:s0 + 4096]
                d = haversine_m(lat[chunk][:, None], lon[chunk][:, None],
                                rlat[c][None, :], rlon[c][None, :])
                out[chunk] = np.minimum(out[chunk], d.min(axis=1))
        out[~valid] = np.inf
        return np.where(np.isfinite(out) & (out <= max_radius_m), out, np.nan)

    # -- interface --------------------------------------------------------
    def poi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.empty_frame(df)
        layers = (self.cfg.get("layers") or {})
        if not layers:
            self.missing.append("no layers declared")
            return out

        fc = load_feature_config()
        thresholds = {}
        try:
            from ..utils.paths import load_priority_config
            thresholds = load_priority_config().get("proximity_thresholds_m", {})
        except Exception:                                   # pragma: no cover
            pass

        max_radius = float(self.cfg.get("max_search_radius_m", 5000))
        lat = pd.to_numeric(df["latitude"], errors="coerce").to_numpy(np.float64)
        lon = pd.to_numeric(df["longitude"], errors="coerce").to_numpy(np.float64)

        for key, spec in layers.items():
            if key not in LAYER_FEATURES:
                self.missing.append(f"{key}: not a declared feature layer")
                continue
            ref = self._load_layer(key, spec or {})
            if ref is None:
                continue
            near_col, dist_col = LAYER_FEATURES[key]
            d = self._nearest(lat, lon, ref, max_radius)
            out[dist_col] = pd.Series(d, index=df.index, dtype="float64")
            thr = float(thresholds.get(key, spec.get("threshold_m", 200)))
            near = pd.Series(pd.array([pd.NA] * len(df), dtype="boolean"), index=df.index)
            known = ~np.isnan(lat) & ~np.isnan(lon)
            near[known] = pd.array(np.where(np.isnan(d[known]), False, d[known] <= thr),
                                   dtype="boolean")
            out[near_col] = near
            self.loaded[key] = ref
            self.stats[key] = {
                "reference_points": int(len(ref)),
                "incidents_with_a_match_within_search_radius": int(np.isfinite(d).sum()),
                "incidents_beyond_search_radius": int((~np.isfinite(d) & known).sum()),
                "threshold_m": thr,
            }
        return out

    def coverage(self) -> dict:
        declared = declared_columns()
        supplied = sorted({c for k in self.loaded for c in LAYER_FEATURES[k]})
        return {
            "provider": self.name,
            "config": str(self.config_path),
            "config_present": self.config_path.exists(),
            "layers_loaded": sorted(self.loaded),
            "layers_missing_or_unusable": self.missing,
            "columns_supplied": len(supplied),
            "columns_null": len(declared) - len(supplied),
            "per_layer": self.stats,
            "null_policy": ("every column of an absent layer stays NULL; a distance beyond "
                            "max_search_radius_m stays NULL while near_* is False"),
        }


class IndiaPostGISProvider(GeoProvider):
    """
    Stub for a PostGIS-backed deployment. Not implemented, and deliberately
    fails loudly rather than falling back to NULLs that would be indistinguishable
    from the default provider.

    What a real implementation must do, per incident batch:

        SELECT i.incident_id,
               ST_Distance(i.geom::geography, s.geom::geography) AS school_m
        FROM   incident_batch i
        CROSS JOIN LATERAL (
            SELECT geom FROM osm_amenities
            WHERE  amenity = 'school'
            ORDER  BY i.geom <-> geom
            LIMIT  1
        ) s;

    Requirements it must satisfy:
      * one <-> KNN lateral join per layer, with a GiST index on every geom;
      * geography casts, not geometry, or the distances are in degrees;
      * a declared coverage polygon — any incident outside it returns NULL, not
        a distance to whatever happens to be nearest inside it;
      * a licence record per layer, surfaced in coverage().
    """

    name = "india_postgis"

    def poi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "india_postgis is a documented stub, not an implementation. Configure a DSN and "
            "implement the KNN lateral joins in scripts/preprocess/geo_providers.py, or run "
            "with --provider null_provider (schema-complete, values NULL) or "
            "--provider reference_layer (needs data/external/reference_layers.yaml).")


PROVIDERS: dict[str, type[GeoProvider]] = {
    "null_provider": NullProvider,
    "reference_layer": ReferenceLayerProvider,
    "india_postgis": IndiaPostGISProvider,
}


def resolve_provider(name: str | None = None) -> GeoProvider:
    """--provider > feature_config.geo_provider.name > null_provider."""
    if not name:
        name = ((load_feature_config().get("geo_provider") or {}).get("name")
                or "null_provider")
    if name not in PROVIDERS:
        raise SystemExit(f"unknown geo provider '{name}'; available: {sorted(PROVIDERS)}")
    return PROVIDERS[name]()
