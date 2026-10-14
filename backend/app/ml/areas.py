"""
Service areas.

The history features were defined per CITY (the training publisher: Chicago,
San Francisco, New York City), and the spatial grid was anchored at a fixed
per-city origin.  UrbanEye stores no city column, so a city is a configured
bounding box.  An incident inside no box has no history features (they are
reported as missing, which makes a model prediction EXPERIMENTAL).

The default boxes are the ones the Phase 2B database audit used
(UrbanEye_ML_Data_Pipeline/scripts/phase2b/audit_urbaneye_db.py).  Override
with ML_SERVICE_AREAS (JSON list).
"""

import json
from dataclasses import dataclass
from typing import Optional

DEFAULT_AREAS = [
    {"name": "Bengaluru", "lat_min": 12.70, "lat_max": 13.25, "lon_min": 77.35, "lon_max": 77.90},
    {"name": "Chennai", "lat_min": 12.75, "lat_max": 13.35, "lon_min": 79.95, "lon_max": 80.35},
]


@dataclass(frozen=True)
class ServiceArea:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @property
    def ref_lat(self) -> float:
        """Latitude for the equirectangular x-scale (training used the city median)."""
        return (self.lat_min + self.lat_max) / 2.0

    def contains(self, lat: float, lon: float) -> bool:
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max


def parse_areas(raw: str) -> list[ServiceArea]:
    items = json.loads(raw) if raw and raw.strip() else DEFAULT_AREAS
    areas = [ServiceArea(name=str(a["name"]), lat_min=float(a["lat_min"]),
                         lat_max=float(a["lat_max"]), lon_min=float(a["lon_min"]),
                         lon_max=float(a["lon_max"])) for a in items]
    for a in areas:
        if not (a.lat_min < a.lat_max and a.lon_min < a.lon_max):
            raise ValueError(f"service area {a.name!r} has an empty box")
    return areas


def get_areas() -> list[ServiceArea]:
    from app.core.config import get_settings

    return parse_areas(get_settings().ML_SERVICE_AREAS)


def area_for(lat: float, lon: float, areas: Optional[list[ServiceArea]] = None) -> Optional[ServiceArea]:
    for a in areas if areas is not None else get_areas():
        if a.contains(lat, lon):
            return a
    return None


def area_by_name(name: str, areas: Optional[list[ServiceArea]] = None) -> Optional[ServiceArea]:
    for a in areas if areas is not None else get_areas():
        if a.name.lower() == name.lower():
            return a
    return None
