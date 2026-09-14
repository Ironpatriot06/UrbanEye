#!/usr/bin/env python3
"""
Generate SYNTHETIC test fixtures.

=============================================================================
THESE ARE NOT DATA. THEY ARE TEST INPUTS.
=============================================================================
Every value below is machine-generated. Nothing here was downloaded from any
municipal portal. The fixtures exist for exactly one purpose: to execute the
preprocessing, priority and validation code so we can prove the scripts run and
the integrity assertions fire, without a network connection.

They are written to tests/fixtures/ and NEVER to data/raw/. No fixture-derived
output may be presented as a processed dataset. Every file carries a
__SYNTHETIC__ marker column or field so a stray fixture cannot be mistaken for
real data downstream.

The fixtures deliberately include dirty rows that mirror real defects observed
in the sources' published metadata, so the cleaning rules are actually exercised:
  * lat/lon = 0.0                (SF's null-island sentinel: 551,131 real rows)
  * closed_date of 3027-03-30    (NYC's real published maximum)
  * closed_date of 1899-12-31    (NYC's real published minimum)
  * closed before opened         (negative resolution time)
  * an unmapped category value   (to exercise the UNMAPPED path)
  * a Chicago parent pointer     (to exercise duplicate-pair construction)

  python tests/make_fixtures.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

FIX = Path(__file__).parent / "fixtures"
FIX.mkdir(parents=True, exist_ok=True)
rng = random.Random(42)
BASE = datetime(2021, 3, 1, 9, 0, 0)

MARKER = "__SYNTHETIC_TEST_FIXTURE_NOT_REAL_DATA__"


def _dt(days: float) -> str:
    return (BASE + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.000")


# ---------------------------------------------------------------------------
def sf311(n: int = 400) -> Path:
    cats = [("Street Defects", "Pavement_Defect"), ("Streetlights", "Streetlight_Other"),
            ("Sewer Issues", "Sewer_Backup"), ("Graffiti", "Graffiti on Pole"),
            ("Street and Sidewalk Cleaning", "Bulky Items"),
            ("Street and Sidewalk Cleaning", "garbage_and_debris"),
            ("Sign Repair", "Sign_Damaged"), ("Encampments", "Encampment Reports"),
            ("Totally Invented Category", "x")]  # exercises the UNMAPPED path
    rows = []
    for i in range(n):
        cat, sub = rng.choice(cats)
        opened = rng.uniform(0, 1400)
        closed = opened + rng.uniform(1, 400)
        null_island = (i % 17 == 0)
        rows.append({
            MARKER: True,
            "service_request_id": 900000 + i,
            "requested_datetime": _dt(opened),
            "closed_date": _dt(closed) if i % 11 else None,
            "updated_datetime": _dt(closed),
            "status_description": "Closed" if i % 11 else "Open",
            "status_notes": ("Case is a Duplicate - already reported" if i % 23 == 0
                             else "Case Resolved - Pickup completed."),
            "agency_responsible": rng.choice(["DPW Ops Queue", "PUC Sewer Ops", "DPW BSM Queue"]),
            "service_name": cat,
            "service_subtype": sub,
            "service_details": "synthetic detail",
            "address": f"{100 + i} SYNTHETIC ST, SAN FRANCISCO, CA",
            "street": "SYNTHETIC ST",
            "supervisor_district": rng.randint(1, 11),
            "analysis_neighborhood": rng.choice(["Mission", "Tenderloin", "Sunset/Parkside"]),
            "police_district": "MISSION",
            "lat": 0.0 if null_island else round(rng.uniform(37.72, 37.80), 6),
            "long": 0.0 if null_island else round(rng.uniform(-122.48, -122.39), 6),
            "source": rng.choice(["Mobile/Open311", "Phone", "Web"]),
            "media_url": ({"url": f"https://example.invalid/sf/{i}.jpg"} if i % 4 == 0 else None),
        })
    fp = FIX / "sf311_cases.jsonl"
    fp.write_text("\n".join(json.dumps(r) for r in rows))
    return fp


_CHI_HOTSPOTS = [(41.70 + 0.02 * k, -87.80 + 0.015 * k) for k in range(15)]


def _chi_point(i: int) -> dict:
    lat, lon = _CHI_HOTSPOTS[i % len(_CHI_HOTSPOTS)]
    return {"latitude": round(lat + rng.uniform(-0.0008, 0.0008), 6),
            "longitude": round(lon + rng.uniform(-0.0008, 0.0008), 6)}


def chicago311(n: int = 400) -> Path:
    types = ["Pothole in Street Complaint", "Street Light Out Complaint",
             "Traffic Signal Out Complaint", "Graffiti Removal Request",
             "Tree Emergency", "Sanitation Code Violation",
             "Sign Repair Request - All Other Signs",
             "311 INFORMATION ONLY CALL"]  # must be filtered out
    rows = []
    for i in range(n):
        t = rng.choice(types)
        opened = rng.uniform(0, 1400)
        sr = f"SR21-{1000000 + i}"
        # ~15% are children of an earlier request -> duplicate positives
        parent = f"SR21-{1000000 + rng.randint(0, max(0, i - 1))}" if (i > 20 and i % 7 == 0) else None
        rows.append({
            MARKER: True,
            "sr_number": sr,
            "sr_type": t,
            "sr_short_code": "PHF",
            "owner_department": rng.choice(["CDOT - Department of Transportation", "Streets and Sanitation"]),
            "status": "Completed" if i % 9 else "Open",
            "origin": rng.choice(["Phone Call", "Internet", "Mobile Device"]),
            "created_date": _dt(opened),
            "last_modified_date": _dt(opened + 5),
            "closed_date": _dt(opened + rng.uniform(1, 200)) if i % 9 else None,
            "street_address": f"{i} W SYNTHETIC ST",
            "city": "Chicago",
            "zip_code": "60612",
            "duplicate": bool(parent),
            "parent_sr_number": parent,
            "community_area": rng.randint(1, 77),
            "ward": rng.randint(1, 50),
            "police_district": "12",
            "created_hour": rng.randint(0, 23),
            "created_day_of_week": rng.randint(1, 7),
            "created_month": rng.randint(1, 12),
            # Cluster around a small set of hotspots rather than spreading
            # uniformly: real civic incidents are spatially clustered, and a
            # uniform fixture makes near-miss negative sampling untestable.
            **_chi_point(i),
        })
    fp = FIX / "chicago311.jsonl"
    fp.write_text("\n".join(json.dumps(r) for r in rows))
    return fp


def nyc311(n: int = 400) -> Path:
    combos = [("Street Condition", "Pothole"), ("Street Light Condition", "Street Light Out"),
              ("Traffic Signal Condition", "Controller"), ("Sewer", "Catch Basin Clogged"),
              ("Dirty Conditions", "Trash"), ("Sidewalk Condition", "Broken Sidewalk"),
              ("Noise - Residential", "Loud Music/Party")]
    rows = []
    for i in range(n):
        ct, desc = rng.choice(combos)
        opened = rng.uniform(0, 1400)
        # Reproduce NYC's real date pathologies
        if i % 37 == 0:
            closed = "3027-03-30T00:00:00.000"
        elif i % 41 == 0:
            closed = "1899-12-31T19:00:00.000"
        elif i % 29 == 0:
            closed = _dt(opened - 5)                      # negative duration
        else:
            closed = _dt(opened + rng.uniform(1, 300))
        rows.append({
            MARKER: True,
            "unique_key": str(30000000 + i),
            "created_date": _dt(opened),
            "closed_date": closed,
            "agency": "DOT",
            "agency_name": "Department of Transportation",
            "complaint_type": ct,
            "descriptor": desc,
            "descriptor_2": None,
            "location_type": "Street",
            "incident_zip": "11226",
            "incident_address": f"{i} SYNTHETIC AVENUE",
            "status": "Closed",
            "due_date": _dt(opened + rng.choice([1, 3, 7, 14, 30])) if i % 3 else None,
            "resolution_description": "The Department of Transportation inspected this complaint and repaired the problem.",
            "resolution_action_updated_date": _dt(opened + 10),
            "community_board": f"{rng.randint(1, 18):02d} BROOKLYN",
            "council_district": str(rng.randint(1, 51)),
            "borough": "BROOKLYN",
            "latitude": round(rng.uniform(40.55, 40.90), 6),
            "longitude": round(rng.uniform(-74.05, -73.75), 6),
            "open_data_channel_type": rng.choice(["PHONE", "ONLINE", "MOBILE"]),
        })
    fp = FIX / "nyc311_2010_2019.jsonl"
    fp.write_text("\n".join(json.dumps(r) for r in rows))
    return fp


def boston311(n: int = 300) -> Path:
    combos = [("Street Lights", "Street Light Outages"), ("Highway Maintenance", "Requests for Pothole Repair"),
              ("Sanitation", "Schedule a Bulk Item Pickup"), ("Signs & Signals", "Sign Repair"),
              ("Trees", "Tree Maintenance Requests")]
    rows = []
    for i in range(n):
        reason, typ = rng.choice(combos)
        opened = BASE + timedelta(days=rng.uniform(0, 1400))
        sla_days = rng.choice([1, 3, 7, 14, 30])
        target = opened + timedelta(days=sla_days)
        closed = opened + timedelta(days=rng.uniform(0.2, 40))
        rows.append({
            MARKER: True,
            "CASE_ENQUIRY_ID": 101000000000 + i,
            "OPEN_DT": opened.strftime("%Y-%m-%d %H:%M:%S"),
            "TARGET_DT": target.strftime("%Y-%m-%d %H:%M:%S"),
            "CLOSED_DT": closed.strftime("%Y-%m-%d %H:%M:%S") if i % 8 else "",
            "OnTime_Status": "ONTIME" if closed <= target else "OVERDUE",
            "CASE_STATUS": "Closed" if i % 8 else "Open",
            "CLOSURE_REASON": "Case Closed. Resolved.",
            "CASE_TITLE": typ,
            "SUBJECT": "Public Works Department",
            "REASON": reason,
            "TYPE": typ,
            "QUEUE": "PWDx_" + typ.replace(" ", "_"),
            "Department": "PWDx",
            "SubmittedPhoto": (f"https://example.invalid/bos/{i}_sub.jpg" if i % 5 == 0 else ""),
            "ClosedPhoto": (f"https://example.invalid/bos/{i}_clo.jpg" if i % 9 == 0 else ""),
            "Location": f"{i} Synthetic St  Boston  MA  02124",
            "neighborhood": rng.choice(["Dorchester", "Roxbury", "South Boston"]),
            "ward": rng.randint(1, 22),
            "precinct": "0101",
            "latitude": round(rng.uniform(42.24, 42.40), 6),
            "longitude": round(rng.uniform(-71.19, -70.99), 6),
            "source": rng.choice(["Constituent Call", "Citizens Connect App", "City Worker App"]),
        })
    fp = FIX / "boston311_2021.csv"
    pd.DataFrame(rows).to_csv(fp, index=False)
    return fp


def rdd2022_voc(n: int = 60) -> Path:
    """Minimal VOC tree: fixtures/rdd2022/India/{train,test}/annotations/xmls/*.xml"""
    root = FIX / "rdd2022"
    classes = ["D00", "D10", "D20", "D40", "D50"]
    made = 0
    for split in ("train", "test"):
        d = root / "India" / split / "annotations" / "xmls"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n if split == "train" else n // 4):
            objs = ""
            for _ in range(rng.randint(1, 3)):
                c = rng.choice(classes)
                x1, y1 = rng.randint(0, 400), rng.randint(0, 400)
                objs += (f"<object><name>{c}</name><bndbox>"
                         f"<xmin>{x1}</xmin><ymin>{y1}</ymin>"
                         f"<xmax>{x1 + rng.randint(20, 150)}</xmax>"
                         f"<ymax>{y1 + rng.randint(20, 150)}</ymax></bndbox></object>")
            (d / f"India_{split}_{i:05d}.xml").write_text(
                f"<annotation><!-- {MARKER} -->"
                f"<filename>India_{split}_{i:05d}.jpg</filename>"
                f"<size><width>600</width><height>600</height><depth>3</depth></size>"
                f"{objs}</annotation>")
            made += 1
    (root / "README_SYNTHETIC.txt").write_text(MARKER + "\nGenerated by tests/make_fixtures.py\n")
    return root


def smartathon(n: int = 300) -> Path:
    classes = ["GARBAGE", "POTHOLES", "CONSTRUCTION_ROAD", "CLUTTER_SIDEWALK",
               "GRAFFITI", "BAD_BILLBOARD", "FADED_SIGNAGE", "BAD_STREETLIGHT"]
    weights = [40, 22, 14, 10, 7, 4, 2, 1]     # mimics the real long tail
    rows = []
    for i in range(n):
        c = rng.choices(classes, weights=weights, k=1)[0]
        x1, y1 = rng.randint(0, 500), rng.randint(0, 500)
        rows.append({
            "class": classes.index(c),
            "image_path": f"synthetic_{i % 120:04d}.jpg",
            "name": c,
            "xmax": x1 + rng.randint(20, 200),
            "xmin": x1,
            "ymax": y1 + rng.randint(20, 200),
            "ymin": y1,
        })
    d = FIX / "smartathon"
    d.mkdir(parents=True, exist_ok=True)
    fp = d / "train.csv"
    pd.DataFrame(rows).to_csv(fp, index=False)
    (d / "README_SYNTHETIC.txt").write_text(MARKER + "\nGenerated by tests/make_fixtures.py\n")
    return fp


# ---------------------------------------------------------------------------
# Bulk-CSV variants.
#
# The portals' Export buttons produce CSVs with HUMAN-READABLE headers, which
# differ from the Socrata API field names the JSONL fixtures use. Since the user
# downloads by hand, the CSV path is the one that will actually run in anger, so
# it needs its own fixture rather than being assumed to work.
# ---------------------------------------------------------------------------
SF_CSV_HEADERS = {
    "service_request_id": "CaseID", "requested_datetime": "Opened", "closed_date": "Closed",
    "updated_datetime": "Updated", "status_description": "Status", "status_notes": "Status Notes",
    "agency_responsible": "Responsible Agency", "service_name": "Category",
    "service_subtype": "Request Type", "service_details": "Request Details",
    "address": "Address", "street": "Street", "supervisor_district": "Supervisor District",
    "analysis_neighborhood": "Analysis Neighborhood", "police_district": "Police District",
    "lat": "Latitude", "long": "Longitude", "source": "Source", "media_url": "Media URL",
}

NYC_CSV_HEADERS = {
    "unique_key": "Unique Key", "created_date": "Created Date", "closed_date": "Closed Date",
    "agency": "Agency", "agency_name": "Agency Name", "complaint_type": "Complaint Type",
    "descriptor": "Descriptor", "location_type": "Location Type", "incident_zip": "Incident Zip",
    "incident_address": "Incident Address", "status": "Status", "due_date": "Due Date",
    "resolution_description": "Resolution Description", "community_board": "Community Board",
    "borough": "Borough", "latitude": "Latitude", "longitude": "Longitude",
    "open_data_channel_type": "Open Data Channel Type",
}


def _jsonl_to_bulk_csv(jsonl: Path, out: Path, header_map: dict) -> Path:
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if "media_url" in df.columns:            # CSV export flattens the url object
        df["media_url"] = df["media_url"].map(lambda v: v.get("url") if isinstance(v, dict) else v)
    df = df.rename(columns=header_map)
    df.to_csv(out, index=False)
    return out


def bulk_csv_variants() -> list[Path]:
    made = []
    made.append(_jsonl_to_bulk_csv(FIX / "sf311_cases.jsonl",
                                   FIX / "sf311_bulk_export.csv", SF_CSV_HEADERS))
    made.append(_jsonl_to_bulk_csv(FIX / "nyc311_2010_2019.jsonl",
                                   FIX / "nyc311_bulk_export.csv", NYC_CSV_HEADERS))
    # Chicago's bulk CSV already uses UPPERCASE API-equivalent names.
    rows = [json.loads(l) for l in (FIX / "chicago311.jsonl").read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows); df.columns = [c.upper() for c in df.columns]
    fp = FIX / "chicago311_bulk_export.csv"; df.to_csv(fp, index=False); made.append(fp)
    return made


if __name__ == "__main__":
    print("Generating SYNTHETIC fixtures (NOT real data) in", FIX)
    # Scope: tabular only. rdd2022_voc/smartathon remain defined for reference but
    # are NOT generated — this pipeline has no image task.
    for fn in (sf311, chicago311, nyc311, boston311):
        print("  ", fn.__name__, "->", fn())
    for pth in bulk_csv_variants():
        print("   bulk_csv ->", pth)
    (FIX / "README.md").write_text(
        "# SYNTHETIC TEST FIXTURES\n\n"
        "**These files are machine-generated and are NOT data.**\n\n"
        "Nothing here was downloaded from any municipal portal. They exist solely to\n"
        "execute the pipeline code offline and prove the integrity assertions fire.\n"
        "No output derived from these fixtures may be reported as a processed dataset.\n")
    print("done")