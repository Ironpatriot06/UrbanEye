#!/usr/bin/env python3
"""
Phase 2B — read-only audit of the live UrbanEye database.

Question: does UrbanEye's own incident history contain enough data to train or
validate any of the three models?

The connection is opened with `default_transaction_read_only=on` AND a
read-only session, so a typo in a query cannot write. Only aggregate facts are
stored in the report — no titles, descriptions, names or user ids.

    DATABASE_URL=postgresql://urbaneye:...@localhost:5432/urbaneye \
        python scripts/phase2b/audit_urbaneye_db.py

Falls back to the backend's documented dev defaults (docker-compose.yml) when
DATABASE_URL is not set.
"""
from __future__ import annotations

import json
import os
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.phase2b.common import REPORT_DIR, now_utc, write_json

# Bounding boxes used only to say which city a coordinate falls in.
CITY_BBOX = {
    "Bengaluru": (12.70, 13.25, 77.35, 77.90),
    "Chennai": (12.75, 13.35, 79.95, 80.35),
}
# Minimums below which a model trained or validated on UrbanEye data would be
# noise. They are deliberately modest; the point is the order of magnitude.
MIN_RESOLVED_PER_CATEGORY = 500
MIN_WEEKS_OF_HISTORY = 26


def _dsn() -> dict:
    url = os.environ.get("DATABASE_URL",
                         "postgresql://urbaneye:urbaneye_dev@localhost:5432/urbaneye")
    u = urlparse(url.replace("+psycopg2", ""))
    return dict(host=u.hostname, port=u.port or 5432, user=u.username,
                password=u.password, dbname=u.path.lstrip("/"))


def which_city(lat, lon) -> str:
    for name, (a, b, c, d) in CITY_BBOX.items():
        if lat is not None and a <= lat <= b and c <= lon <= d:
            return name
    return "outside known Indian bboxes"


def main() -> int:
    import psycopg2
    try:
        conn = psycopg2.connect(options="-c default_transaction_read_only=on", **_dsn())
    except Exception as e:  # noqa: BLE001
        print(f"database unreachable: {e}")
        return 1
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()

    def q(sql):
        cur.execute(sql)
        return cur.fetchall()

    assert q("show transaction_read_only")[0][0] == "on"

    tables = [r[0] for r in q("select table_name from information_schema.tables "
                              "where table_schema='public' and table_type='BASE TABLE' order by 1")]
    counts = {t: q(f'select count(*) from "{t}"')[0][0] for t in tables}
    inc_cols = [r[0] for r in q("select column_name from information_schema.columns "
                                "where table_name='incidents' order by ordinal_position")]

    rows = q("select category::text, source::text, status::text, priority_level::text, sla_hours, "
             "latitude, longitude, created_at from incidents")
    span = q("select min(created_at), max(created_at) from incidents")[0]

    def tally(i):
        out = {}
        for r in rows:
            out[str(r[i])] = out.get(str(r[i]), 0) + 1
        return out

    # resolution timestamps only exist as history events
    resolved = q("""
        select i.category::text, extract(epoch from (h.created_at - i.created_at))/3600.0
        from incidents i join incident_history h on h.incident_id = i.id
        where h.action = 'INCIDENT_RESOLVED'""")
    transitions = q("select action, count(*) from incident_history group by action order by 2 desc")
    indexes = [r[0] for r in q("select indexdef from pg_indexes where tablename='incidents'")]

    resolved_by_cat: dict[str, int] = {}
    for cat, _ in resolved:
        resolved_by_cat[cat] = resolved_by_cat.get(cat, 0) + 1
    weeks = ((span[1] - span[0]).total_seconds() / 604800.0) if span[0] else 0.0

    report = {
        "generated_at_utc": now_utc(),
        "connection": "read-only (default_transaction_read_only=on, readonly session)",
        "tables": counts,
        "incident_columns": inc_cols,
        "has_resolved_at_column": "resolved_at" in inc_cols,
        "has_zone_or_ward_column": any(c in inc_cols for c in ("ward", "zone", "zone_id", "city")),
        "incidents": {
            "n": len(rows),
            "created_at_min": str(span[0]), "created_at_max": str(span[1]),
            "history_span_weeks": round(weeks, 2),
            "by_category": tally(0), "by_source": tally(1), "by_status": tally(2),
            "by_priority": tally(3), "sla_hours_values": tally(4),
            "by_city_bbox": {k: v for k, v in
                             _count(which_city(r[5], r[6]) for r in rows).items()},
        },
        "resolved": {
            "n": len(resolved),
            "by_category": resolved_by_cat,
            "resolution_hours": [round(float(h), 6) for _, h in resolved],
        },
        "history_transitions": {a: n for a, n in transitions},
        "incident_indexes": indexes,
        "requirements": {
            "min_resolved_per_category": MIN_RESOLVED_PER_CATEGORY,
            "min_weeks_of_history": MIN_WEEKS_OF_HISTORY,
        },
    }
    enough = (all(resolved_by_cat.get(c, 0) >= MIN_RESOLVED_PER_CATEGORY
                  for c in ("POTHOLE", "FLOOD", "GARBAGE", "STREETLIGHT"))
              and weeks >= MIN_WEEKS_OF_HISTORY)
    report["verdict"] = {
        "trainable": enough,
        "reason": ("sufficient" if enough else
                   f"{len(rows)} incidents over {weeks:.1f} weeks, {len(resolved)} resolved; "
                   f"needs >= {MIN_RESOLVED_PER_CATEGORY} resolved incidents per supported category "
                   f"and >= {MIN_WEEKS_OF_HISTORY} weeks of history"),
    }
    write_json(REPORT_DIR / "urbaneye_db_audit.json", report)
    print(json.dumps(report["verdict"], indent=2))
    return 0


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
