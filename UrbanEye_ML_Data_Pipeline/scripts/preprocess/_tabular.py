"""
Shared machinery for the four 311 processors.

Everything here is chunked. The largest source is 22.6M rows; loading that into a
DataFrame costs ~20 GB of RAM. Each processor streams fixed-size chunks, cleans
them, and appends to a Parquet writer, so peak memory is bounded by chunk size
regardless of dataset size.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..utils.cleaning import (CleaningStats, clean_coordinates, clean_text,
                              compute_resolution_hours, drop_exact_duplicate_rows,
                              parse_timestamps)
from ..utils.logging_setup import get_logger
from ..utils.mapping import map_categories, report_unmapped
from ..utils.paths import ensure_dir, p
from ..utils.schema import INCIDENT_SCHEMA, finalise_incidents

log = get_logger("preprocess")

DEFAULT_CHUNK = 200_000

_STATUS_MAP = {
    "closed": "closed", "completed": "closed", "completed - dup": "closed",
    "closed - testing": "closed", "resolved": "closed",
    "open": "open", "pending": "open", "assigned": "open", "in progress": "open",
    "started": "open", "unassigned": "open", "open - dup": "open", "draft": "open",
    "email sent": "open", "scheduled": "open",
    "canceled": "cancelled", "cancelled": "cancelled",
}

_CHANNEL_MAP = {
    "phone": "phone", "phone call": "phone", "constituent call": "phone",
    "web": "web", "online": "web", "internet": "web", "website": "web",
    "mobile": "mobile", "mobile device": "mobile", "mobile/open311": "mobile",
    "salesforce mobile app": "mobile", "citizens connect app": "mobile",
    "city worker app": "mobile", "twitter": "social", "email": "email",
    "in person": "walk_in", "walk-in": "walk_in", "mail": "mail",
}


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def read_jsonl_chunks(path: Path, chunk_size: int = DEFAULT_CHUNK) -> Iterator[pd.DataFrame]:
    buf: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                buf.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(buf) >= chunk_size:
                yield pd.DataFrame(buf)
                buf = []
    if buf:
        yield pd.DataFrame(buf)


def read_csv_chunks(path: Path, chunk_size: int = DEFAULT_CHUNK) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(path, chunksize=chunk_size, dtype=str,
                             low_memory=False, encoding_errors="replace"):
        yield chunk


# ---------------------------------------------------------------------------
# Format-agnostic entry point
# ---------------------------------------------------------------------------
# The same dataset reaches us in two shapes depending on how it was obtained:
#
#   * bulk CSV export, downloaded by hand from the portal's Export button.
#     Columns carry HUMAN-READABLE names: "CaseID", "Media URL", "Status Notes".
#   * Socrata API pull via scripts/download/*.py, written as JSONL.
#     Columns carry API FIELD names: "service_request_id", "media_url".
#
# Both are first-class. resolve_col() normalises case/spacing, and each
# preprocessor passes both spellings to get(), so neither path is privileged.
RAW_SUFFIXES = {".csv", ".jsonl", ".json", ".ndjson", ".tsv",
                ".csv.gz", ".jsonl.gz", ".tsv.gz", ".zip"}


def read_source_chunks(path: Path, chunk_size: int = DEFAULT_CHUNK) -> Iterator[pd.DataFrame]:
    """Dispatch on file extension. Handles .gz transparently via pandas."""
    name = Path(path).name.lower()
    if name.endswith((".jsonl", ".ndjson", ".jsonl.gz")):
        if name.endswith(".gz"):
            import gzip, json as _json
            buf = []
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        buf.append(_json.loads(line))
                    except Exception:
                        continue
                    if len(buf) >= chunk_size:
                        yield pd.DataFrame(buf); buf = []
            if buf:
                yield pd.DataFrame(buf)
            return
        yield from read_jsonl_chunks(Path(path), chunk_size)
        return
    if name.endswith(".json"):
        # A single JSON array (Socrata's query.json bulk export).
        df = pd.read_json(path)
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size]
        return
    sep = "\t" if ".tsv" in name else ","
    for chunk in pd.read_csv(path, chunksize=chunk_size, dtype=str, sep=sep,
                             low_memory=False, encoding_errors="replace"):
        yield chunk


def find_raw_file(dataset: str, patterns: list[str], explicit: str | None = None) -> Path | None:
    """
    Locate whatever the user actually dropped into data/raw/<dataset>/.

    We cannot dictate the filename — the portals' Export buttons produce names
    like 'rows.csv', '311_Cases_20260913.csv' or 'tmp4myyj_u8.csv'. Rather than
    making the user rename files, we glob for a recognisable pattern and, as a
    last resort, take the single largest data file in the directory.
    """
    if explicit:
        pth = Path(explicit)
        return pth if pth.exists() else None

    raw_dir = p("raw", dataset)
    if not raw_dir.exists():
        return None

    for pat in patterns:
        hits = sorted(raw_dir.glob(pat))
        if hits:
            return max(hits, key=lambda f: f.stat().st_size)

    candidates = [f for f in raw_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in {".csv", ".jsonl", ".json", ".tsv", ".gz"}
                  and not f.name.startswith(".")]
    if candidates:
        best = max(candidates, key=lambda f: f.stat().st_size)
        log.warning("no filename matched %s in %s; falling back to the largest data file: %s",
                    patterns, raw_dir, best.name)
        return best
    return None


def missing_raw_message(dataset: str, patterns: list[str], where: str) -> str:
    return (
        f"\n  No raw file found for '{dataset}'.\n"
        f"    Looked in : {where}\n"
        f"    Expected  : {' or '.join(patterns)} (any filename with a .csv/.jsonl suffix also works)\n"
        f"    Fix       : download the file manually and place it in that directory.\n"
        f"                See DOWNLOAD_INSTRUCTIONS.md, section for {dataset}.\n"
    )



def resolve_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """
    Case- and separator-insensitive column lookup.

    Boston's yearly CSVs are not consistent about casing ('OPEN_DT' vs 'open_dt')
    or spacing across the 2011-2024 range, so exact-name access breaks on
    arbitrary years. This normalises before matching.
    """
    norm = {c.lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_").replace("-", "_")
        if key in norm:
            return norm[key]
    return None


def get(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """Fetch a column by any of its aliases; an all-NA series if absent."""
    c = resolve_col(df, *candidates)
    if c is None:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    return df[c]


def extract_url(series: pd.Series) -> pd.Series:
    """
    Socrata 'url' columns arrive as {"url": "...", "description": "..."} in JSON
    and as a bare string in CSV. Normalise both to a plain string.
    """
    def _one(v):
        if isinstance(v, dict):
            return v.get("url")
        if isinstance(v, str) and v.strip():
            return v.strip()
        return None
    return series.map(_one).astype("string")


# ---------------------------------------------------------------------------
# Normalisers
# ---------------------------------------------------------------------------
def normalise_status(s: pd.Series) -> pd.Series:
    return (s.astype("string").str.strip().str.lower().map(_STATUS_MAP)
            .fillna("other").astype("string"))


def normalise_channel(s: pd.Series) -> pd.Series:
    return (s.astype("string").str.strip().str.lower().map(_CHANNEL_MAP)
            .fillna("other").astype("string"))


def build_incident_id(dataset: str, source_id: pd.Series) -> pd.Series:
    return (dataset + ":" + source_id.astype("string").str.strip()).astype("string")


def compute_sla(df: pd.DataFrame, target_col: str, opened_col: str, stats: CleaningStats) -> pd.Series:
    """
    sla_target_hours = target - opened.

    DERIVED, and only where the publisher actually ships a target/due timestamp.
    Boston (TARGET_DT) and NYC (Due Date) do; SF and Chicago do not, and get NULL.
    Negative windows (target before open) are a source inconsistency and are
    nulled rather than clipped.
    """
    if target_col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")
    hrs = (df[target_col] - df[opened_col]).dt.total_seconds() / 3600.0
    bad = (hrs < 0).fillna(False)
    stats.null("sla_target_negative", int(bad.sum()))
    return hrs.where(~bad)


def compute_sla_met(resolution_hours: pd.Series, sla_hours: pd.Series) -> pd.Series:
    """True/False only when BOTH values are valid. Never defaults to False."""
    both = resolution_hours.notna() & sla_hours.notna()
    return (resolution_hours <= sla_hours).where(both).astype("boolean")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
class IncidentWriter:
    """Append-mode Parquet writer so a 22M-row source never lands in memory."""

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.path = ensure_dir(p("processed", "incidents", f"{dataset}.parquet"))
        self._writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = None
        self.rows = 0

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = finalise_incidents(df)          # conform + provenance guard
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(self.path, self._schema, compression="snappy")
        else:
            table = table.cast(self._schema)
        self._writer.write_table(table)
        self.rows += len(df)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        log.info("wrote %s (%d rows)", self.path, self.rows)


def finish(dataset: str, writer: IncidentWriter, stats: CleaningStats, extra: dict | None = None) -> dict:
    writer.close()
    stats.rows_out = writer.rows
    out = stats.to_dict()
    out["output_parquet"] = str(writer.path)
    if extra:
        out.update(extra)
    sp = ensure_dir(p("processed", "incidents", f"{dataset}.cleaning_stats.json"))
    sp.write_text(json.dumps(out, indent=2))
    log.info("cleaning stats -> %s", sp)
    return out


def apply_common(
    out: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    dataset: str,
    city: str,
    category_col: str,
    subcategory_col: str | None,
    stats: CleaningStats,
    unmapped_accum: list,
) -> pd.DataFrame:
    """Coordinate cleaning, category mapping, and the derived-field block."""
    out = clean_coordinates(out, "latitude", "longitude", city, stats)

    mapped = map_categories(out, dataset, category_col, subcategory_col)
    out["category"] = mapped
    unmapped_accum.append(out.loc[mapped.eq("UNMAPPED"), category_col].astype("string"))

    out["incident_id"] = build_incident_id(dataset, out["source_incident_id"])
    out["source_dataset"] = dataset
    out["city"] = city

    # Fields with NULL provenance — set explicitly so the guard has something to
    # check, and so the intent is visible at the point of construction.
    out["occurred_at"] = pd.NaT            # no source records incident onset
    out["severity"] = pd.NA                # no source provides a severity grade
    out["response_time_hours"] = pd.NA     # no source records first response
    out["priority_label"] = pd.NA          # no public 311 dataset has a genuine priority
    out["priority_label_source"] = pd.NA
    return out


def write_unmapped(dataset: str, raw_category_series: list, category_col: str) -> None:
    if not raw_category_series:
        return
    allvals = pd.concat(raw_category_series, ignore_index=True)
    df = pd.DataFrame({category_col: allvals})
    report_unmapped(df, dataset, category_col, pd.Series(["UNMAPPED"] * len(df)))
