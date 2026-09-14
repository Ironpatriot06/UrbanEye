"""
Data manifests.

Every raw file this pipeline writes gets a manifest entry so that a future run
can prove the source has not changed underneath us. Manifests live in
data/manifests/<dataset>.json and are append-and-replace by file name.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import p, ensure_dir, load_config

# Files above this size get a sampled checksum rather than a full one: hashing a
# 6 GB CSV costs minutes and buys little for a change-detection use case.
FULL_CHECKSUM_MAX_BYTES = 2 * 1024**3
_SAMPLE_CHUNK = 1024 * 1024


def sha256_file(path: Path, full: bool | None = None) -> dict[str, Any]:
    """Return {algo, value, mode}. mode is 'full' or 'sampled'."""
    path = Path(path)
    size = path.stat().st_size
    if full is None:
        full = size <= FULL_CHECKSUM_MAX_BYTES
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        if full:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
            mode = "full"
        else:
            # head, middle, tail + size — enough to detect a changed export
            for offset in (0, max(0, size // 2 - _SAMPLE_CHUNK // 2), max(0, size - _SAMPLE_CHUNK)):
                fh.seek(offset)
                h.update(fh.read(_SAMPLE_CHUNK))
            h.update(str(size).encode())
            mode = "sampled"
    return {"algo": "sha256", "value": h.hexdigest(), "mode": mode}


def record(
    dataset: str,
    file_path: Path,
    source_url: str,
    *,
    row_count: int | None = None,
    image_count: int | None = None,
    annotation_count: int | None = None,
    extra: dict | None = None,
) -> dict:
    """Write/replace a manifest entry for one raw file. Returns the entry."""
    cfg = load_config()
    ds_cfg = cfg["datasets"].get(dataset, {})
    lic = ds_cfg.get("licence", {})
    file_path = Path(file_path)

    entry = {
        "dataset": dataset,
        "file_name": file_path.name,
        "file_path": str(file_path.relative_to(_root_of(file_path))) if _is_under_repo(file_path) else str(file_path),
        "source_url": source_url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_size_bytes": file_path.stat().st_size if file_path.exists() else None,
        "checksum": sha256_file(file_path) if file_path.exists() else None,
        "row_count": row_count,
        "image_count": image_count,
        "annotation_count": annotation_count,
        "licence": {
            "name": lic.get("name"),
            "url": lic.get("url"),
            "decision": lic.get("decision"),
        },
        "pipeline_version": cfg.get("pipeline_version"),
        "schema_version": cfg.get("schema_version"),
    }
    if extra:
        entry["extra"] = extra

    mpath = ensure_dir(p("manifests", f"{dataset}.json"))
    existing: list[dict] = []
    if mpath.exists():
        try:
            existing = json.loads(mpath.read_text())
        except json.JSONDecodeError:
            existing = []
    existing = [e for e in existing if e.get("file_name") != entry["file_name"]]
    existing.append(entry)
    existing.sort(key=lambda e: e.get("file_name", ""))
    mpath.write_text(json.dumps(existing, indent=2))
    return entry


def record_failure(dataset: str, source_url: str, method: str, error: str, status: str = "FAILED") -> dict:
    """
    Record an acquisition that did NOT happen.

    This exists so that a blocked or failed download is visible in the manifest
    rather than simply absent. An absent dataset and a failed dataset look
    identical on disk; they are not the same thing.
    """
    entry = {
        "dataset": dataset,
        "status": status,
        "source_url": source_url,
        "method": method,
        "error": error,
        "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    mpath = ensure_dir(p("manifests", f"{dataset}.FAILED.json"))
    existing = []
    if mpath.exists():
        try:
            existing = json.loads(mpath.read_text())
        except json.JSONDecodeError:
            existing = []
    existing.append(entry)
    mpath.write_text(json.dumps(existing, indent=2))
    return entry


def _root_of(path: Path) -> Path:
    from .paths import repo_root
    return repo_root()


def _is_under_repo(path: Path) -> bool:
    from .paths import repo_root
    try:
        Path(path).resolve().relative_to(repo_root())
        return True
    except ValueError:
        return False


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n} B"


def free_space_bytes(path: Path | None = None) -> int:
    from .paths import repo_root
    st = os.statvfs(str(path or repo_root()))
    return st.f_bavail * st.f_frsize
