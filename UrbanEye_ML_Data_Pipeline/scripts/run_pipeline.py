#!/usr/bin/env python3
"""
Run the whole pipeline in dependency order.

    python scripts/run_pipeline.py                 # real data in data/raw/
    python scripts/run_pipeline.py --fixtures      # synthetic fixtures (offline smoke test)
    python scripts/run_pipeline.py --from 4-features
    python scripts/run_pipeline.py --only build_ml_dataset
    python scripts/run_pipeline.py --dry-run

The phases encode the real dependency graph:

    2-preprocess -> 3-incidents -> 4-features -> 5-priority -> 6-ml -> 7-validate

You cannot build features before incidents exist, and you cannot score priority
before the feature tables exist, because the score is normalised over the
features that were actually available — run it out of order and every row scores
on its category alone. Each script also fails with an explicit "run X first"
message if its input is missing, so the order is enforced from both ends.

Sources whose raw files are absent are skipped with a warning rather than
aborting the run; reports record which sources were included.

--fixtures runs the identical code against tests/fixtures/. Its outputs are
SYNTHETIC and a marker file is written so they cannot be mistaken for real data.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import ensure_dir, p, repo_root

log = get_logger("run_pipeline")

FIXTURES = {
    "sf311": "tests/fixtures/sf311_cases.jsonl",
    "chicago311": "tests/fixtures/chicago311.jsonl",
    "nyc311": "tests/fixtures/nyc311_2010_2019.jsonl",
}


def build_steps(fixtures: bool, sources: list[str] | None = None) -> list[tuple[str, str, list[str]]]:
    """
    The dependency graph, in order. Each phase consumes the previous one's output:

        2-preprocess   raw -> one normalised table per city
        3-incidents    union -> all_incidents.parquet
        4-features     time features (local city time), then geospatial/density
        5-priority     baseline policy engine + priority_dataset + split
        6-ml           resolution / hotspot / duplicate tables, the consolidated
                       urbaneye_ml view, then one clean dataset per ML task
        7-validate     quality, provenance, leakage, outputs, pipeline audit,
                       per-task audit, statistical audit, mapping candidates

    `sources` restricts phase 2 to the datasets whose raw files are actually
    present, so a partial download runs cleanly instead of failing on the first
    missing city.
    """
    pre, val, feat = "scripts/preprocess", "scripts/validation", "scripts/features"
    steps: list[tuple[str, str, list[str]]] = [
        ("1-verify", "dataset_status", [f"{val}/dataset_status.py"]),
        ("1-verify", "validate_mappings", [f"{val}/validate_mappings.py"]),
    ]
    wanted = sources if sources is not None else ["sf311", "chicago311", "nyc311", "boston311"]
    for ds in ("sf311", "chicago311", "nyc311"):
        if ds not in wanted:
            continue
        argv = [f"{pre}/preprocess_{ds}.py"]
        if fixtures:
            argv += ["--input", FIXTURES[ds]]
        steps.append(("2-preprocess", f"preprocess_{ds}", argv))
    if "boston311" in wanted:
        argv = [f"{pre}/preprocess_boston311.py"]
        if fixtures:
            argv += ["--input-dir", "tests/fixtures"]
        steps.append(("2-preprocess", "preprocess_boston311", argv))

    steps += [
        ("3-incidents", "build_incidents", [f"{pre}/build_incidents.py"]),
        ("4-features", "build_time_features", [f"{pre}/build_time_features.py"]),
        ("4-features", "build_geo_features", [f"{pre}/build_geo_features.py"]),
        ("5-priority", "build_priority_features",
         [f"{pre}/build_priority_features.py", "--accept-draft-config"]),
        ("6-ml", "build_resolution_dataset", [f"{pre}/build_resolution_dataset.py"]),
        ("6-ml", "build_hotspot_dataset", [f"{pre}/build_hotspot_dataset.py"]),
        ("6-ml", "build_duplicate_pairs", [f"{pre}/build_duplicate_pairs.py"]),
        ("6-ml", "build_ml_dataset", [f"{feat}/build_ml_dataset.py"]),
        ("6-ml", "build_task_datasets", [f"{feat}/build_task_datasets.py"]),
        ("7-validate", "data_quality_report", [f"{val}/data_quality_report.py"]),
        ("7-validate", "validate_provenance", [f"{val}/validate_provenance.py"]),
        ("7-validate", "validate_leakage", [f"{val}/validate_leakage.py"]),
        ("7-validate", "validate_outputs", [f"{val}/validate_outputs.py"]),
        ("7-validate", "audit_pipeline", [f"{val}/audit_pipeline.py"]),
        ("7-validate", "audit_task_datasets", [f"{val}/audit_task_datasets.py"]),
        ("7-validate", "audit_statistics", [f"{val}/audit_statistics.py"]),
        ("7-validate", "report_category_candidates", [f"{val}/report_category_candidates.py"]),
    ]
    return steps


def raw_present() -> dict[str, bool]:
    out = {}
    for ds in ("sf311", "boston311", "chicago311", "nyc311"):
        d = p("raw", ds)
        files = [f for f in d.glob("*") if f.is_file() and f.name != ".gitkeep"] if d.exists() else []
        out[ds] = bool(files)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", action="store_true",
                    help="run against tests/fixtures instead of data/raw (offline smoke test)")
    ap.add_argument("--from", dest="from_phase", default=None, help="start at a phase, e.g. 4-geo")
    ap.add_argument("--only", default=None, help="run a single step by name")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    root = repo_root()
    os.chdir(root)
    present = raw_present()

    sources = None
    if not args.fixtures:
        missing = [k for k, v in present.items() if not v]
        found = [k for k, v in present.items() if v]
        if not found:
            log.error("No raw files found for any dataset: %s", ", ".join(missing))
            log.error("Download them first (see DATASET_DOWNLOAD_GUIDE.md) into data/raw/<dataset>/.")
            log.error("To smoke-test without real data: python scripts/run_pipeline.py --fixtures")
            return 2
        if missing:
            # A partial corpus is a legitimate state (a city may not be downloaded
            # yet). Skip it loudly rather than refusing to build anything.
            log.warning("no raw files for %s — those cities are SKIPPED, not failed. "
                        "Every report records which sources were included.",
                        ", ".join(missing))
        sources = found
    else:
        log.warning("FIXTURE MODE — all outputs are SYNTHETIC and are not real data")

    steps = build_steps(args.fixtures, sources)
    if args.from_phase:
        idx = next((i for i, (ph, _, _) in enumerate(steps) if ph.startswith(args.from_phase)), None)
        if idx is None:
            log.error("unknown phase %s; phases: %s", args.from_phase, sorted({s[0] for s in steps}))
            return 2
        steps = steps[idx:]
    if args.only:
        steps = [s for s in steps if s[1] == args.only]
        if not steps:
            log.error("unknown step %s", args.only)
            return 2

    results, t0 = [], time.time()
    for phase, name, argv in steps:
        cmd = [sys.executable] + argv
        log.info("[%s] %s", phase, name)
        if args.dry_run:
            results.append({"phase": phase, "step": name, "command": " ".join(cmd), "status": "DRY_RUN"})
            continue
        st = time.time()
        proc = subprocess.run(cmd, cwd=root)
        results.append({"phase": phase, "step": name, "command": " ".join(cmd),
                        "returncode": proc.returncode, "seconds": round(time.time() - st, 2),
                        "status": "OK" if proc.returncode == 0 else "FAILED"})
        if proc.returncode != 0:
            log.error("step %s failed (rc=%d)", name, proc.returncode)
            if not args.continue_on_error:
                break

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "FIXTURES_SYNTHETIC" if args.fixtures else "REAL_DATA",
        "data_source_warning": ("Outputs derive from SYNTHETIC test fixtures and are NOT real data."
                                if args.fixtures else
                                "Outputs derive from the raw files present in data/raw/."),
        "raw_files_present": present,
        "total_seconds": round(time.time() - t0, 2),
        "steps": results,
        "failed_steps": [r["step"] for r in results if r.get("status") == "FAILED"],
    }
    dest = ensure_dir(p("reports", "pipeline_run.json"))
    dest.write_text(json.dumps(summary, indent=2))

    if args.fixtures and not args.dry_run:
        ensure_dir(p("processed", "SYNTHETIC_FIXTURE_OUTPUT.txt")).write_text(
            "The contents of data/processed/ were generated from SYNTHETIC TEST FIXTURES.\n"
            "They are NOT real data. Delete this directory before processing real downloads.\n"
            f"Generated: {datetime.now(timezone.utc).isoformat()}\n")

    ok = sum(1 for r in results if r.get("status") in ("OK", "DRY_RUN"))
    log.info("%d/%d steps OK in %.1fs — report: %s", ok, len(results), summary["total_seconds"], dest)
    return 0 if not summary["failed_steps"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
