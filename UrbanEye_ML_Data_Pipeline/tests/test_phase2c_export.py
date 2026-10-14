"""
Phase 2C: the serving bundle the backend loads must be exactly the Phase 2B
artifacts, and its JSON preprocessor must encode exactly as the original.
Skipped when the bundle or the v2 artifacts are absent (both are regenerable).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.phase2b.common import ROOT, sha256_file  # noqa: E402
from scripts.phase2c.export_serving_bundle import BUNDLE_VERSION  # noqa: E402

BUNDLE = ROOT.parent / "backend" / "ml_bundle" / BUNDLE_VERSION
needs = pytest.mark.skipif(
    not (BUNDLE / "MANIFEST.json").exists() or not (ROOT / "models/sla_v2/model.joblib").exists(),
    reason="serving bundle or v2 artifacts absent")


@needs
def test_bundle_files_match_manifest_and_phase2b_artifacts():
    m = json.loads((BUNDLE / "MANIFEST.json").read_text())
    for rel, digest in m["files_sha256"].items():
        assert sha256_file(BUNDLE / rel) == digest, rel
    for task in ("resolution", "sla"):
        for f in ("model.joblib", "metadata.json"):
            assert sha256_file(ROOT / "models" / f"{task}_v2" / f) == \
                m["source_artifacts_sha256"][task][f] == m["files_sha256"][f"{task}/{f}"]
    assert sha256_file(ROOT / "models/hotspot_v2/metadata.json") == \
        m["files_sha256"]["hotspot/metadata.json"]


@needs
@pytest.mark.parametrize("task", ["resolution", "sla"])
def test_exported_preprocessor_reproduces_original(task):
    import joblib

    pre = joblib.load(ROOT / "models" / f"{task}_v2" / "preprocessor.joblib")
    spec = json.loads((BUNDLE / task / "preprocessor.json").read_text())
    assert spec["predictors"] == pre.predictors and spec["vocabulary"] == pre.vocabulary_
    cat = "model_category" if task == "resolution" else "ue_category"
    df = pd.DataFrame({c: [np.nan] * 4 for c in pre.predictors})
    df[cat] = pd.array(["POTHOLE", "FIRE_HAZARD", None, "FLOOD"], dtype="string")
    df["is_weekend"] = pd.array([True, False, None, True], dtype="boolean")
    df["is_night"] = pd.array([False, False, True, None], dtype="boolean")
    X = pre.transform(df)
    # the documented JSON rules, applied by hand
    for i in range(len(df)):
        for c in spec["categorical"]:
            v = df[c].iloc[i]
            s = "__MISSING__" if pd.isna(v) else str(v)
            s = s if s in spec["vocabulary"][c] else "__OTHER__"
            code = spec["vocabulary"][c].index(s) if s in spec["vocabulary"][c] else -1
            assert X[c].iloc[i] == code, (c, v)
