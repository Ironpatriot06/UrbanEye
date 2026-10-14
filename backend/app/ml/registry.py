"""
Model registry: load the serving bundle once, verify it, report readiness.

Lifecycle
---------
`load_registry()` runs at application startup (main.py lifespan) and again,
lazily, the first time a prediction is requested if startup did not run it.
It never raises: every failure is recorded as a component status and the app
keeps serving incidents.  Nothing is reloaded or retrained while running; a new
bundle means a new directory, a new pinned manifest hash and a restart.

Verification order (a pickle is opened only after all of these pass)
  1. ML_ENABLED
  2. MANIFEST.json exists and its SHA-256 equals ML_BUNDLE_MANIFEST_SHA256
  3. every file listed in the manifest exists with the listed SHA-256
  4. the running numpy / scikit-learn / joblib versions equal the manifest's
     (scikit-learn pickles are not portable across versions)
  5. each model's feature list equals its preprocessor's predictor order
"""

import hashlib
import json
import logging
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.ml.encoding import Encoder

log = logging.getLogger(__name__)


@dataclass
class Component:
    name: str
    ready: bool = False
    error: Optional[str] = None
    model_version: Optional[str] = None
    detail: dict = field(default_factory=dict)


@dataclass
class Registry:
    enabled: bool
    bundle_dir: str
    bundle_version: Optional[str] = None
    manifest: Optional[dict] = None
    error: Optional[str] = None
    loaded_at: Optional[str] = None
    components: dict[str, Component] = field(default_factory=dict)
    # loaded objects (never serialised)
    resolution_models: Optional[dict] = None
    resolution_encoder: Optional[Encoder] = None
    resolution_meta: Optional[dict] = None
    sla_model: Any = None
    sla_encoder: Optional[Encoder] = None
    sla_meta: Optional[dict] = None
    hotspot_meta: Optional[dict] = None
    baselines: Optional[dict] = None
    feature_ranges: Optional[dict] = None

    def ready(self, name: str) -> bool:
        c = self.components.get(name)
        return bool(c and c.ready)

    def public_status(self) -> dict:
        """Readiness report — no file system paths, no stack traces."""
        m = self.manifest or {}
        return {
            "enabled": self.enabled,
            "bundle_version": self.bundle_version,
            "loaded_at": self.loaded_at,
            "error": self.error,
            "training_data_through": m.get("training_data_through"),
            "training_cities": m.get("training_cities"),
            "bundle_created_at": m.get("created_at_utc"),
            "runtime_requirements": m.get("runtime_requirements"),
            "components": {
                k: {"ready": c.ready, "error": c.error, "model_version": c.model_version,
                    **c.detail}
                for k, c in self.components.items()
            },
        }


_lock = threading.Lock()
_registry: Optional[Registry] = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _runtime_versions() -> dict:
    import joblib
    import numpy
    import sklearn

    return {"numpy": numpy.__version__, "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__}


def build_registry(enabled: bool, bundle_dir: str, manifest_sha256: str) -> Registry:
    """Pure function of its arguments (tests call it with temporary bundles)."""
    from datetime import datetime, timezone

    reg = Registry(enabled=enabled, bundle_dir=bundle_dir,
                   loaded_at=datetime.now(timezone.utc).isoformat())
    names = ("resolution", "sla", "hotspot", "baselines", "feature_ranges")
    for n in names:
        reg.components[n] = Component(name=n)
    if not enabled:
        reg.error = "ML_DISABLED"
        for c in reg.components.values():
            c.error = "ML_DISABLED"
        return reg

    root = Path(bundle_dir)
    mpath = root / "MANIFEST.json"

    def fail_all(code: str) -> Registry:
        reg.error = code
        for c in reg.components.values():
            c.error = code
        return reg

    if not mpath.is_file():
        return fail_all("BUNDLE_NOT_FOUND")
    if _sha256(mpath) != manifest_sha256:
        return fail_all("MANIFEST_HASH_MISMATCH")
    try:
        manifest = json.loads(mpath.read_text())
    except ValueError:
        return fail_all("MANIFEST_UNREADABLE")
    reg.manifest = manifest
    reg.bundle_version = manifest.get("bundle_version")

    bad_files = set()
    for rel, digest in manifest.get("files_sha256", {}).items():
        p = root / rel
        if not p.is_file() or _sha256(p) != digest:
            bad_files.add(rel)

    want = manifest.get("runtime_requirements", {})
    have = _runtime_versions()
    version_errors = {k: {"bundle": want.get(k), "runtime": v} for k, v in have.items()
                      if want.get(k) != v}

    def files_ok(prefix: str) -> Optional[str]:
        broken = sorted(f for f in bad_files if f.startswith(prefix))
        return f"FILE_HASH_MISMATCH:{','.join(broken)}" if broken else None

    def load_json(rel: str) -> dict:
        return json.loads((root / rel).read_text())

    # --- JSON components (no pickles involved)
    for name, rel in (("baselines", "baselines.json"), ("feature_ranges", "feature_ranges.json"),
                      ("hotspot", "hotspot/metadata.json")):
        comp = reg.components[name]
        err = files_ok(rel)
        if err:
            comp.error = err
            continue
        try:
            obj = load_json(rel)
        except (OSError, ValueError):
            comp.error = "FILE_UNREADABLE"
            continue
        setattr(reg, {"baselines": "baselines", "feature_ranges": "feature_ranges",
                      "hotspot": "hotspot_meta"}[name], obj)
        comp.ready = True
        if name == "hotspot":
            comp.model_version = obj.get("model_version")
            comp.detail = {"method": "EWMA (non-ML statistic)", "alpha": obj.get("ewma_alpha"),
                           "min_history_weeks": obj.get("min_history_weeks")}

    # --- pickled models
    for name in ("resolution", "sla"):
        comp = reg.components[name]
        err = files_ok(f"{name}/")
        if err:
            comp.error = err
            continue
        if version_errors:
            comp.error = "RUNTIME_VERSION_MISMATCH"
            comp.detail = {"versions": version_errors}
            continue
        try:
            import joblib

            meta = load_json(f"{name}/metadata.json")
            enc = Encoder(load_json(f"{name}/preprocessor.json"))
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=UserWarning)  # version warnings -> fail
                model = joblib.load(root / name / "model.joblib")
            boosters = model.values() if isinstance(model, dict) else [model]
            for b in boosters:
                if list(getattr(b, "feature_names_in_", [])) != enc.predictors:
                    raise ValueError("model feature order differs from the preprocessor")
            if meta.get("features") != enc.predictors:
                raise ValueError("metadata feature list differs from the preprocessor")
        except Exception as exc:  # noqa: BLE001 — recorded, never raised
            log.exception("failed to load %s model", name)
            comp.error = f"LOAD_FAILED:{type(exc).__name__}"
            continue
        comp.ready = True
        comp.model_version = meta.get("model_version")
        comp.detail = {"trained_at": meta.get("created_at_utc"),
                       "training_period": (meta.get("training_data") or {}).get("period"),
                       "n_features": len(enc.predictors)}
        if name == "resolution":
            if set(model) != {"point", "q10", "q90"}:
                comp.ready, comp.error = False, "LOAD_FAILED:unexpected model layout"
                continue
            reg.resolution_models, reg.resolution_encoder, reg.resolution_meta = model, enc, meta
            comp.detail["range_calibrated"] = (meta.get("interval_calibration") or {}).get("calibrated")
        else:
            reg.sla_model, reg.sla_encoder, reg.sla_meta = model, enc, meta
            comp.detail.update(decision_threshold=meta.get("decision_threshold"),
                               risk_bands=meta.get("risk_bands"),
                               sla_hours_range=meta.get("sla_hours_range"))
    if bad_files:
        reg.error = "FILE_HASH_MISMATCH"
    elif version_errors:
        reg.error = "RUNTIME_VERSION_MISMATCH"
    return reg


def load_registry(force: bool = False) -> Registry:
    """Load (once) the bundle named by the settings.  Never raises."""
    global _registry
    with _lock:
        if _registry is not None and not force:
            return _registry
        from app.core.config import get_settings

        s = get_settings()
        try:
            _registry = build_registry(s.ML_ENABLED, s.ML_BUNDLE_DIR, s.ML_BUNDLE_MANIFEST_SHA256)
        except Exception as exc:  # noqa: BLE001
            log.exception("ML registry failed to initialise")
            _registry = Registry(enabled=s.ML_ENABLED, bundle_dir=s.ML_BUNDLE_DIR,
                                 error=f"REGISTRY_FAILED:{type(exc).__name__}")
        return _registry


def get_registry() -> Registry:
    return _registry if _registry is not None else load_registry()


def set_registry(reg: Optional[Registry]) -> None:
    """Test hook: install a specific registry (None -> reload from settings)."""
    global _registry
    with _lock:
        _registry = reg
