"""Folder path resolution and config loading."""
from __future__ import annotations
import os, functools
from pathlib import Path
import yaml


def repo_root() -> Path:
    env = os.environ.get("URBANEYE_ROOT")
    if env:
        return Path(env).resolve()
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "config" / "dataset_config.yaml").exists():
            return parent
    raise RuntimeError("Could not locate pipeline root (config/dataset_config.yaml not found)")


def _load(name: str) -> dict:
    with open(repo_root() / "config" / name) as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=1)
def load_config() -> dict:
    return _load("dataset_config.yaml")


@functools.lru_cache(maxsize=1)
def load_priority_config() -> dict:
    return _load("priority_config.yaml")


@functools.lru_cache(maxsize=1)
def load_feature_config() -> dict:
    return _load("feature_config.yaml")


def p(key: str, *parts) -> Path:
    """Resolve a configured path key ('raw','processed','manifests','reports','external')."""
    base = repo_root() / load_config()["paths"][key]
    return base.joinpath(*[str(x) for x in parts])


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    target = path if path.suffix == "" else path.parent
    target.mkdir(parents=True, exist_ok=True)
    return path
