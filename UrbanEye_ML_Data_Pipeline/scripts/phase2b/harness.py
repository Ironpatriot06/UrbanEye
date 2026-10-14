"""
Phase 2B — evaluation protocol shared by the resolution and SLA experiments.

ROLLING ORIGIN
--------------
Six origins, two-month test windows, Chicago Sep 2019 → Aug 2020 (the only
city with a long clean timeline). For origin T:

    train  every cleaned row (any city) reported before T
    test   Chicago rows reported in [T, next origin)

LABEL AVAILABILITY
------------------
A model retrained at T can only learn from outcomes known at T. The Phase 2A
training fold contained labels that closed after validation began (8.2%).
Policy "known" (the default) keeps a training row only if its outcome was
observable before T:

    resolution  closed_at < T
    SLA         reported_at + deadline < T (the deadline has passed, so whether
                it was met is known whatever happens later)

Policy "all" reproduces the Phase 2A behaviour, for the sensitivity check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ORIGINS = [pd.Timestamp(x, tz="UTC") for x in
           ("2019-09-01", "2019-11-01", "2020-01-01", "2020-03-01", "2020-05-01",
            "2020-07-01", "2020-09-01")]


def rolling_folds(df: pd.DataFrame, label_policy: str = "known",
                  sla_hours_col: str | None = None, test_source: str = "chicago311"):
    """Yield (name, train_mask, test_mask) for each rolling origin."""
    rep = df["reported_at"]
    for a, b in zip(ORIGINS[:-1], ORIGINS[1:]):
        tr = rep < a
        if label_policy == "known":
            if sla_hours_col is None:
                tr &= df["closed_at"].notna() & (df["closed_at"] < a)
            else:
                tr &= (rep + pd.to_timedelta(df[sla_hours_col], unit="h")) < a
        elif label_policy != "all":
            raise ValueError(label_policy)
        te = (rep >= a) & (rep < b) & (df["source_dataset"] == test_source)
        yield f"{a:%Y-%m}", tr.to_numpy(), te.to_numpy()


def loco_folds(df: pd.DataFrame):
    """Leave-one-city-out: train on two publishers, test on the third."""
    for city in ("sf311", "nyc311", "chicago311"):
        te = (df["source_dataset"] == city).to_numpy()
        yield f"holdout_{city}", ~te, te


def lookup_predict(train: pd.DataFrame, test: pd.DataFrame, keys: list[str], target: str,
                   agg: str = "median", fallback: float | None = None,
                   transform=None) -> np.ndarray:
    """Training-fold aggregate per key combination; unseen keys -> global."""
    y = train[target].astype(float)
    if transform is not None:
        y = transform(y)
    glob = float(getattr(y, agg)()) if fallback is None else fallback
    k_tr = train[keys].astype("string").fillna("NA").reset_index(drop=True)
    k_te = test[keys].astype("string").fillna("NA").reset_index(drop=True)
    table = (k_tr.assign(_y=y.to_numpy()).groupby(keys, observed=True)["_y"].agg(agg)
             .rename("_pred").reset_index())
    out = k_te.merge(table, on=keys, how="left")["_pred"]
    return out.fillna(glob).to_numpy(float)
