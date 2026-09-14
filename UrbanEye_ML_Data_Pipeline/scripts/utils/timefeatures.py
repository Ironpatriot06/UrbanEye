"""
Report-time temporal features, derived in the incident's own LOCAL city time.

WHY LOCAL AND NOT UTC
---------------------
`reported_at` is stored in UTC. That is correct for storage: it makes timestamps
from four cities directly comparable on one axis. It is wrong for features.

"Reported at 23:00" means something (dark, few people about, depot closed). The
UTC hour of that same event is 04:00 in Chicago, 07:00 in San Francisco and
06:00 in New York, so an `hour` column read straight off the UTC timestamp
encodes the city's longitude, not the time of day. `is_night` computed from UTC
would flag a Chicago afternoon as night. A model trained on that learns which
city a row came from and calls it a circadian effect.

So every temporal feature here is computed after converting to the source city's
own timezone, taken from the SINGLE existing mapping in
`config/dataset_config.yaml -> cleaning.timestamps.source_timezones` — the same
mapping the cleaning layer used to localise the raw wall-clock strings in the
first place. There is no second copy of the timezone table anywhere in the code.

`hour_utc` is retained alongside so that the conversion remains auditable.
"""
from __future__ import annotations

import pandas as pd

from .paths import load_config, load_feature_config

# Columns produced by time_features(), in output order.
TIME_FEATURE_COLUMNS = [
    "reported_at_local",
    "local_timezone",
    "hour",
    "day_of_week",
    "month",
    "year",
    "is_weekend",
    "is_night",
    "hour_utc",
    "date_local",
]

# Features a model may use (reported_at_local / local_timezone / date_local are
# metadata for auditing and grouping, not predictors).
TIME_MODEL_FEATURES = ["hour", "day_of_week", "month", "year", "is_weekend", "is_night"]


def source_timezones() -> dict[str, str]:
    """The one authoritative dataset -> IANA timezone mapping."""
    return dict(load_config()["cleaning"]["timestamps"]["source_timezones"])


def night_window() -> tuple[int, int]:
    """(night_start_hour, night_end_hour) in local time, from feature_config."""
    fc = load_feature_config()
    pol = fc.get("temporal_policy", {}) or {}
    return int(pol.get("night_start_hour", 22)), int(pol.get("night_end_hour", 6))


def to_local(ts_utc: pd.Series, source_dataset: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Convert a UTC timestamp series to each row's local city time.

    Returns (local_naive_timestamps, timezone_name). The local series is
    tz-naive on purpose: it is a wall-clock reading, and mixing several tz-aware
    offsets in one pandas column is not representable anyway.

    Rows whose source_dataset has no configured timezone keep NaT rather than
    silently falling back to UTC — an unmapped source is a configuration bug and
    must be visible, not papered over.
    """
    tzmap = source_timezones()
    ts_utc = pd.to_datetime(ts_utc, utc=True, errors="coerce")
    src = source_dataset.astype("string")

    local = pd.Series(pd.NaT, index=ts_utc.index, dtype="datetime64[ns]")
    tzname = pd.Series(pd.NA, index=ts_utc.index, dtype="string")

    for ds, idx in src.groupby(src, dropna=True).groups.items():
        tz = tzmap.get(str(ds))
        if tz is None:
            continue
        # One vectorised conversion per source, not one per row.
        local.loc[idx] = ts_utc.loc[idx].dt.tz_convert(tz).dt.tz_localize(None)
        tzname.loc[idx] = tz
    return local, tzname


def time_features(df: pd.DataFrame, ts_col: str = "reported_at",
                  source_col: str = "source_dataset") -> pd.DataFrame:
    """
    Build the report-time temporal feature block for `df`.

    Every feature is strictly a function of the row's own report timestamp, so
    none of them can leak: they are all knowable the instant the citizen presses
    submit.
    """
    ts_utc = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    local, tzname = to_local(ts_utc, df[source_col])
    n_start, n_end = night_window()

    out = pd.DataFrame(index=df.index)
    out["reported_at_local"] = local
    out["local_timezone"] = tzname
    out["hour"] = local.dt.hour.astype("Int64")
    out["day_of_week"] = local.dt.dayofweek.astype("Int64")
    out["month"] = local.dt.month.astype("Int64")
    out["year"] = local.dt.year.astype("Int64")
    out["is_weekend"] = local.dt.dayofweek.isin([5, 6]).where(local.notna()).astype("boolean")
    hour = local.dt.hour
    if n_start > n_end:      # window wraps midnight, e.g. 22:00 -> 06:00
        night = (hour >= n_start) | (hour < n_end)
    else:
        night = (hour >= n_start) & (hour < n_end)
    out["is_night"] = night.where(local.notna()).astype("boolean")
    out["hour_utc"] = ts_utc.dt.hour.astype("Int64")
    out["date_local"] = local.dt.date.astype("string")
    return out[TIME_FEATURE_COLUMNS]
