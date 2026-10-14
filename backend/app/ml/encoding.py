"""
The training-time FeaturePreprocessor, re-implemented from its JSON export.

The original (UrbanEye_ML_Data_Pipeline/scripts/train/_common.py) is an ordinal
encoder over pandas columns.  Its rules, which this module reproduces exactly
(tests/test_ml_unit.py checks it against outputs of the original):

  numeric      float; missing or non-numeric -> NaN (the booster's missing branch)
  categorical  str(value) — booleans become 'True'/'False' — with missing ->
               '__MISSING__'; a value outside the training vocabulary ->
               '__OTHER__'; the code is its index in the vocabulary, or -1 when
               the folded value is not in it.
"""

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


class Encoder:
    def __init__(self, spec: Mapping[str, Any]):
        self.predictors: list[str] = list(spec["predictors"])
        self.categorical: set[str] = set(spec["categorical"])
        self.vocabulary: dict[str, dict[str, int]] = {
            c: {lvl: i for i, lvl in enumerate(levels)}
            for c, levels in spec["vocabulary"].items()
        }
        self.missing_token = spec.get("missing_token", "__MISSING__")
        self.other_token = spec.get("other_token", "__OTHER__")
        self.unknown_code = float(spec.get("unknown_code", -1.0))
        unknown = set(self.predictors) - set(spec["numeric"]) - self.categorical
        if unknown:
            raise ValueError(f"preprocessor spec does not type {sorted(unknown)}")

    def _cat(self, column: str, v: Any) -> float:
        s = self.missing_token if _is_missing(v) else (str(bool(v)) if isinstance(v, bool) else str(v))
        codes = self.vocabulary[column]
        if s not in codes:
            s = self.other_token
        return float(codes.get(s, self.unknown_code))

    @staticmethod
    def _num(v: Any) -> float:
        if _is_missing(v) or isinstance(v, bool):
            return float("nan") if _is_missing(v) else float(v)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    def transform(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Rows (dicts) -> float matrix in predictor order.  Absent keys are missing."""
        out = np.empty((len(rows), len(self.predictors)), dtype=np.float64)
        for i, r in enumerate(rows):
            for j, c in enumerate(self.predictors):
                v = r.get(c)
                out[i, j] = self._cat(c, v) if c in self.categorical else self._num(v)
        return out
