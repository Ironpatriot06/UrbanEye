"""
Response schemas for the ML endpoints (Phase 2C).  See app/ml/__init__.py for
what the statuses mean; every field that can be absent for a given method is
Optional and is null rather than invented.
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["AVAILABLE", "EXPERIMENTAL", "FALLBACK", "UNAVAILABLE"]


class Reason(BaseModel):
    code: str
    message: str
    detail: Optional[Any] = None


class _PredictionBase(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Status
    method: Optional[str] = Field(None, description="ML_MODEL or the BASELINE_* used")
    method_description: Optional[str] = None
    reasons: list[Reason] = []
    fallback_reason: Optional[str] = Field(None, description="Why the model was not used")
    model_version: Optional[str] = None
    bundle_version: Optional[str] = None
    local_reliability_established: bool = Field(
        False, description="False until predictions are evaluated against UrbanEye outcomes")
    warnings: list[str] = []
    baseline_sample_size: Optional[int] = None


class ResolutionPrediction(_PredictionBase):
    task: Literal["resolution"]
    estimate_hours: Optional[float] = None
    range_p10_hours: Optional[float] = Field(
        None, description="Model's 10th-percentile estimate. NOT a calibrated interval bound.")
    range_p90_hours: Optional[float] = None
    range_nominal_coverage: Optional[float] = None
    range_empirical_coverage: Optional[float] = Field(
        None, description="Share of outcomes inside p10–p90 in offline tests (US data)")
    range_calibrated: Optional[bool] = None


class SLAPrediction(_PredictionBase):
    task: Literal["sla"]
    sla_hours: Optional[float] = Field(None, description="The incident's current SLA window")
    breach_probability: Optional[float] = Field(None, description="Model output (ML_MODEL only)")
    risk_band: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    decision_threshold: Optional[float] = None
    historical_breach_rate: Optional[float] = Field(None, description="Baseline output (FALLBACK only)")


class IncidentPredictions(BaseModel):
    incident_id: int
    generated_at: datetime
    features_as_of: datetime = Field(..., description="Predictions use only data before this time")
    incident_status: str
    sla_deadline: Optional[datetime] = None
    service_area: Optional[str] = None
    bundle_version: Optional[str] = None
    resolution: ResolutionPrediction
    sla: SLAPrediction
    inputs: Optional[dict[str, Any]] = Field(None, description="Model inputs (admins only)")
    logged: bool


class HotspotCell(BaseModel):
    cell_id: str
    center_lat: float
    center_lon: float
    bounds: dict[str, float]
    predicted_next_week_count: float
    last_week_count: int
    mean_last_4_weeks: float
    reports_in_history: int


class HotspotForecast(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    area: Optional[str] = None
    category: str
    method: str
    status: Status
    reasons: list[Reason] = []
    alpha: Optional[float] = None
    min_history_weeks: Optional[int] = None
    cell_size_m: Optional[float] = None
    week_definition: Optional[str] = None
    first_complete_week: Optional[str] = None
    last_complete_week: Optional[str] = None
    forecast_week: Optional[str] = None
    weeks_of_history: int = 0
    cells: list[HotspotCell] = []
    cells_total: int = 0
    available_areas: list[str] = []
    model_version: Optional[str] = None
    bundle_version: Optional[str] = None
    baseline_comparison: Optional[str] = None
    generated_at: str
