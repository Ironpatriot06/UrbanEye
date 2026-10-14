"""
MLPrediction ORM model — the prediction log (Phase 2C shadow evaluation).

One row per distinct prediction served for an incident: task (resolution /
sla), the method that produced it (ML_MODEL or a named baseline), its status,
the model and bundle versions, the reasons and the inputs.  Predictions are
computed as of the incident's created_at, so the same incident, bundle, method
and SLA window always give the same result; `dedupe_key` makes the log record
each of them once however often it is viewed.

Why log at all: Phase 2B's metrics are offline US metrics.  Whether these
predictions are any good for UrbanEye can only be measured by joining this
table to the outcomes that incident_history later records.

Nothing reads this table to make a decision, and nothing here changes an
incident.  Rows are written by app/ml/service.py only.
"""

from sqlalchemy import (Column, DateTime, ForeignKey, Integer, String, Text,
                        UniqueConstraint, func)
from sqlalchemy.dialects.postgresql import JSONB

from app.db.base import Base


class MLPrediction(Base):
    __tablename__ = "ml_predictions"
    __table_args__ = (
        UniqueConstraint("incident_id", "dedupe_key", name="uq_ml_predictions_incident_key"),
    )

    id: int = Column(Integer, primary_key=True)
    incident_id: int = Column(Integer, ForeignKey("incidents.id", ondelete="CASCADE"),
                              nullable=False, index=True)
    task: str = Column(String(20), nullable=False)               # resolution | sla
    status: str = Column(String(20), nullable=False)             # AVAILABLE ... UNAVAILABLE
    method: str = Column(String(64), nullable=True)              # ML_MODEL | BASELINE_* | NULL
    model_version: str = Column(String(64), nullable=True)
    bundle_version: str = Column(String(64), nullable=True)
    sla_hours: int = Column(Integer, nullable=True)
    fallback_reason: str = Column(String(64), nullable=True)
    reasons = Column(JSONB, nullable=False, default=list)
    output = Column(JSONB, nullable=False)                       # the values served
    features = Column(JSONB, nullable=True)                      # model inputs, if any
    features_as_of = Column(DateTime(timezone=True), nullable=False)
    generated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    dedupe_key: str = Column(Text, nullable=False)
