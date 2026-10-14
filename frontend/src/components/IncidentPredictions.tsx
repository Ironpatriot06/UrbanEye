'use client';
/* ======================================================
   Experimental ML predictions for one incident.

   Shown to every role that can open the incident; the backend applies the
   same access rule as for the incident itself. Every value rendered here comes
   from /api/v1/incidents/{id}/predictions — nothing is computed, defaulted or
   invented in the browser. The panel always says how a number was produced
   (ML model vs a named baseline), shows the backend's warnings, and renders an
   explicit "unavailable" state instead of a number when there is none.

   A failure to load predictions is contained here: the rest of the incident
   view keeps working.
   ====================================================== */

import { useEffect, useState } from 'react';
import {
  IncidentPredictions as Predictions,
  PredictionReason,
  PredictionStatus,
  ResolutionPrediction,
  SlaPrediction,
  apiGetPredictions,
  formatDateTime,
  formatHours,
} from '@/lib/api';

const STATUS_LABEL: Record<PredictionStatus, string> = {
  AVAILABLE: 'Model',
  EXPERIMENTAL: 'Experimental',
  FALLBACK: 'Baseline, not ML',
  UNAVAILABLE: 'Unavailable',
};

const STATUS_BADGE: Record<PredictionStatus, string> = {
  AVAILABLE: 'badge-accent',
  EXPERIMENTAL: 'badge-sla-at_risk',
  FALLBACK: 'badge-neutral',
  UNAVAILABLE: 'badge-sla-none',
};

const BAND_BADGE: Record<string, string> = {
  LOW: 'badge-low',
  MEDIUM: 'badge-medium',
  HIGH: 'badge-high',
};

function pct(v: number | null | undefined): string {
  return v == null ? '—' : `${Math.round(v * 100)}%`;
}

function Reasons({ reasons }: { reasons: PredictionReason[] }) {
  if (!reasons.length) return null;
  return (
    <ul className="prediction__reasons">
      {reasons.map((r, i) => (
        <li key={`${r.code}-${i}`}>
          {r.message} <span className="prediction__code">({r.code})</span>
        </li>
      ))}
    </ul>
  );
}

function Header({ title, status }: { title: string; status: PredictionStatus }) {
  return (
    <div className="prediction__head">
      <span className="prediction__title">{title}</span>
      <span className={`badge ${STATUS_BADGE[status]}`}>{STATUS_LABEL[status]}</span>
    </div>
  );
}

function ResolutionBlock({ p }: { p: ResolutionPrediction }) {
  const isModel = p.method === 'ML_MODEL';
  return (
    <div className="prediction">
      <Header title="Time to resolve" status={p.status} />
      {p.estimate_hours == null ? (
        <p className="muted-note">No estimate can be given for this incident.</p>
      ) : (
        <>
          <p className="prediction__value">
            {formatHours(p.estimate_hours)}
            <span className="prediction__sub">
              {isModel ? ' typical estimate from the model' : ` — ${p.method_description}`}
              {!isModel && p.baseline_sample_size != null && ` (${p.baseline_sample_size} incidents)`}
            </span>
          </p>
          {isModel && p.range_p10_hours != null && p.range_p90_hours != null && (
            <p className="prediction__sub">
              Model range {formatHours(p.range_p10_hours)} – {formatHours(p.range_p90_hours)}
              {p.range_calibrated === false && (
                <>
                  {' '}— <strong>not a reliable interval</strong>
                  {p.range_empirical_coverage != null &&
                    ` (it held about ${pct(p.range_empirical_coverage)} of outcomes in offline tests)`}
                </>
              )}
            </p>
          )}
        </>
      )}
      <Reasons reasons={p.reasons} />
    </div>
  );
}

function SlaBlock({ p }: { p: SlaPrediction }) {
  const isModel = p.method === 'ML_MODEL';
  return (
    <div className="prediction">
      <Header title="Risk of missing the SLA" status={p.status} />
      {isModel && p.breach_probability != null ? (
        <p className="prediction__value">
          {p.risk_band && (
            <span className={`badge ${BAND_BADGE[p.risk_band]}`}>{p.risk_band} risk</span>
          )}{' '}
          {pct(p.breach_probability)}
          <span className="prediction__sub">
            {' '}model probability of missing the {p.sla_hours} h target
          </span>
        </p>
      ) : p.historical_breach_rate != null ? (
        <p className="prediction__value">
          {pct(p.historical_breach_rate)}
          <span className="prediction__sub">
            {' '}— {p.method_description}
            {p.baseline_sample_size != null && ` (${p.baseline_sample_size} incidents)`}. No risk
            band: this is a historical rate, not a prediction for this incident.
          </span>
        </p>
      ) : (
        <p className="muted-note">No breach estimate can be given for this incident.</p>
      )}
      <Reasons reasons={p.reasons} />
    </div>
  );
}

export function IncidentPredictions({
  incidentId,
  slaHours,
  showInputs = false,
}: {
  incidentId: number;
  /** Re-fetch when an admin changes the SLA window. */
  slaHours: number | null;
  /** Admins only: the backend ignores the request for anyone else. */
  showInputs?: boolean;
}) {
  const [data, setData] = useState<Predictions | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError('');
    apiGetPredictions(incidentId, showInputs)
      .then((d) => alive && setData(d))
      .catch((e: Error) => {
        if (!alive) return;
        setData(null);
        setError(e.message || 'Predictions could not be loaded.');
      })
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [incidentId, slaHours, showInputs]);

  const warnings = data
    ? Array.from(new Set([...data.resolution.warnings, ...data.sla.warnings]))
    : [];

  return (
    <section className="detail-section" aria-busy={loading}>
      <h4 className="detail-section__title">Predictions (experimental)</h4>
      {loading && <p className="muted-note">Loading predictions…</p>}
      {!loading && error && (
        <p className="muted-note">Predictions are unavailable right now: {error}</p>
      )}
      {!loading && data && (
        <>
          <ResolutionBlock p={data.resolution} />
          <SlaBlock p={data.sla} />
          {warnings.length > 0 && (
            <div className="alert alert-warning alert-sm" role="note">
              {warnings.map((w) => (
                <p key={w}>{w}</p>
              ))}
            </div>
          )}
          <p className="prediction__meta">
            Uses only information available when the incident was reported (
            {formatDateTime(data.features_as_of)}).
            {data.bundle_version && <> Model bundle {data.bundle_version}.</>}
            {' '}Priority and SLA are never changed by these predictions.
          </p>
          {showInputs && data.inputs && (
            <details className="prediction__inputs">
              <summary>Model inputs (admin)</summary>
              <dl className="field-grid">
                {Object.entries(data.inputs).map(([k, v]) => (
                  <div className="field" key={k}>
                    <dt className="field__label">{k}</dt>
                    <dd className="field__value field__value--mono">
                      {v == null ? 'missing' : String(v)}
                    </dd>
                  </div>
                ))}
              </dl>
            </details>
          )}
        </>
      )}
    </section>
  );
}
