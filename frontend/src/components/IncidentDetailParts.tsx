'use client';
/* ======================================================
   Shared read-only pieces of an incident detail panel.

   Used by all three dashboards so a citizen, an agent and an admin see the
   same facts about an incident, presented identically. Role-specific controls
   live in the dashboards themselves.
   ====================================================== */

import {
  CATEGORY_LABELS,
  Incident,
  PRIORITY_LABELS,
  PriorityLevel,
  formatDateTime,
  formatDeadlineDelta,
} from '@/lib/api';
import { PriorityBadge, SlaBadge, StatusBadge } from '@/components/IncidentCard';

/** One label/value row in the metadata grid. */
export function Field({
  label,
  children,
  mono = false,
}: {
  label: string;
  children: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="field">
      <dt className="field__label">{label}</dt>
      <dd className={`field__value${mono ? ' field__value--mono' : ''}`}>{children}</dd>
    </div>
  );
}

export function DetailHeader({
  incident,
  onClose,
}: {
  incident: Incident;
  onClose: () => void;
}) {
  return (
    <header className="detail-header">
      <div className="detail-header__top">
        <p className="detail-header__eyebrow">Incident #{incident.id}</p>
        <button
          type="button"
          className="btn btn-sm btn-outline"
          onClick={onClose}
          aria-label={`Close details for incident ${incident.id}`}
        >
          ✕
        </button>
      </div>
      <h2 className="detail-header__title">{incident.title}</h2>
      <div className="badge-row">
        <StatusBadge status={incident.status} />
        <PriorityBadge
          level={incident.priority_level}
          label={incident.priority_label}
        />
        <SlaBadge status={incident.sla_status} />
      </div>
    </header>
  );
}

/**
 * Priority / severity block.
 *
 * Always states that the value came from the system, because citizens no
 * longer choose it and agents cannot change it.
 */
export function PrioritySummary({ incident }: { incident: Incident }) {
  const label =
    incident.priority_label ||
    PRIORITY_LABELS[incident.priority_level as PriorityLevel] ||
    incident.priority_level;

  return (
    <section className="detail-section">
      <h4 className="detail-section__title">Priority</h4>
      <dl className="field-grid">
        <Field label="Priority level">
          {incident.priority_level} — {label}
        </Field>
        <Field label="Severity">{incident.severity}</Field>
      </dl>
      <p className="system-note">
        <span aria-hidden="true">⚙</span> Assigned automatically by the system
        from the incident category. Reporters cannot set it.
      </p>
    </section>
  );
}

/** SLA block — identical for every role; only admins get edit controls. */
export function SlaSummary({
  incident,
  children,
}: {
  incident: Incident;
  children?: React.ReactNode;
}) {
  const delta = formatDeadlineDelta(incident.sla_deadline);

  return (
    <section className="detail-section">
      <h4 className="detail-section__title">Service level agreement</h4>
      {incident.sla_hours == null ? (
        <p className="muted-note">No SLA is set for this incident.</p>
      ) : (
        <dl className="field-grid">
          <Field label="Target">{incident.sla_hours} hours from report</Field>
          <Field label="Status">
            <SlaBadge status={incident.sla_status} />
          </Field>
          <Field label="Deadline">
            {formatDateTime(incident.sla_deadline)}
            {delta && <span className="field__hint"> · {delta}</span>}
          </Field>
        </dl>
      )}
      {children}
    </section>
  );
}

/**
 * Everything else worth knowing about the incident.
 *
 * `showReporter` is off for the citizen's own dashboard, where the reporter is
 * always the viewer.
 */
export function IncidentMetadata({
  incident,
  showReporter = true,
}: {
  incident: Incident;
  showReporter?: boolean;
}) {
  return (
    <section className="detail-section">
      <h4 className="detail-section__title">Details</h4>
      {incident.description ? (
        <p className="detail-description">{incident.description}</p>
      ) : (
        <p className="muted-note">No description was provided.</p>
      )}
      <dl className="field-grid">
        <Field label="Category">
          {CATEGORY_LABELS[incident.category] ?? incident.category}
        </Field>
        <Field label="Source">{incident.source}</Field>
        {showReporter && (
          <Field label="Reported by">
            {incident.reported_by_name ?? 'Unknown'}
          </Field>
        )}
        <Field label="Assigned agent">
          {incident.assigned_agent_name ?? (
            <span className="value-empty">Unassigned</span>
          )}
        </Field>
        <Field label="Reported at">{formatDateTime(incident.created_at)}</Field>
        <Field label="Last updated">{formatDateTime(incident.updated_at)}</Field>
        <Field label="Latitude" mono>
          {incident.latitude.toFixed(6)}
        </Field>
        <Field label="Longitude" mono>
          {incident.longitude.toFixed(6)}
        </Field>
      </dl>
      <a
        className="map-link"
        href={`https://www.openstreetmap.org/?mlat=${incident.latitude}&mlon=${incident.longitude}#map=17/${incident.latitude}/${incident.longitude}`}
        target="_blank"
        rel="noreferrer noopener"
      >
        View location on a map ↗
      </a>
    </section>
  );
}
