'use client';
/* ======================================================
   Incident badges and the summary card used in list views.
   ====================================================== */

import {
  CATEGORY_LABELS,
  Incident,
  IncidentStatus,
  PRIORITY_LABELS,
  PriorityLevel,
  SLA_LABELS,
  STATUS_LABELS,
  SlaStatus,
  formatDate,
} from '@/lib/api';

export function StatusBadge({ status }: { status: IncidentStatus | string }) {
  const key = status as IncidentStatus;
  return (
    <span className={`badge badge-${String(status).toLowerCase()}`}>
      {STATUS_LABELS[key] ?? status}
    </span>
  );
}

/**
 * Priority is assigned by the system, never chosen by the reporter, so the
 * badge always shows the P-code alongside its meaning.
 */
export function PriorityBadge({
  level,
  label,
}: {
  level: PriorityLevel | string;
  label?: string | null;
}) {
  const key = level as PriorityLevel;
  const text = label || PRIORITY_LABELS[key] || String(level);
  return (
    <span
      className={`badge badge-priority badge-${String(level).toLowerCase()}`}
      title="Priority is assigned automatically by the system"
    >
      <span className="badge__code">{level}</span>
      {text}
    </span>
  );
}

/** Kept for the legacy severity column, which mirrors priority_level. */
export function SeverityBadge({ severity }: { severity: string }) {
  return (
    <span className={`badge badge-${severity.toLowerCase()}`}>
      {severity.charAt(0) + severity.slice(1).toLowerCase()}
    </span>
  );
}

export function SlaBadge({ status }: { status?: SlaStatus | null }) {
  if (!status) return <span className="badge badge-sla-none">No SLA</span>;
  return (
    <span className={`badge badge-sla-${status.toLowerCase()}`}>
      {SLA_LABELS[status] ?? status}
    </span>
  );
}

export function CategoryBadge({ category }: { category: string }) {
  return (
    <span className="badge badge-neutral">
      {CATEGORY_LABELS[category] ?? category}
    </span>
  );
}

export function IncidentCard({
  incident,
  selected = false,
  onClick,
}: {
  incident: Incident;
  selected?: boolean;
  onClick?: () => void;
}) {
  const body = (
    <>
      <div className="incident-card__head">
        <h3 className="incident-card__title">{incident.title}</h3>
        <StatusBadge status={incident.status} />
      </div>
      <div className="badge-row">
        <PriorityBadge
          level={incident.priority_level}
          label={incident.priority_label}
        />
        <CategoryBadge category={incident.category} />
        <SlaBadge status={incident.sla_status} />
        {incident.image_count > 0 && (
          <span className="badge badge-accent">
            <span aria-hidden="true">📷</span>
            {incident.image_count}
          </span>
        )}
      </div>
      {incident.description && (
        <p className="incident-card__desc">{incident.description}</p>
      )}
      <div className="incident-card__foot">
        <span>#{incident.id}</span>
        <span aria-hidden="true">·</span>
        <span>{formatDate(incident.created_at)}</span>
        {incident.assigned_agent_name && (
          <>
            <span aria-hidden="true">·</span>
            <span>Agent: {incident.assigned_agent_name}</span>
          </>
        )}
      </div>
    </>
  );

  if (!onClick) return <article className="card incident-card">{body}</article>;

  return (
    <button
      type="button"
      className={`card incident-card incident-card--clickable${selected ? ' is-selected' : ''}`}
      onClick={onClick}
      aria-pressed={selected}
    >
      {body}
    </button>
  );
}
