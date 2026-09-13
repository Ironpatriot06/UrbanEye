'use client';
/* ======================================================
   Detailed incident history — the audit trail.

   This is the companion to WorkflowTimeline, not a replacement for it.
   WorkflowTimeline answers "where is this incident in the lifecycle"; this
   answers "what actually happened, when, and who did it".

   Every role sees the same component. The backend decides what each role is
   allowed to read, so there is no client-side filtering here to be bypassed:
   citizens simply receive a shorter list. `showActorRole` only adds the actor's
   role chip for the admin audit view — it hides nothing the API sent.

   Rendered as a <ol> of real list items so screen readers announce it as the
   ordered sequence of events that it is.
   ====================================================== */

import { useCallback, useEffect, useState } from 'react';
import {
  HISTORY_ACTION_ICONS,
  HISTORY_ACTION_LABELS,
  HISTORY_ACTION_TONE,
  HistoryEvent,
  apiGetHistory,
  formatDateTime,
  formatTime,
} from '@/lib/api';

/** Events whose old -> new pair is worth showing as an explicit change chip. */
function ChangeChip({ event }: { event: HistoryEvent }) {
  const { old_value: from, new_value: to } = event;
  if (!to && !from) return null;

  // A value that only appeared (a first SLA, a photo name) reads better as a
  // single chip than as an arrow from nothing.
  if (!from) {
    return <span className="history__chip">{to}</span>;
  }
  if (!to) {
    return <span className="history__chip history__chip--removed">{from}</span>;
  }
  return (
    <span className="history__change">
      <span className="history__chip history__chip--from">{from}</span>
      <span className="history__arrow" aria-hidden="true">
        →
      </span>
      <span className="history__chip history__chip--to">{to}</span>
      <span className="sr-only">changed from {from} to {to}</span>
    </span>
  );
}

function HistoryEntry({
  event,
  showActorRole,
}: {
  event: HistoryEvent;
  showActorRole: boolean;
}) {
  const tone = HISTORY_ACTION_TONE[event.action] ?? 'status';
  const label = HISTORY_ACTION_LABELS[event.action] ?? event.action;
  const icon = HISTORY_ACTION_ICONS[event.action] ?? '•';
  const isSystem = event.actor_role === 'SYSTEM';

  return (
    <li className={`history__item history__item--${tone}`}>
      <span className="history__marker" aria-hidden="true">
        {icon}
      </span>

      <div className="history__body">
        <div className="history__head">
          {/* The full date is in the title/datetime attributes so the visible
              row can stay short without losing the day for older events. */}
          <time
            className="history__time"
            dateTime={event.created_at}
            title={formatDateTime(event.created_at)}
          >
            {formatTime(event.created_at)}
          </time>
          <h5 className="history__label">{label}</h5>
          {showActorRole && (
            <span className={`history__role history__role--${event.actor_role.toLowerCase()}`}>
              {event.actor_role}
            </span>
          )}
        </div>

        <ChangeChip event={event} />

        {event.description && (
          <p className="history__description">{event.description}</p>
        )}

        <p className="history__actor">
          {isSystem ? (
            <>
              <span aria-hidden="true">⚙</span> by the system
            </>
          ) : (
            <>by {event.actor_name ?? 'a removed account'}</>
          )}
          <span className="history__actor-date"> · {formatDateTime(event.created_at)}</span>
        </p>
      </div>
    </li>
  );
}

export function IncidentHistory({
  incidentId,
  showActorRole = false,
  title = 'History',
}: {
  incidentId: number;
  /** Admin audit view: also show each actor's role. */
  showActorRole?: boolean;
  title?: string;
}) {
  const [events, setEvents] = useState<HistoryEvent[] | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setEvents(await apiGetHistory(incidentId));
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : 'Could not load this incident’s history.',
      );
    } finally {
      setLoading(false);
    }
  }, [incidentId]);

  // Reload whenever the incident changes, and whenever the caller remounts
  // this component after an action — a status change or an assignment adds
  // events, and a stale trail would be worse than none.
  useEffect(() => {
    setExpanded(false);
    load();
  }, [load]);

  const COLLAPSE_AFTER = 6;
  const total = events?.length ?? 0;
  const hidden = Math.max(0, total - COLLAPSE_AFTER);
  // Collapsing keeps the most recent events visible: the newest activity is
  // what a reader opening a live incident needs first.
  const visible =
    events && !expanded && hidden > 0 ? events.slice(hidden) : events ?? [];

  return (
    <section className="detail-section" aria-labelledby={`history-heading-${incidentId}`}>
      <div className="history__header">
        <h4 className="detail-section__title" id={`history-heading-${incidentId}`}>
          {title}
        </h4>
        {total > 0 && (
          <span className="history__count">
            {total} {total === 1 ? 'event' : 'events'}
          </span>
        )}
      </div>

      {loading ? (
        <div className="loading-block loading-block--inline" role="status">
          <span className="spinner spinner-sm" />
          <p>Loading history…</p>
        </div>
      ) : error ? (
        <div className="alert alert-error alert-sm" role="alert">
          <p>{error}</p>
          <button type="button" className="btn btn-sm btn-outline mt-1" onClick={load}>
            Try again
          </button>
        </div>
      ) : total === 0 ? (
        <p className="muted-note">No recorded activity for this incident yet.</p>
      ) : (
        <>
          {hidden > 0 && !expanded && (
            <button
              type="button"
              className="history__toggle"
              onClick={() => setExpanded(true)}
              aria-expanded={false}
            >
              Show {hidden} earlier {hidden === 1 ? 'event' : 'events'}
            </button>
          )}

          <ol className="history" aria-label={`Activity on incident ${incidentId}`}>
            {visible.map((event) => (
              <HistoryEntry
                key={event.id}
                event={event}
                showActorRole={showActorRole}
              />
            ))}
          </ol>

          {hidden > 0 && expanded && (
            <button
              type="button"
              className="history__toggle"
              onClick={() => setExpanded(false)}
              aria-expanded
            >
              Show only the {COLLAPSE_AFTER} most recent
            </button>
          )}

          <p className="system-note">
            <span aria-hidden="true">🔒</span> This trail is recorded
            automatically and cannot be edited or deleted by anyone, including
            administrators.
          </p>
        </>
      )}
    </section>
  );
}
