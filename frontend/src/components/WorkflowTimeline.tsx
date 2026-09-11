'use client';
/* ======================================================
   Visual incident lifecycle.

   REPORTED → TRIAGED → ASSIGNED → IN_PROGRESS → RESOLVED → CLOSED

   Stages before the current one read as completed, the current stage is
   highlighted, and later stages are muted but still legible.
   ====================================================== */

import { IncidentStatus, STATUS_LABELS, WORKFLOW } from '@/lib/api';

const STAGE_HINTS: Record<IncidentStatus, string> = {
  REPORTED: 'Report received',
  TRIAGED: 'Reviewed and categorised',
  ASSIGNED: 'Assigned to a field agent',
  IN_PROGRESS: 'Agent working on site',
  RESOLVED: 'Work completed',
  CLOSED: 'Closed by the city',
};

export function WorkflowTimeline({ status }: { status: IncidentStatus }) {
  const currentIndex = WORKFLOW.indexOf(status);

  return (
    <section className="detail-section">
      <h4 className="detail-section__title">Workflow</h4>
      <ol className="timeline" aria-label="Incident progress">
        {WORKFLOW.map((stage, i) => {
          const state =
            i < currentIndex ? 'done' : i === currentIndex ? 'current' : 'upcoming';
          return (
            <li key={stage} className={`timeline__step timeline__step--${state}`}>
              <span className="timeline__marker" aria-hidden="true">
                {state === 'done' ? '✓' : i + 1}
              </span>
              <span className="timeline__body">
                <span className="timeline__label">
                  {STATUS_LABELS[stage]}
                  {state === 'current' && (
                    <span className="timeline__now"> · current</span>
                  )}
                </span>
                <span className="timeline__hint">{STAGE_HINTS[stage]}</span>
              </span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
