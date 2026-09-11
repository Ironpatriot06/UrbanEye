'use client';
import { useState, useEffect, useCallback, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { Navbar } from '@/components/Navbar';
import { PriorityBadge, SlaBadge, StatusBadge } from '@/components/IncidentCard';
import { IncidentImages } from '@/components/IncidentImages';
import { WorkflowTimeline } from '@/components/WorkflowTimeline';
import {
  DetailHeader,
  Field,
  IncidentMetadata,
  SlaSummary,
} from '@/components/IncidentDetailParts';
import {
  ADMIN_TRANSITIONS,
  Agent,
  ApiError,
  Incident,
  IncidentStatus,
  PRIORITY_LABELS,
  PriorityLevel,
  STATUS_LABELS,
  WORKFLOW,
  apiAssignAgent,
  apiGetAgents,
  apiGetIncidents,
  apiSetAgentAvailability,
  apiUpdatePriority,
  apiUpdateSla,
  apiUpdateStatus,
  formatDate,
  formatDateTime,
  getStoredUser,
} from '@/lib/api';

const PRIORITIES: PriorityLevel[] = ['P1', 'P2', 'P3', 'P4'];

/** Availability/workload wording shown next to an agent. */
function agentStateLabel(agent: Agent): { text: string; className: string } {
  if (!agent.is_available) return { text: 'Unavailable', className: 'state-unavailable' };
  if (agent.active_incident_count > 0) {
    return {
      text: `Busy · ${agent.active_incident_count} open`,
      className: 'state-busy',
    };
  }
  return { text: 'Available', className: 'state-available' };
}

export default function AdminPage() {
  const router = useRouter();
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');

  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [statusFilter, setStatusFilter] = useState<'ALL' | IncidentStatus>('ALL');

  const [panelError, setPanelError] = useState('');
  const [panelNotice, setPanelNotice] = useState('');
  const [busy, setBusy] = useState<string | null>(null);

  // Pending unavailable-agent assignment awaiting admin confirmation.
  const [overridePrompt, setOverridePrompt] = useState<Agent | null>(null);

  const [slaDraft, setSlaDraft] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError('');
    try {
      const [incs, agts] = await Promise.all([apiGetIncidents(), apiGetAgents()]);
      setIncidents(incs);
      setAgents(agts);
    } catch (err: unknown) {
      setLoadError(err instanceof Error ? err.message : 'Could not load the dashboard.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const user = getStoredUser();
    if (!user || user.role !== 'ADMIN') {
      router.replace('/login');
      return;
    }
    load();
  }, [router, load]);

  const selected = incidents.find((i) => i.id === selectedId) ?? null;

  // Reset the panel when a DIFFERENT incident is opened. Deliberately keyed on
  // selectedId alone: keying on the incident's sla_hours too would re-run this
  // right after a successful save and wipe the confirmation message.
  useEffect(() => {
    const inc = incidents.find((i) => i.id === selectedId);
    setSlaDraft(inc?.sla_hours != null ? String(inc.sla_hours) : '');
    setPanelError('');
    setPanelNotice('');
    setOverridePrompt(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId]);

  const applyIncident = (updated: Incident) => {
    setIncidents((prev) => prev.map((i) => (i.id === updated.id ? updated : i)));
  };

  const run = async (key: string, fn: () => Promise<void>) => {
    setBusy(key);
    setPanelError('');
    setPanelNotice('');
    try {
      await fn();
    } catch (err: unknown) {
      setPanelError(err instanceof Error ? err.message : 'Something went wrong.');
    } finally {
      setBusy(null);
    }
  };

  const handleStatus = (next: IncidentStatus) =>
    run(`status:${next}`, async () => {
      if (!selected) return;
      applyIncident(await apiUpdateStatus(selected.id, next));
      setPanelNotice(`Status moved to ${STATUS_LABELS[next]}.`);
    });

  const handleAssign = (agentId: number | null, override = false) =>
    run('assign', async () => {
      if (!selected) return;
      try {
        applyIncident(await apiAssignAgent(selected.id, agentId, override));
        setOverridePrompt(null);
        setPanelNotice(
          agentId === null ? 'Agent unassigned.' : 'Agent assigned to this incident.',
        );
        // Workload counters change with every (un)assignment.
        setAgents(await apiGetAgents());
      } catch (err: unknown) {
        // 422 from the backend means the agent is unavailable. Surface the
        // explicit override confirmation rather than failing silently.
        const agent = agents.find((a) => a.id === agentId);
        if (err instanceof ApiError && err.status === 422 && agent && !override) {
          setOverridePrompt(agent);
          return;
        }
        throw err;
      }
    });

  const handleSla = () =>
    run('sla', async () => {
      if (!selected) return;
      const trimmed = slaDraft.trim();
      if (trimmed === '') {
        applyIncident(await apiUpdateSla(selected.id, null));
        setPanelNotice('SLA cleared.');
        return;
      }
      const hours = Number(trimmed);
      if (!Number.isInteger(hours) || hours < 1 || hours > 8760) {
        setPanelError('SLA hours must be a whole number between 1 and 8760.');
        return;
      }
      applyIncident(await apiUpdateSla(selected.id, hours));
      setPanelNotice(`SLA target set to ${hours} hours.`);
    });

  const handlePriority = (level: PriorityLevel) =>
    run('priority', async () => {
      if (!selected) return;
      const updated = await apiUpdatePriority(selected.id, level);
      applyIncident(updated);
      // Re-prioritising also resets the SLA window, so refresh the draft input.
      setSlaDraft(updated.sla_hours != null ? String(updated.sla_hours) : '');
      setPanelNotice(`Priority overridden to ${level} — ${PRIORITY_LABELS[level]}.`);
    });

  const handleAgentAvailability = (agent: Agent, isAvailable: boolean) =>
    run(`avail:${agent.id}`, async () => {
      const updated = await apiSetAgentAvailability(agent.id, isAvailable);
      setAgents((prev) => prev.map((a) => (a.id === updated.id ? updated : a)));
      setPanelNotice(
        `${updated.name} is now ${updated.is_available ? 'available' : 'unavailable'}.`,
      );
    });

  const counts = useMemo(() => {
    const byStatus = Object.fromEntries(
      WORKFLOW.map((s) => [s, incidents.filter((i) => i.status === s).length]),
    ) as Record<IncidentStatus, number>;
    return {
      byStatus,
      breached: incidents.filter((i) => i.sla_status === 'BREACHED').length,
      atRisk: incidents.filter((i) => i.sla_status === 'AT_RISK').length,
      unassigned: incidents.filter((i) => !i.assigned_agent_id).length,
    };
  }, [incidents]);

  const visible =
    statusFilter === 'ALL'
      ? incidents
      : incidents.filter((i) => i.status === statusFilter);

  const nextStatuses: IncidentStatus[] = selected
    ? ADMIN_TRANSITIONS[selected.status] ?? []
    : [];

  return (
    <>
      <Navbar />
      <main className="container page">
        <div className="page-header">
          <div className="page-header__row">
            <div>
              <h1>Admin dashboard</h1>
              <p>Every incident, every agent, and full control of the workflow.</p>
            </div>
            <button className="btn btn-outline btn-sm" onClick={load} disabled={loading}>
              {loading && <span className="spinner spinner-sm" />} Refresh
            </button>
          </div>
        </div>

        <div className="grid-4 mb-3">
          <div className="stat-card">
            <div className="stat-value stat-value--accent">{incidents.length}</div>
            <div className="stat-label">Total incidents</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{counts.unassigned}</div>
            <div className="stat-label">Unassigned</div>
          </div>
          <div className="stat-card">
            <div className="stat-value stat-value--warning">{counts.atRisk}</div>
            <div className="stat-label">SLA at risk</div>
          </div>
          <div className="stat-card">
            <div className="stat-value stat-value--danger">{counts.breached}</div>
            <div className="stat-label">SLA breached</div>
          </div>
        </div>

        {loadError && (
          <div className="alert alert-error mb-3">
            <p>{loadError}</p>
            <button className="btn btn-sm btn-outline mt-1" onClick={load}>
              Try again
            </button>
          </div>
        )}

        <div className={`workspace${selected ? ' workspace--split' : ''}`}>
          <div className="workspace__main">
            <section>
              <div className="section-head">
                <h2>All incidents</h2>
                <div className="form-group form-group--inline">
                  <label className="form-label" htmlFor="filter-status">
                    Status
                  </label>
                  <select
                    id="filter-status"
                    className="form-select form-select--compact"
                    value={statusFilter}
                    onChange={(e) =>
                      setStatusFilter(e.target.value as 'ALL' | IncidentStatus)
                    }
                  >
                    <option value="ALL">All ({incidents.length})</option>
                    {WORKFLOW.map((s) => (
                      <option key={s} value={s}>
                        {STATUS_LABELS[s]} ({counts.byStatus[s]})
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {loading ? (
                <div className="loading-block" role="status">
                  <span className="spinner" />
                  <p>Loading incidents…</p>
                </div>
              ) : visible.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-state__icon" aria-hidden="true">🔍</div>
                  <h3>Nothing to show</h3>
                  <p>
                    {incidents.length === 0
                      ? 'No incidents have been reported yet.'
                      : 'No incidents match the selected status filter.'}
                  </p>
                </div>
              ) : (
                <div className="table-wrapper">
                  <table>
                    <caption className="sr-only">All reported incidents</caption>
                    <thead>
                      <tr>
                        <th scope="col">ID</th>
                        <th scope="col">Title</th>
                        <th scope="col">Priority</th>
                        <th scope="col">Status</th>
                        <th scope="col">SLA</th>
                        <th scope="col">Reporter</th>
                        <th scope="col">Agent</th>
                        <th scope="col">Reported</th>
                      </tr>
                    </thead>
                    <tbody>
                      {visible.map((inc) => (
                        <tr
                          key={inc.id}
                          tabIndex={0}
                          role="button"
                          aria-pressed={inc.id === selectedId}
                          className={inc.id === selectedId ? 'is-selected' : undefined}
                          onClick={() => setSelectedId(inc.id)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault();
                              setSelectedId(inc.id);
                            }
                          }}
                        >
                          <td className="cell-id">#{inc.id}</td>
                          <td className="cell-title">{inc.title}</td>
                          <td>
                            <PriorityBadge
                              level={inc.priority_level}
                              label={inc.priority_label}
                            />
                          </td>
                          <td><StatusBadge status={inc.status} /></td>
                          <td><SlaBadge status={inc.sla_status} /></td>
                          <td className="cell-meta">{inc.reported_by_name ?? '—'}</td>
                          <td className="cell-meta">
                            {inc.assigned_agent_name ?? (
                              <span className="value-empty">Unassigned</span>
                            )}
                          </td>
                          <td className="cell-meta">{formatDate(inc.created_at)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>

            <section className="mt-3">
              <div className="section-head">
                <h2>Agents</h2>
                <p className="section-head__note">
                  Availability is persistent — it is not reset when an agent
                  signs in.
                </p>
              </div>
              {agents.length === 0 ? (
                <div className="empty-state empty-state--compact">
                  <h3>No agents yet</h3>
                  <p>Create agent accounts with the backend seeding script.</p>
                </div>
              ) : (
                <div className="table-wrapper">
                  <table>
                    <caption className="sr-only">Field agents and availability</caption>
                    <thead>
                      <tr>
                        <th scope="col">Name</th>
                        <th scope="col">Email</th>
                        <th scope="col">State</th>
                        <th scope="col">Open work</th>
                        <th scope="col">Last assigned</th>
                        <th scope="col">Availability</th>
                      </tr>
                    </thead>
                    <tbody>
                      {agents.map((a) => {
                        const state = agentStateLabel(a);
                        return (
                          <tr key={a.id} className="row-static">
                            <td>{a.name}</td>
                            <td className="cell-meta">{a.email}</td>
                            <td>
                              <span className={`state-pill ${state.className}`}>
                                {state.text}
                              </span>
                            </td>
                            <td className="cell-meta">{a.active_incident_count}</td>
                            <td className="cell-meta">
                              {a.last_assigned_at ? formatDateTime(a.last_assigned_at) : 'Never'}
                            </td>
                            <td>
                              <button
                                className={`btn btn-sm ${a.is_available ? 'btn-outline' : 'btn-success'}`}
                                onClick={() => handleAgentAvailability(a, !a.is_available)}
                                disabled={busy === `avail:${a.id}`}
                              >
                                {busy === `avail:${a.id}` && (
                                  <span className="spinner spinner-sm" />
                                )}
                                {a.is_available ? 'Mark unavailable' : 'Mark available'}
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          </div>

          {selected && (
            <aside className="workspace__side card detail-panel">
              <DetailHeader incident={selected} onClose={() => setSelectedId(null)} />

              {panelError && (
                <div className="alert alert-error alert-sm mb-2" role="alert">
                  {panelError}
                </div>
              )}
              {panelNotice && !panelError && (
                <div className="alert alert-success alert-sm mb-2" role="status">
                  {panelNotice}
                </div>
              )}

              <WorkflowTimeline status={selected.status} />

              {/* ── Status control ───────────────────────────── */}
              <section className="detail-section">
                <h4 className="detail-section__title">Change status</h4>
                {nextStatuses.length > 0 ? (
                  <div className="button-row">
                    {nextStatuses.map((next) => (
                      <button
                        key={next}
                        className="btn btn-sm btn-primary"
                        onClick={() => handleStatus(next)}
                        disabled={busy !== null}
                      >
                        {busy === `status:${next}` && <span className="spinner spinner-sm" />}
                        {STATUS_LABELS[next]}
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="alert alert-info alert-sm">
                    This incident is closed. Closed is a terminal state.
                  </p>
                )}
                <p className="form-hint">
                  Only transitions valid from {STATUS_LABELS[selected.status]} are
                  offered; the backend rejects anything else.
                </p>
              </section>

              {/* ── Agent assignment ─────────────────────────── */}
              <section className="detail-section">
                <h4 className="detail-section__title">Assigned agent</h4>
                <dl className="field-grid">
                  <Field label="Currently">
                    {selected.assigned_agent_name ?? (
                      <span className="value-empty">Unassigned</span>
                    )}
                  </Field>
                </dl>

                <div className="form-group">
                  <label className="form-label" htmlFor="assign-agent">
                    Assign to
                  </label>
                  <select
                    id="assign-agent"
                    className="form-select"
                    value={selected.assigned_agent_id ?? ''}
                    onChange={(e) =>
                      handleAssign(e.target.value ? Number(e.target.value) : null)
                    }
                    disabled={busy !== null}
                  >
                    <option value="">— Unassigned —</option>
                    {agents.map((a) => {
                      const state = agentStateLabel(a);
                      return (
                        <option key={a.id} value={a.id}>
                          {a.name} — {state.text}
                        </option>
                      );
                    })}
                  </select>
                  <p className="form-hint">
                    Unavailable agents are listed but the server refuses the
                    assignment until you confirm an override.
                  </p>
                </div>

                {overridePrompt && (
                  <div className="alert alert-warning" role="alert">
                    <p>
                      <strong>{overridePrompt.name}</strong> is currently
                      unavailable. Override their availability and assign this
                      incident anyway?
                    </p>
                    <div className="button-row mt-1">
                      <button
                        className="btn btn-sm btn-danger"
                        onClick={() => handleAssign(overridePrompt.id, true)}
                        disabled={busy !== null}
                      >
                        {busy === 'assign' && <span className="spinner spinner-sm" />}
                        Override and assign
                      </button>
                      <button
                        className="btn btn-sm btn-outline"
                        onClick={() => setOverridePrompt(null)}
                        disabled={busy !== null}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </section>

              {/* ── SLA override ─────────────────────────────── */}
              <SlaSummary incident={selected}>
                <div className="form-group form-group--inline mt-2">
                  <label className="form-label" htmlFor="sla-hours">
                    Override SLA (hours)
                  </label>
                  <input
                    id="sla-hours"
                    className="form-input form-input--compact"
                    type="number"
                    min={1}
                    max={8760}
                    placeholder="empty = no SLA"
                    value={slaDraft}
                    onChange={(e) => setSlaDraft(e.target.value)}
                  />
                  <button
                    className="btn btn-sm btn-primary"
                    onClick={handleSla}
                    disabled={busy !== null}
                  >
                    {busy === 'sla' && <span className="spinner spinner-sm" />}
                    Save
                  </button>
                </div>
                <p className="form-hint">
                  The deadline is recalculated from the report time, so an
                  override never grants a fresh window.
                </p>
              </SlaSummary>

              {/* ── Priority override ────────────────────────── */}
              <section className="detail-section">
                <h4 className="detail-section__title">Priority</h4>
                <dl className="field-grid">
                  <Field label="Current">
                    {selected.priority_level} —{' '}
                    {selected.priority_label ||
                      PRIORITY_LABELS[selected.priority_level]}
                  </Field>
                  <Field label="Severity">{selected.severity}</Field>
                </dl>
                <div className="button-row">
                  {PRIORITIES.map((p) => (
                    <button
                      key={p}
                      className={`btn btn-sm ${p === selected.priority_level ? 'btn-primary' : 'btn-outline'}`}
                      onClick={() => handlePriority(p)}
                      disabled={busy !== null || p === selected.priority_level}
                      aria-pressed={p === selected.priority_level}
                    >
                      {p} · {PRIORITY_LABELS[p]}
                    </button>
                  ))}
                </div>
                <p className="form-hint">
                  Priority is normally derived from the category. Overriding it
                  also resets severity and the default SLA window.
                </p>
              </section>

              <IncidentMetadata incident={selected} />
              <IncidentImages
                incidentId={selected.id}
                expectedCount={selected.image_count}
              />
            </aside>
          )}
        </div>
      </main>
    </>
  );
}
