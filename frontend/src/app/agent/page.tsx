'use client';
import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { Navbar } from '@/components/Navbar';
import {
  CategoryBadge,
  PriorityBadge,
  SlaBadge,
  StatusBadge,
} from '@/components/IncidentCard';
import { IncidentImages } from '@/components/IncidentImages';
import { WorkflowTimeline } from '@/components/WorkflowTimeline';
import {
  DetailHeader,
  IncidentMetadata,
  PrioritySummary,
  SlaSummary,
} from '@/components/IncidentDetailParts';
import {
  AGENT_TRANSITIONS,
  Agent,
  Incident,
  IncidentStatus,
  STATUS_LABELS,
  apiGetIncidents,
  apiGetMyAgentProfile,
  apiSetAvailability,
  apiUpdateStatus,
  formatDate,
  getStoredUser,
} from '@/lib/api';

export default function AgentPage() {
  const router = useRouter();
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');

  // Availability is server state, not a local default: it persists across
  // logout/login, so it is always read from the API rather than assumed.
  const [profile, setProfile] = useState<Agent | null>(null);
  const [togglingAvail, setTogglingAvail] = useState(false);
  const [availError, setAvailError] = useState('');

  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [updatingStatus, setUpdatingStatus] = useState<IncidentStatus | null>(null);
  const [statusError, setStatusError] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError('');
    try {
      const [incs, me] = await Promise.all([
        apiGetIncidents(),
        apiGetMyAgentProfile(),
      ]);
      setIncidents(incs);
      setProfile(me);
    } catch (err: unknown) {
      setLoadError(err instanceof Error ? err.message : 'Could not load your work queue.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const user = getStoredUser();
    if (!user || user.role !== 'AGENT') {
      router.replace('/login');
      return;
    }
    load();
  }, [router, load]);

  const toggleAvailability = async () => {
    if (!profile) return;
    setTogglingAvail(true);
    setAvailError('');
    try {
      setProfile(await apiSetAvailability(!profile.is_available));
    } catch (err: unknown) {
      setAvailError(err instanceof Error ? err.message : 'Could not update availability.');
    } finally {
      setTogglingAvail(false);
    }
  };

  const handleStatus = async (next: IncidentStatus) => {
    if (selectedId == null) return;
    setUpdatingStatus(next);
    setStatusError('');
    try {
      const updated = await apiUpdateStatus(selectedId, next);
      setIncidents((prev) => prev.map((i) => (i.id === updated.id ? updated : i)));
    } catch (err: unknown) {
      setStatusError(err instanceof Error ? err.message : 'Status update failed.');
    } finally {
      setUpdatingStatus(null);
    }
  };

  const selected = incidents.find((i) => i.id === selectedId) ?? null;
  const allowed: IncidentStatus[] = selected
    ? AGENT_TRANSITIONS[selected.status] ?? []
    : [];
  const available = profile?.is_available ?? false;
  const openCount = incidents.filter(
    (i) => i.status !== 'RESOLVED' && i.status !== 'CLOSED',
  ).length;

  return (
    <>
      <Navbar />
      <main className="container page">
        <div className="page-header">
          <div className="page-header__row">
            <div>
              <h1>Agent dashboard</h1>
              <p>
                Incidents assigned to you. You can move work from assigned to in
                progress, then to resolved.
              </p>
            </div>

            <div className="availability-control">
              <div className="availability-control__state">
                <span className={available ? 'avail-on' : 'avail-off'}>
                  <span aria-hidden="true">{available ? '●' : '○'}</span>{' '}
                  {available ? 'Available' : 'Unavailable'}
                </span>
                <span className="availability-control__hint">
                  {available
                    ? 'You can receive new assignments'
                    : 'You are out of the assignment queue'}
                </span>
              </div>
              <button
                id="btn-toggle-availability"
                className={`btn btn-sm ${available ? 'btn-danger' : 'btn-success'}`}
                onClick={toggleAvailability}
                disabled={togglingAvail || !profile}
              >
                {togglingAvail && <span className="spinner spinner-sm" />}
                {available ? 'Go unavailable' : 'Go available'}
              </button>
            </div>
          </div>
          {availError && (
            <div className="alert alert-error alert-sm mt-2" role="alert">
              {availError}
            </div>
          )}
          <p className="page-header__note">
            Your availability is saved on the server and stays as you set it
            after you sign out and back in.
          </p>
        </div>

        <div className="grid-3 mb-3">
          <div className="stat-card">
            <div className="stat-value">{incidents.length}</div>
            <div className="stat-label">Assigned to me</div>
          </div>
          <div className="stat-card">
            <div className="stat-value stat-value--warning">{openCount}</div>
            <div className="stat-label">Still open</div>
          </div>
          <div className="stat-card">
            <div className="stat-value stat-value--danger">
              {incidents.filter((i) => i.sla_status === 'BREACHED').length}
            </div>
            <div className="stat-label">SLA breached</div>
          </div>
        </div>

        <div className={`workspace${selected ? ' workspace--split' : ''}`}>
          <div className="workspace__main">
            {loading ? (
              <div className="loading-block" role="status">
                <span className="spinner" />
                <p>Loading your assignments…</p>
              </div>
            ) : loadError ? (
              <div className="alert alert-error">
                <p>{loadError}</p>
                <button className="btn btn-sm btn-outline mt-1" onClick={load}>
                  Try again
                </button>
              </div>
            ) : incidents.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state__icon" aria-hidden="true">🧰</div>
                <h3>No assigned incidents</h3>
                <p>
                  {available
                    ? 'New incidents will be routed to you automatically.'
                    : 'You are marked unavailable, so new incidents are going to other agents.'}
                </p>
              </div>
            ) : (
              <div className="table-wrapper">
                <table>
                  <caption className="sr-only">Incidents assigned to you</caption>
                  <thead>
                    <tr>
                      <th scope="col">ID</th>
                      <th scope="col">Title</th>
                      <th scope="col">Category</th>
                      <th scope="col">Priority</th>
                      <th scope="col">Status</th>
                      <th scope="col">SLA</th>
                      <th scope="col">Reported</th>
                    </tr>
                  </thead>
                  <tbody>
                    {incidents.map((inc) => (
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
                        <td><CategoryBadge category={inc.category} /></td>
                        <td>
                          <PriorityBadge
                            level={inc.priority_level}
                            label={inc.priority_label}
                          />
                        </td>
                        <td><StatusBadge status={inc.status} /></td>
                        <td><SlaBadge status={inc.sla_status} /></td>
                        <td className="cell-meta">{formatDate(inc.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {selected && (
            <aside className="workspace__side card detail-panel">
              <DetailHeader incident={selected} onClose={() => setSelectedId(null)} />

              <section className="detail-section">
                <h4 className="detail-section__title">Move this incident forward</h4>
                {statusError && (
                  <div className="alert alert-error alert-sm mb-2" role="alert">
                    {statusError}
                  </div>
                )}
                {allowed.length > 0 ? (
                  <div className="button-row">
                    {allowed.map((next) => (
                      <button
                        key={next}
                        id={`btn-status-${next.toLowerCase()}`}
                        className="btn btn-primary btn-sm"
                        onClick={() => handleStatus(next)}
                        disabled={updatingStatus !== null}
                      >
                        {updatingStatus === next && <span className="spinner spinner-sm" />}
                        Mark {STATUS_LABELS[next]}
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="alert alert-info alert-sm">
                    {selected.status === 'RESOLVED' || selected.status === 'CLOSED'
                      ? 'This incident is finished. Only an admin can reopen or close it.'
                      : `Nothing for you to do while this incident is ${STATUS_LABELS[selected.status]}. An admin moves it on from here.`}
                  </p>
                )}
              </section>

              <WorkflowTimeline status={selected.status} />
              <PrioritySummary incident={selected} />
              <SlaSummary incident={selected}>
                <p className="system-note">
                  <span aria-hidden="true">⚙</span> SLA targets are set by the
                  system and can only be changed by an admin.
                </p>
              </SlaSummary>
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
