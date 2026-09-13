'use client';
/**
 * Admin → User Management.
 *
 * Shows every account and lets an admin grant USER / AGENT / ADMIN, or
 * deactivate an account.
 *
 * The role check on mount only decides what to render. It is not the security
 * boundary: /api/v1/admin/users answers 401 without a token and 403 to a USER
 * or an AGENT, whatever this component believes. Clearing localStorage by hand
 * gets you this page's chrome and an empty table full of errors.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Navbar } from '@/components/Navbar';
import {
  AUTH_METHOD_LABELS,
  AdminUser,
  ROLES,
  ROLE_LABELS,
  Role,
  USER_AUDIT_LABELS,
  UserAuditEvent,
  apiGetUserAudit,
  apiGetUsers,
  apiSetUserActive,
  apiSetUserRole,
  formatDate,
  formatDateTime,
  getStoredUser,
} from '@/lib/api';

const ROLE_BADGE: Record<Role, string> = {
  USER: 'badge-user',
  AGENT: 'badge-agent',
  ADMIN: 'badge-admin',
};

/** A pending role change awaiting confirmation. */
type PendingChange = { user: AdminUser; role: Role };

export default function UserManagementPage() {
  const router = useRouter();

  const [users, setUsers] = useState<AdminUser[]>([]);
  const [audit, setAudit] = useState<UserAuditEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');

  const [roleFilter, setRoleFilter] = useState<'ALL' | Role>('ALL');
  const [search, setSearch] = useState('');

  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState<number | null>(null);

  // Granting ADMIN, and deactivating an account, are the two changes worth
  // stopping for. Promoting to AGENT is routine and confirms inline.
  const [pending, setPending] = useState<PendingChange | null>(null);
  const [pendingDeactivate, setPendingDeactivate] = useState<AdminUser | null>(null);

  const [showAudit, setShowAudit] = useState(false);

  // The signed-in admin — used to mark their own row, which the backend
  // refuses to let them edit.
  const [selfId, setSelfId] = useState<number | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError('');
    try {
      const [list, events] = await Promise.all([
        apiGetUsers(),
        apiGetUserAudit().catch(() => [] as UserAuditEvent[]),
      ]);
      setUsers(list);
      setAudit(events);
    } catch (err: unknown) {
      setLoadError(err instanceof Error ? err.message : 'Could not load users.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const stored = getStoredUser();
    if (!stored || stored.role !== 'ADMIN') {
      router.replace('/login');
      return;
    }
    setSelfId(stored.user_id);
    load();
  }, [router, load]);

  const applyUser = (updated: AdminUser) =>
    setUsers((prev) => prev.map((u) => (u.id === updated.id ? updated : u)));

  const run = async (userId: number, fn: () => Promise<void>) => {
    setBusy(userId);
    setError('');
    setNotice('');
    try {
      await fn();
      // Role and status changes are audited; refresh the trail so the change
      // the admin just made is visible without a page reload.
      apiGetUserAudit()
        .then(setAudit)
        .catch(() => undefined);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Something went wrong.');
    } finally {
      setBusy(null);
    }
  };

  const changeRole = (user: AdminUser, role: Role) =>
    run(user.id, async () => {
      const updated = await apiSetUserRole(user.id, role);
      applyUser(updated);
      setPending(null);
      setNotice(`${updated.name} is now ${ROLE_LABELS[updated.role]}.`);
    });

  const changeActive = (user: AdminUser, isActive: boolean) =>
    run(user.id, async () => {
      const updated = await apiSetUserActive(user.id, isActive);
      applyUser(updated);
      setPendingDeactivate(null);
      setNotice(
        `${updated.name}'s account is ${updated.is_active ? 'active' : 'deactivated'}.`,
      );
    });

  /** Route a requested role through confirmation when it grants ADMIN. */
  const requestRole = (user: AdminUser, role: Role) => {
    if (role === user.role) return;
    setError('');
    setNotice('');
    if (role === 'ADMIN') {
      setPending({ user, role });
      return;
    }
    changeRole(user, role);
  };

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return users.filter((u) => {
      if (roleFilter !== 'ALL' && u.role !== roleFilter) return false;
      if (!needle) return true;
      return (
        u.name.toLowerCase().includes(needle) || u.email.toLowerCase().includes(needle)
      );
    });
  }, [users, roleFilter, search]);

  const counts = useMemo(
    () => ({
      total: users.length,
      admins: users.filter((u) => u.role === 'ADMIN' && u.is_active).length,
      agents: users.filter((u) => u.role === 'AGENT').length,
      inactive: users.filter((u) => !u.is_active).length,
    }),
    [users],
  );

  return (
    <>
      <Navbar />
      <main className="container page">
        <div className="page-header">
          <div className="page-header__row">
            <div>
              <h1>User management</h1>
              <p>
                Every account on UrbanEye+. New registrations arrive as citizens —
                grant agent or admin access here.
              </p>
            </div>
            <div className="button-row">
              <button
                className="btn btn-outline btn-sm"
                onClick={() => router.push('/admin')}
              >
                ← Incidents
              </button>
              <button className="btn btn-outline btn-sm" onClick={load} disabled={loading}>
                {loading && <span className="spinner spinner-sm" />} Refresh
              </button>
            </div>
          </div>
        </div>

        <div className="grid-4 mb-3">
          <div className="stat-card">
            <div className="stat-value stat-value--accent">{counts.total}</div>
            <div className="stat-label">Total accounts</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{counts.admins}</div>
            <div className="stat-label">Active admins</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{counts.agents}</div>
            <div className="stat-label">Agents</div>
          </div>
          <div className="stat-card">
            <div className="stat-value stat-value--warning">{counts.inactive}</div>
            <div className="stat-label">Deactivated</div>
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
        {error && (
          <div className="alert alert-error mb-3" role="alert">
            {error}
          </div>
        )}
        {notice && (
          <div className="alert alert-success mb-3" role="status">
            {notice}
          </div>
        )}

        {pending && (
          <div className="alert alert-warning mb-3" role="alertdialog">
            <p>
              <strong>Grant full administrator access?</strong>
            </p>
            <p>
              {pending.user.name} ({pending.user.email}) will be able to manage
              every incident, every agent, and everyone&apos;s role — including
              yours.
            </p>
            <div className="button-row mt-1">
              <button
                className="btn btn-sm btn-danger"
                onClick={() => changeRole(pending.user, pending.role)}
                disabled={busy === pending.user.id}
              >
                {busy === pending.user.id && <span className="spinner spinner-sm" />}
                Yes, grant admin
              </button>
              <button className="btn btn-sm btn-outline" onClick={() => setPending(null)}>
                Cancel
              </button>
            </div>
          </div>
        )}

        {pendingDeactivate && (
          <div className="alert alert-warning mb-3" role="alertdialog">
            <p>
              <strong>Deactivate {pendingDeactivate.name}?</strong>
            </p>
            <p>
              They will be signed out immediately and cannot sign in again until
              an admin reactivates them. Their incidents and history are kept.
            </p>
            <div className="button-row mt-1">
              <button
                className="btn btn-sm btn-danger"
                onClick={() => changeActive(pendingDeactivate, false)}
                disabled={busy === pendingDeactivate.id}
              >
                {busy === pendingDeactivate.id && <span className="spinner spinner-sm" />}
                Deactivate
              </button>
              <button
                className="btn btn-sm btn-outline"
                onClick={() => setPendingDeactivate(null)}
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        <section>
          <div className="section-head">
            <h2>Accounts</h2>
            <div className="filter-row">
              <div className="form-group form-group--inline">
                <label className="form-label" htmlFor="filter-role">
                  Role
                </label>
                <select
                  id="filter-role"
                  className="form-select form-select--compact"
                  value={roleFilter}
                  onChange={(e) => setRoleFilter(e.target.value as 'ALL' | Role)}
                >
                  <option value="ALL">All ({users.length})</option>
                  {ROLES.map((r) => (
                    <option key={r} value={r}>
                      {ROLE_LABELS[r]} ({users.filter((u) => u.role === r).length})
                    </option>
                  ))}
                </select>
              </div>
              <div className="form-group form-group--inline">
                <label className="form-label" htmlFor="filter-search">
                  Search
                </label>
                <input
                  id="filter-search"
                  className="form-input form-input--compact"
                  type="search"
                  placeholder="Name or email"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
              </div>
            </div>
          </div>

          {loading ? (
            <div className="loading-block" role="status">
              <span className="spinner" />
              <p>Loading accounts…</p>
            </div>
          ) : visible.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state__icon" aria-hidden="true">👥</div>
              <h3>No accounts match</h3>
              <p>
                {users.length === 0
                  ? 'No accounts exist yet.'
                  : 'Try a different role filter or search term.'}
              </p>
            </div>
          ) : (
            <div className="table-wrapper">
              <table className="user-table">
                <caption className="sr-only">All UrbanEye+ accounts</caption>
                <thead>
                  <tr>
                    <th scope="col">Name</th>
                    <th scope="col">Email</th>
                    <th scope="col">Role</th>
                    <th scope="col">Sign-in</th>
                    <th scope="col">Created</th>
                    <th scope="col">Last login</th>
                    <th scope="col">Status</th>
                    <th scope="col">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {visible.map((u) => {
                    const isSelf = u.id === selfId;
                    return (
                      <tr key={u.id} className="row-static">
                        <td data-label="Name">
                          <span className="user-name">{u.name}</span>
                          {isSelf && <span className="badge badge-neutral ml-1">You</span>}
                        </td>
                        <td data-label="Email" className="cell-email">
                          {u.email}
                        </td>
                        <td data-label="Role">
                          <span className={`badge ${ROLE_BADGE[u.role]}`}>
                            {ROLE_LABELS[u.role]}
                          </span>
                        </td>
                        <td data-label="Sign-in" className="cell-meta">
                          {u.auth_methods.length === 0
                            ? '—'
                            : u.auth_methods
                                .map((m) => AUTH_METHOD_LABELS[m] ?? m)
                                .join(' + ')}
                        </td>
                        <td data-label="Created" className="cell-meta">
                          {formatDate(u.created_at)}
                        </td>
                        <td data-label="Last login" className="cell-meta">
                          {u.last_login_at ? (
                            formatDateTime(u.last_login_at)
                          ) : (
                            <span className="value-empty">Never</span>
                          )}
                        </td>
                        <td data-label="Status">
                          <span
                            className={`state-pill ${
                              u.is_active ? 'state-available' : 'state-unavailable'
                            }`}
                          >
                            {u.is_active ? 'Active' : 'Deactivated'}
                          </span>
                        </td>
                        <td data-label="Actions">
                          {isSelf ? (
                            <span className="form-hint">
                              You cannot change your own role.
                            </span>
                          ) : (
                            <div className="row-actions">
                              <label className="sr-only" htmlFor={`role-${u.id}`}>
                                Role for {u.name}
                              </label>
                              <select
                                id={`role-${u.id}`}
                                className="form-select form-select--compact"
                                value={u.role}
                                disabled={busy === u.id}
                                onChange={(e) => requestRole(u, e.target.value as Role)}
                              >
                                {ROLES.map((r) => (
                                  <option key={r} value={r}>
                                    {ROLE_LABELS[r]}
                                  </option>
                                ))}
                              </select>
                              <button
                                className="btn btn-sm btn-outline"
                                disabled={busy === u.id}
                                onClick={() =>
                                  u.is_active
                                    ? setPendingDeactivate(u)
                                    : changeActive(u, true)
                                }
                              >
                                {u.is_active ? 'Deactivate' : 'Reactivate'}
                              </button>
                            </div>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="mt-3">
          <div className="section-head">
            <h2>Account activity</h2>
            <button
              className="btn btn-sm btn-outline"
              onClick={() => setShowAudit((v) => !v)}
              aria-expanded={showAudit}
            >
              {showAudit ? 'Hide' : `Show (${audit.length})`}
            </button>
          </div>
          <p className="section-head__note">
            Registrations, role changes and account activations — kept separately
            from incident history, and never modified once written.
          </p>

          {showAudit &&
            (audit.length === 0 ? (
              <div className="empty-state empty-state--compact">
                <p>No account events recorded yet.</p>
              </div>
            ) : (
              <div className="card card-sm">
                <ul className="audit-list">
                  {audit.map((e) => (
                    <li key={e.id} className="audit-item">
                      <div className="audit-item__head">
                        <strong>{USER_AUDIT_LABELS[e.action] ?? e.action}</strong>
                        <span className="cell-meta">{formatDateTime(e.created_at)}</span>
                      </div>
                      <p className="audit-item__body">
                        {e.description ??
                          `${e.target_email ?? 'A user'}: ${e.old_value ?? '—'} → ${
                            e.new_value ?? '—'
                          }`}
                      </p>
                      <p className="audit-item__meta">
                        by {e.actor_name ?? 'System'}
                        {e.actor_email ? ` (${e.actor_email})` : ''}
                      </p>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
        </section>
      </main>
    </>
  );
}
